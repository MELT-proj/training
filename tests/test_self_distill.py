"""Online self-distillation (``melt/training/self_distill.py``).

The method rests on four claims, each pinned here:

1. The teacher is the frozen decoder on the text path -- the student's own
   decoder, with no adapter in the loop.
2. Response logits line up token for token on both paths, despite left-padded
   prompts of different lengths and audio of different lengths (MELT injects
   the audio frames and re-pads the merged sequence).
3. Only the adapter receives a gradient.
4. The loss is the divergence it claims (TRL's generalized JSD at beta 0/1 is
   forward/reverse KL).

Everything except the last class builds a tiny MELT model in-process, so it
needs no Hub access. The real config files are loaded too, so a broken overlay
fails here rather than at submission.
"""

import contextlib
import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from melt.modeling import MELTConfig, MELTForCausalLM
from melt.training import train as standard_train
from melt.training.config import trainer_args_dict
from melt.training.self_distill import (
    MELTSelfDistillTrainer,
    SelfDistillConfig,
    assign_instructions,
    check_adapter_only,
    check_adapter_only_gradients,
    distillation_loss,
    response_logits,
    response_mask,
)
from melt.training.train_self_distill import (
    _to_dotlist,
    install_self_distill_trainer,
    load_layered_config,
)
from transformers import LlamaConfig, Seq2SeqTrainingArguments, Wav2Vec2BertConfig


REPO_ROOT = Path(__file__).resolve().parents[1]

# Tiny decoder vocabulary: ids 64.. stand for the tokens MELT adds (audio
# placeholder, its bos/eos) plus the padding rows a real resize leaves behind.
VOCAB, LIMIT = 72, 64
AUDIO, AUDIO_BOS, AUDIO_EOS = 64, 65, 66
PAD, BOS, EOS = 0, 1, 2
FEATURE_DIM = 16


def _tiny_model(seed: int = 0) -> MELTForCausalLM:
    """A MELT model frozen the way an MA run is: encoder and decoder frozen, adapter trainable."""
    torch.manual_seed(seed)
    encoder = Wav2Vec2BertConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        feature_projection_input_dim=FEATURE_DIM, output_hidden_size=32,
    )
    decoder = LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2,
        pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS,
    )
    decoder.audio_token_id = AUDIO
    config = MELTConfig(
        audio_encoder_config=encoder,
        text_decoder_config=decoder,
        adapter_config={"_type": "mlp", "stack_factor": 2},
    )
    config.audio_encoder_config.max_audio_seq_len = 1500
    model = MELTForCausalLM(config, load_backbones=False)
    for module, trainable in (
        (model.audio_stack.encoder, False),
        (model.text_decoder, False),
        (model.audio_stack.adapter, True),
    ):
        for param in module.parameters():
            param.requires_grad = trainable
    # Deterministic forwards (no encoder dropout/layerdrop), as eval_when_frozen gives a real run.
    model.eval()
    return model


def _left_pad(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(r) for r in rows)
    ids = torch.tensor([[PAD] * (width - len(r)) + r for r in rows])
    mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])
    return ids, mask


# Two utterances whose prompts AND audio differ in length, so every padding
# path in the merge is exercised.
STUDENT_ROWS = [[BOS, 5, AUDIO_BOS, AUDIO, AUDIO_EOS, 6, 7], [BOS, 5, 9, AUDIO_BOS, AUDIO, AUDIO_EOS, 6, 7]]
TEACHER_ROWS = [[BOS, 5, 11, 12, 13, 6, 7], [BOS, 5, 9, 14, 15, 16, 17, 18, 6, 7]]
AUDIO_FRAMES = [20, 14]


def _batch(seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    student_ids, student_mask = _left_pad(STUDENT_ROWS)
    teacher_ids, teacher_mask = _left_pad(TEACHER_ROWS)
    features = torch.randn(2, max(AUDIO_FRAMES), FEATURE_DIM, generator=gen)
    features_mask = torch.zeros(2, max(AUDIO_FRAMES), dtype=torch.long)
    for i, n in enumerate(AUDIO_FRAMES):
        features_mask[i, :n] = 1
        features[i, n:] = 0.0
    return {
        "student_prompt_input_ids": student_ids,
        "student_prompt_attention_mask": student_mask,
        "teacher_prompt_input_ids": teacher_ids,
        "teacher_prompt_attention_mask": teacher_mask,
        "input_features": features,
        "features_attention_mask": features_mask,
    }


def _responses() -> tuple[torch.Tensor, torch.Tensor]:
    """Row 0 stops at EOS after three tokens; row 1 runs to the budget."""
    response = torch.tensor([[20, 21, EOS, PAD, PAD], [30, 31, 32, 33, 34]])
    return response, response_mask(response, torch.tensor([EOS]))


def _student_logits(model, batch, response, mask):
    return response_logits(
        model, batch["student_prompt_input_ids"], batch["student_prompt_attention_mask"],
        response, mask.long(), LIMIT,
        input_features=batch["input_features"], features_attention_mask=batch["features_attention_mask"],
    )


def _teacher_logits(model, batch, response, mask):
    return response_logits(
        model.text_decoder, batch["teacher_prompt_input_ids"], batch["teacher_prompt_attention_mask"],
        response, mask.long(), LIMIT,
    )


def _bare_trainer(model, distill: SelfDistillConfig) -> MELTSelfDistillTrainer:
    """The real compute_loss/_rollout, without HF Trainer's construction."""
    with patch.object(MELTSelfDistillTrainer, "__init__", lambda self, **kw: None):
        trainer = MELTSelfDistillTrainer()
    trainer.model = model
    trainer.distill = distill
    trainer.accelerator = SimpleNamespace(autocast=contextlib.nullcontext)
    trainer._vocab_limit = LIMIT
    trainer._eos_token_ids = [EOS]
    trainer._pad_token_id = PAD
    trainer._rng = random.Random(0)
    trainer._distill_stats = defaultdict(list)
    return trainer


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


class TestSelfDistillConfig:
    def test_defaults_are_on_policy_reverse_kl(self):
        cfg = SelfDistillConfig.from_config(None)
        assert (cfg.lmbda, cfg.loss, cfg.beta) == (1.0, "jsd", 1.0)

    def test_a_misspelt_key_is_an_error_not_a_silent_default(self):
        with pytest.raises(ValueError, match="lambda"):
            SelfDistillConfig.from_config({"lambda": 0.5})

    def test_cross_entropy_on_student_samples_is_refused(self):
        with pytest.raises(ValueError, match="lmbda=0"):
            SelfDistillConfig(loss="ce", lmbda=0.5)
        SelfDistillConfig(loss="ce", lmbda=0.0)

    @pytest.mark.parametrize(
        "field,value",
        [("beta", 1.5), ("lmbda", -0.1), ("top_p", 0.0), ("loss", "kl"), ("teacher_prompt", "echo"),
         ("gold_ce_weight", -0.5), ("translate_frac", 1.5), ("translate_targets", " , ")],
    )
    def test_out_of_range_values_are_refused(self, field, value):
        with pytest.raises(ValueError):
            SelfDistillConfig(**{field: value})

    def test_a_translate_mix_needs_the_mirror_teacher(self):
        with pytest.raises(ValueError, match="mirror"):
            SelfDistillConfig(translate_frac=0.5)
        cfg = SelfDistillConfig(translate_frac=0.5, teacher_prompt="mirror", translate_targets="EN, de")
        assert cfg.targets == ("en", "de")


class TestInstructionMix:
    TEXTS = [f"utterance number {i} says something" for i in range(2000)]

    def test_no_translation_keeps_every_task(self):
        tasks, tgt = assign_instructions(self.TEXTS, ["asr"] * 2000, ["de"] * 2000, 0.0, ("en", "de", "fr"))
        assert set(tasks) == {"asr"} and set(tgt) == {""}

    def test_share_targets_and_determinism(self):
        langs = ["de", "fr"] * 1000
        tasks, tgt = assign_instructions(self.TEXTS, ["asr"] * 2000, langs, 0.3, ("en", "de", "fr", "es", "it"))
        share = tasks.count("st") / len(tasks)
        assert 0.25 < share < 0.35
        # Never into the utterance's own language; every other target is used.
        for lang, task, t in zip(langs, tasks, tgt):
            assert (task == "st") == bool(t) and t != lang
        assert {t for t in tgt if t} == {"en", "de", "fr", "es", "it"}
        # The same utterance always gets the same instruction.
        assert (tasks, tgt) == assign_instructions(self.TEXTS, ["asr"] * 2000, langs, 0.3, ("en", "de", "fr", "es", "it"))


# ----------------------------------------------------------------------------
# Claim 1: the teacher is the frozen decoder on the text path
# ----------------------------------------------------------------------------


class TestTeacherIsTheTextPath:
    def test_teacher_logits_are_the_models_own_text_forward(self):
        """MELT's forward without audio is the decoder alone -- the same module the teacher calls."""
        model = _tiny_model()
        batch = _batch()
        ids, mask = batch["teacher_prompt_input_ids"], batch["teacher_prompt_attention_mask"]

        with torch.no_grad():
            via_melt = model(input_ids=ids, attention_mask=mask).logits
            via_decoder = model.text_decoder(input_ids=ids, attention_mask=mask).logits

        torch.testing.assert_close(via_melt, via_decoder)

    def test_the_adapter_moves_the_student_and_not_the_teacher(self):
        model = _tiny_model()
        batch = _batch()
        response, mask = _responses()
        with torch.no_grad():
            teacher_before = _teacher_logits(model, batch, response, mask)
            student_before = _student_logits(model, batch, response, mask)
            for param in model.audio_stack.adapter.parameters():
                param.add_(torch.randn_like(param))
            teacher_after = _teacher_logits(model, batch, response, mask)
            student_after = _student_logits(model, batch, response, mask)

        torch.testing.assert_close(teacher_before, teacher_after)
        assert not torch.allclose(student_before[mask], student_after[mask])


# ----------------------------------------------------------------------------
# Claim 2: response logits line up on both paths
# ----------------------------------------------------------------------------


class TestResponseAlignment:
    """Batched, padded logits must equal each utterance scored on its own."""

    @staticmethod
    def _single(batch: dict, i: int) -> dict:
        n = AUDIO_FRAMES[i]
        s_ids = torch.tensor([STUDENT_ROWS[i]])
        t_ids = torch.tensor([TEACHER_ROWS[i]])
        return {
            "student_prompt_input_ids": s_ids,
            "student_prompt_attention_mask": torch.ones_like(s_ids),
            "teacher_prompt_input_ids": t_ids,
            "teacher_prompt_attention_mask": torch.ones_like(t_ids),
            "input_features": batch["input_features"][i : i + 1, :n],
            "features_attention_mask": batch["features_attention_mask"][i : i + 1, :n],
        }

    @pytest.mark.parametrize("path", ["student", "teacher"])
    def test_batched_logits_match_unpadded_single_utterances(self, path):
        model = _tiny_model()
        batch = _batch()
        response, mask = _responses()
        logits_fn = _student_logits if path == "student" else _teacher_logits

        with torch.no_grad():
            batched = logits_fn(model, batch, response, mask)
            for i in range(2):
                length = int(mask[i].sum())
                single = logits_fn(model, self._single(batch, i), response[i : i + 1, :length], mask[i : i + 1, :length])
                torch.testing.assert_close(batched[i, :length], single[0], atol=1e-4, rtol=1e-4)

    def test_the_slice_is_the_tail_of_the_full_forward(self):
        """logits_to_keep keeps exactly the positions that predict the response."""
        model = _tiny_model()
        batch = _batch()
        response, mask = _responses()
        with torch.no_grad():
            full = model(
                input_ids=torch.cat([batch["student_prompt_input_ids"], response], 1),
                attention_mask=torch.cat([batch["student_prompt_attention_mask"], mask.long()], 1),
                input_features=batch["input_features"],
                features_attention_mask=batch["features_attention_mask"],
            ).logits
            sliced = _student_logits(model, batch, response, mask)
        length = response.shape[1]
        torch.testing.assert_close(sliced, full[:, -length - 1 : -1, :LIMIT])


class TestResponseMask:
    def test_keeps_up_to_and_including_the_first_eos(self):
        response = torch.tensor([[5, EOS, 7, EOS], [5, 6, 7, 8], [EOS, PAD, PAD, PAD]])
        mask = response_mask(response, torch.tensor([EOS, 3]))
        assert mask.tolist() == [
            [True, True, False, False],
            [True, True, True, True],  # truncated at the budget: kept whole
            [True, False, False, False],
        ]


class TestRecordStats:
    @staticmethod
    def _stats(response):
        trainer = _bare_trainer(None, SelfDistillConfig())
        mask = response_mask(response, torch.tensor([EOS]))
        logits = torch.randn(*response.shape, LIMIT)
        trainer._record_stats(logits, logits, response, mask, from_student=True)
        return trainer._distill_stats

    def test_the_longest_row_ending_on_eos_is_not_truncated(self):
        stats = self._stats(torch.tensor([[20, EOS, PAD], [30, 31, EOS]]))
        assert stats["truncated_frac"] == [0.0]
        assert stats["response_len"] == [2.5]

    def test_a_row_without_eos_is_truncated(self):
        stats = self._stats(_responses()[0])
        assert stats["truncated_frac"] == [0.5]

    def test_translate_rows_get_their_own_monitor(self):
        trainer = _bare_trainer(None, SelfDistillConfig())
        response = torch.tensor([[20, EOS, PAD], [30, 31, EOS]])
        mask = response_mask(response, torch.tensor([EOS]))
        logits = torch.randn(2, 3, LIMIT)
        trainer._record_stats(logits, logits, response, mask, True, torch.tensor([False, True]))
        stats = trainer._distill_stats
        assert stats["translate_rows"] == [0.5]
        assert stats["response_len_translate"] == [3.0]
        assert len(stats["teacher_logp_translate"]) == 1


# ----------------------------------------------------------------------------
# Claim 3: only the adapter learns
# ----------------------------------------------------------------------------


class TestAdapterOnly:
    @pytest.mark.parametrize(
        "distill",
        [
            SelfDistillConfig(lmbda=1.0, loss="jsd", beta=1.0, max_new_tokens=6),
            SelfDistillConfig(lmbda=1.0, loss="jsd", beta=0.5, max_new_tokens=6),
            SelfDistillConfig(lmbda=0.0, loss="jsd", beta=0.0, max_new_tokens=6),
            SelfDistillConfig(lmbda=0.0, loss="ce", max_new_tokens=6),
        ],
        ids=["on-policy-reverse-kl", "on-policy-jsd", "teacher-forward-kl", "teacher-ce"],
    )
    def test_the_real_compute_loss_only_reaches_the_adapter(self, distill):
        model = _tiny_model()
        model.train()  # as during training; the encoder stays frozen either way
        trainer = _bare_trainer(model, distill)

        loss = trainer.compute_loss(model, _batch())
        assert torch.isfinite(loss) and loss.requires_grad
        loss.backward()

        check_adapter_only_gradients(model)
        adapter_grads = [p.grad for p in model.audio_stack.adapter.parameters()]
        assert any(g is not None and g.abs().sum() > 0 for g in adapter_grads)
        assert all(p.grad is None for p in model.text_decoder.parameters())
        assert all(p.grad is None for p in model.audio_stack.encoder.parameters())

    def test_a_trainable_decoder_is_refused(self):
        model = _tiny_model()
        next(model.text_decoder.parameters()).requires_grad = True
        with pytest.raises(ValueError, match="adapter only"):
            check_adapter_only(model)

    def test_a_frozen_adapter_is_refused(self):
        model = _tiny_model()
        for param in model.audio_stack.adapter.parameters():
            param.requires_grad = False
        with pytest.raises(ValueError, match="trainable adapter"):
            check_adapter_only(model)

    def test_the_gradient_check_catches_a_stray_gradient(self):
        model = _tiny_model()
        for param in model.audio_stack.adapter.parameters():
            param.grad = torch.ones_like(param)
        decoder_param = next(model.text_decoder.parameters())
        decoder_param.grad = torch.zeros_like(decoder_param)
        with pytest.raises(RuntimeError, match="Non-adapter"):
            check_adapter_only_gradients(model)


class TestRollouts:
    @pytest.mark.parametrize("from_student", [True, False])
    def test_rollouts_never_emit_melt_added_tokens(self, from_student):
        """Generation is restricted to the support the loss scores (distill_vocab_limit)."""
        model = _tiny_model()
        trainer = _bare_trainer(model, SelfDistillConfig(max_new_tokens=40, temperature=1.0))
        torch.manual_seed(0)
        response, mask = trainer._rollout(_batch(), from_student=from_student)
        assert response.shape[1] <= 40
        assert int(response[mask].max()) < LIMIT
        assert (response[~mask] == PAD).all()

    def test_greedy_rollouts_are_deterministic(self):
        model = _tiny_model()
        trainer = _bare_trainer(model, SelfDistillConfig(max_new_tokens=8))
        first, _ = trainer._rollout(_batch(), from_student=False, greedy=True)
        second, _ = trainer._rollout(_batch(), from_student=False, greedy=True)
        assert torch.equal(first, second)


# ----------------------------------------------------------------------------
# Claim 4: the loss is the divergence it claims
# ----------------------------------------------------------------------------


class TestDistillationLoss:
    @staticmethod
    def _logits():
        gen = torch.Generator().manual_seed(0)
        student = torch.randn(2, 5, LIMIT, generator=gen)
        teacher = torch.randn(2, 5, LIMIT, generator=gen)
        response, mask = _responses()
        return student, teacher, response, mask

    def test_beta_zero_is_forward_kl_per_token(self):
        student, teacher, response, mask = self._logits()
        s, t = student[mask].log_softmax(-1), teacher[mask].log_softmax(-1)
        expected = (t.exp() * (t - s)).sum(-1).mean()
        got = distillation_loss(student, teacher, response, mask, SelfDistillConfig(beta=0.0))
        torch.testing.assert_close(got, expected)

    def test_beta_one_is_reverse_kl_per_token(self):
        student, teacher, response, mask = self._logits()
        s, t = student[mask].log_softmax(-1), teacher[mask].log_softmax(-1)
        expected = (s.exp() * (s - t)).sum(-1).mean()
        got = distillation_loss(student, teacher, response, mask, SelfDistillConfig(beta=1.0))
        torch.testing.assert_close(got, expected)

    def test_ce_is_hard_label_cross_entropy_on_real_tokens(self):
        student, teacher, response, mask = self._logits()
        expected = torch.nn.functional.cross_entropy(student[mask], response[mask])
        got = distillation_loss(student, teacher, response, mask, SelfDistillConfig(loss="ce", lmbda=0.0))
        torch.testing.assert_close(got, expected)

    def test_identical_distributions_cost_nothing(self):
        student, _, response, mask = self._logits()
        got = distillation_loss(student, student.clone(), response, mask, SelfDistillConfig(beta=0.5))
        assert got.abs() < 1e-6


# ----------------------------------------------------------------------------
# Entrypoint and config
# ----------------------------------------------------------------------------


class TestLayeredConfig:
    def test_defaults_then_base_then_overlay_then_cli(self, tmp_path):
        (tmp_path / "base.yaml").write_text("trainer:\n  seed: 1\n  eval_steps: 10\nrun:\n  exp_name: base\n")
        overlay = tmp_path / "sub" / "overlay.yaml"
        overlay.parent.mkdir()
        overlay.write_text("base_config: ../base.yaml\ntrainer:\n  seed: 2\ndistill:\n  beta: 0.5\n")

        cfg = load_layered_config(["--config", str(overlay), "--trainer.eval_steps", "30"])

        assert cfg.trainer.seed == 2  # overlay over base
        assert cfg.trainer.eval_steps == 30  # CLI over everything
        assert cfg.run.exp_name == "base"  # base over defaults
        assert cfg.distill.beta == 0.5
        assert "base_config" not in cfg
        assert Path(cfg.run.base_config) == (tmp_path / "base.yaml").resolve()

    def test_cli_parsing_matches_the_standard_entrypoint(self, tmp_path, monkeypatch):
        from melt.training.config import parse_args_and_load_config

        config = tmp_path / "c.yaml"
        config.write_text("trainer:\n  seed: 1\n")
        cli = ["--trainer.seed", "7", "--run.exp_name=x", "--trainer.do_eval", "--data.prompt_template", "'{audio_token}'"]
        monkeypatch.setattr("sys.argv", ["train.py", "--config", str(config), *cli])

        standard = parse_args_and_load_config()
        layered = load_layered_config(["--config", str(config), *cli])

        assert OmegaConf.to_container(layered.trainer) == OmegaConf.to_container(standard.trainer)
        assert layered.run.exp_name == standard.run.exp_name
        assert layered.data.prompt_template == standard.data.prompt_template
        assert _to_dotlist(["--a.b", "1", "--c"]) == ["a.b=1", "c=true"]

    def test_the_shipped_overlay_builds_a_valid_run(self, monkeypatch):
        """projects/ablation-campaign/self-distill.yaml layered over the MA config."""
        monkeypatch.setenv("OUTPUT_DIR", "/tmp/outputs")
        monkeypatch.setenv("LOCAL_DATASETS_DIR", "/tmp/shar")
        cfg = load_layered_config(["--config", str(REPO_ROOT / "projects/ablation-campaign/self-distill.yaml")])

        SelfDistillConfig.from_config(cfg.distill)
        # Data and model come from the campaign config, untouched.
        assert cfg.data.train_ds.input_cfg
        assert cfg.model.decoder.freeze and cfg.model.encoder.freeze and not cfg.model.adapter.freeze
        # The student trains instruction-free; WER is scored with an instruction.
        assert cfg.data.prompt_template_selection == "custom"
        assert cfg.data.prompt_template == "{audio_token}"
        assert "{audio_token}" in cfg.data.validation_ds.prompt_template
        assert cfg.data.validation_ds.prompt_template != cfg.data.prompt_template
        assert cfg.run.memory_preallocation is False

        # Every trainer key the overlay sets must reach TrainingArguments:
        # trainer_args_dict drops unknown keys with only a warning (that is how
        # warmup_ratio silently stopped working under transformers 5).
        import dataclasses

        import yaml

        overlay = yaml.safe_load((REPO_ROOT / "projects/ablation-campaign/self-distill.yaml").read_text())
        known = {f.name for f in dataclasses.fields(Seq2SeqTrainingArguments)}
        assert set(overlay["trainer"]) <= known, set(overlay["trainer"]) - known

        cfg.trainer.bf16 = False  # CPU-only test runner
        Seq2SeqTrainingArguments(**trainer_args_dict(cfg))


class TestTrainerSwap:
    def test_train_main_constructs_the_self_distill_trainer(self, monkeypatch):
        # Registered first so monkeypatch restores the real class afterwards.
        monkeypatch.setattr(standard_train, "MELTTrainer", standard_train.MELTTrainer)
        install_self_distill_trainer()
        assert standard_train.MELTTrainer is MELTSelfDistillTrainer

    def test_a_train_main_that_stops_using_the_name_is_detected(self, monkeypatch):
        monkeypatch.setattr(standard_train, "MELTTrainer", standard_train.MELTTrainer)
        monkeypatch.setattr(standard_train, "main", lambda cfg: None)
        with pytest.raises(RuntimeError, match="no longer constructs"):
            install_self_distill_trainer()


class TestTrainerEndToEnd:
    """The real MELTSelfDistillTrainer, built by HF's Trainer on CPU around the tiny model.

    Covers what the bare-trainer tests skip: construction checks, HF's own
    training_step (accelerator backward, loss scaling), the post-backward
    gradient check, log() aggregation and the eval prediction_step.
    """

    @staticmethod
    def _trainer(tmp_path, **distill):
        processor = SimpleNamespace(
            audio_token_id=AUDIO,
            audio_bos_token_id=AUDIO_BOS,
            audio_eos_token_id=AUDIO_EOS,
            tokenizer=SimpleNamespace(eos_token_id=EOS, pad_token_id=PAD),
        )
        config = OmegaConf.create(
            {
                "run": {"memory_preallocation": False},
                "data": {
                    "apply_chat_template": True,
                    "prompt_template_selection": "custom",
                    "prompt_template": "{audio_token}",
                },
                "optimization": {"adapter_lr": 1e-3, "adam_beta1": 0.9, "adam_beta2": 0.95},
                "distill": {"max_new_tokens": 6, **distill},
            }
        )
        args = Seq2SeqTrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to=[], predict_with_generate=True,
            gradient_accumulation_steps=2, seed=0,
        )
        return MELTSelfDistillTrainer(model=_tiny_model(), args=args, config=config, processor=processor)

    def test_training_step_then_log(self, tmp_path):
        trainer = self._trainer(tmp_path, lmbda=1.0)
        # compute_loss returns a per-micro-batch mean, so HF must divide by the
        # accumulation steps itself -- which it reads off this attribute, set
        # by its training loop.
        assert trainer.model_accepts_loss_kwargs is False
        trainer.current_gradient_accumulation_steps = 2

        loss = trainer.training_step(trainer.model, _batch())

        assert torch.isfinite(loss)
        assert trainer._gradients_checked
        logs = {"loss": float(loss)}
        with patch("transformers.Trainer.log"):
            trainer.log(logs)
        assert logs["distill/on_policy_frac"] == 1.0
        assert 1 <= logs["distill/response_len"] <= 6
        assert logs["distill/teacher_logp"] < 0

    def test_eval_loss_is_the_alignment_gap(self, tmp_path):
        trainer = self._trainer(tmp_path)
        batch = _batch()
        prompt_ids, prompt_mask = batch["student_prompt_input_ids"], batch["student_prompt_attention_mask"]
        gold = torch.cat([prompt_ids, torch.tensor([[20, EOS], [21, EOS]])], dim=1)
        inputs = {
            **batch,
            "input_ids": gold,
            "attention_mask": torch.cat([prompt_mask, torch.ones(2, 2, dtype=torch.long)], dim=1),
            "labels": torch.where(torch.arange(gold.shape[1]) >= prompt_ids.shape[1], gold, -100),
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
        }
        trainer.model.eval()

        loss, predictions, labels = trainer.prediction_step(
            trainer.model, inputs, prediction_loss_only=False, max_new_tokens=4
        )

        response, mask = trainer._rollout(trainer._prepare_inputs(_batch()), from_student=False, greedy=True)
        with torch.no_grad():
            expected = distillation_loss(
                _student_logits(trainer.model, _batch(), response, mask),
                _teacher_logits(trainer.model, _batch(), response, mask),
                response, mask, SelfDistillConfig(beta=0.0),
            )
        torch.testing.assert_close(loss, expected)
        assert predictions.shape == (2, 4)

    @staticmethod
    def _gold_inputs(batch: dict) -> dict:
        """The standard MA keys: the student prompt followed by a two-token gold transcript."""
        prompt_ids, prompt_mask = batch["student_prompt_input_ids"], batch["student_prompt_attention_mask"]
        gold = torch.cat([prompt_ids, torch.tensor([[20, EOS], [21, EOS]])], dim=1)
        return {
            "input_ids": gold,
            "attention_mask": torch.cat([prompt_mask, torch.ones(2, 2, dtype=torch.long)], dim=1),
            "labels": torch.where(torch.arange(gold.shape[1]) >= prompt_ids.shape[1], gold, -100),
        }

    def test_gold_ce_anchor_adds_the_ma_loss(self, tmp_path):
        """distill.gold_ce_weight adds weight x the standard MA cross-entropy on the gold transcript."""
        batch = _batch()
        gold_inputs = self._gold_inputs(batch)
        losses = {}
        for weight in (0.0, 0.5):
            trainer = self._trainer(tmp_path, lmbda=0.0, loss="ce", temperature=0.0, gold_ce_weight=weight)
            losses[weight] = trainer.compute_loss(trainer.model, {**batch, **gold_inputs})
        with torch.no_grad():
            gold_ce = trainer.model(
                **gold_inputs, input_features=batch["input_features"],
                features_attention_mask=batch["features_attention_mask"],
            ).loss

        torch.testing.assert_close(losses[0.5], losses[0.0] + 0.5 * gold_ce)
        assert trainer._distill_stats["gold_ce"] == [pytest.approx(gold_ce.item())]

    def test_training_step_with_the_gold_anchor(self, tmp_path):
        """Two forwards, one backward: the adapter alone still gets the gradient."""
        trainer = self._trainer(tmp_path, lmbda=1.0, gold_ce_weight=0.5)
        trainer.current_gradient_accumulation_steps = 2
        batch = _batch()

        loss = trainer.training_step(trainer.model, {**batch, **self._gold_inputs(batch)})

        assert torch.isfinite(loss)
        assert trainer._gradients_checked
        logs = {"loss": float(loss)}
        with patch("transformers.Trainer.log"):
            trainer.log(logs)
        assert logs["distill/gold_ce"] > 0

    def test_training_step_with_translate_rows(self, tmp_path):
        """The instruction-mix flag rides in the batch without reaching a forward."""
        trainer = self._trainer(tmp_path, lmbda=1.0)
        trainer.current_gradient_accumulation_steps = 2
        batch = {**_batch(), "distill_translate": torch.tensor([False, True])}

        loss = trainer.training_step(trainer.model, batch)

        assert torch.isfinite(loss)
        logs = {"loss": float(loss)}
        with patch("transformers.Trainer.log"):
            trainer.log(logs)
        assert logs["distill/translate_rows"] == 0.5
        assert "distill/teacher_logp_translate" in logs

    def test_memory_preallocation_is_refused(self, tmp_path):
        processor = SimpleNamespace(
            audio_token_id=AUDIO, audio_bos_token_id=AUDIO_BOS, audio_eos_token_id=AUDIO_EOS,
            tokenizer=SimpleNamespace(eos_token_id=EOS, pad_token_id=PAD),
        )
        config = OmegaConf.create(
            {
                "run": {"memory_preallocation": True},
                "data": {"apply_chat_template": True, "prompt_template_selection": "custom",
                         "prompt_template": "{audio_token}"},
                "optimization": {"adam_beta1": 0.9, "adam_beta2": 0.95},
            }
        )
        args = Seq2SeqTrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[])
        with pytest.raises(ValueError, match="memory_preallocation"):
            MELTSelfDistillTrainer(model=_tiny_model(), args=args, config=config, processor=processor)


# ----------------------------------------------------------------------------
# Prompts, with a real tokenizer (Hub)
# ----------------------------------------------------------------------------


class TestVocabLimit:
    @staticmethod
    def _processor(eos, pad):
        return SimpleNamespace(
            audio_token_id=100, audio_bos_token_id=101, audio_eos_token_id=102,
            tokenizer=SimpleNamespace(eos_token_id=eos, pad_token_id=pad),
        )

    def test_a_pad_token_added_after_the_audio_tokens_is_allowed(self):
        from melt.training.self_distill import distill_vocab_limit

        assert distill_vocab_limit(self._processor(eos=7, pad=103)) == 100

    def test_an_eos_past_the_limit_is_refused(self):
        from melt.training.self_distill import distill_vocab_limit

        with pytest.raises(ValueError, match="eos_token_id"):
            distill_vocab_limit(self._processor(eos=103, pad=3))


@pytest.mark.hub
class TestPrompts:
    """Student and teacher prompts differ in the user turn, and only there."""

    @staticmethod
    def _processor():
        from melt.modeling.processing_melt import MELTProcessor
        from transformers import AutoTokenizer
        from transformers.feature_extraction_utils import FeatureExtractionMixin

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        specials = {"audio_token": "<|audio|>", "audio_bos_token": "<|audio_bos|>", "audio_eos_token": "<|audio_eos|>"}
        tokenizer.add_special_tokens({"additional_special_tokens": list(specials.values())})
        for attr, token in specials.items():
            setattr(tokenizer, attr, token)
        # Prompts never touch audio; any extractor satisfies the processor's type check.
        return MELTProcessor(feature_extractor=FeatureExtractionMixin(), tokenizer=tokenizer)

    def test_teacher_prompt_is_the_student_prompt_with_the_transcript_for_the_audio(self):
        from melt.training.self_distill import build_distill_prompts

        processor = self._processor()
        texts = ["hello world", "a much longer second transcript to force padding"]
        out = build_distill_prompts(processor, texts, ["asr", "asr"], ["en", "en"], "{audio_token}")
        decode = processor.tokenizer.decode
        audio_block = "<|audio_bos|><|audio|><|audio_eos|>"

        for i, text in enumerate(texts):
            student = decode(out["student_prompt_input_ids"][i][out["student_prompt_attention_mask"][i].bool()])
            teacher = decode(out["teacher_prompt_input_ids"][i][out["teacher_prompt_attention_mask"][i].bool()])
            assert audio_block in student and text not in student
            assert student.replace(audio_block, text) == teacher
        # Left-padded, so a response appended to any row follows its last real token.
        assert out["teacher_prompt_attention_mask"][:, -1].all()
        assert out["student_prompt_attention_mask"][:, -1].all()

    def test_mirror_gives_the_teacher_the_student_instruction(self):
        """teacher_prompt='mirror': same instruction on both paths, transcript where the audio is."""
        from melt.training.self_distill import build_distill_prompts

        processor = self._processor()
        template = "Repeat the following content exactly, word for word, and write nothing else.\n\n{audio_token}"
        texts = ["hello world", "a much longer second transcript to force padding"]
        decode = processor.tokenizer.decode
        audio_block = "<|audio_bos|><|audio|><|audio_eos|>"

        mirror = build_distill_prompts(processor, texts, ["asr", "asr"], ["en", "en"], template, "mirror")
        bare = build_distill_prompts(processor, texts, ["asr", "asr"], ["en", "en"], template, "bare")
        for i, text in enumerate(texts):
            student = decode(mirror["student_prompt_input_ids"][i][mirror["student_prompt_attention_mask"][i].bool()])
            teacher = decode(mirror["teacher_prompt_input_ids"][i][mirror["teacher_prompt_attention_mask"][i].bool()])
            bare_teacher = decode(bare["teacher_prompt_input_ids"][i][bare["teacher_prompt_attention_mask"][i].bool()])
            assert "Repeat the following content" in student
            assert student.replace(audio_block, text) == teacher
            assert "Repeat the following content" not in bare_teacher and text in bare_teacher

    def test_a_translate_instruction_reaches_both_paths(self):
        """Instruction mix: the st row asks both paths to translate, into the drawn language."""
        from melt.training.self_distill import build_distill_prompts

        processor = self._processor()
        template = {
            "asr": "Repeat the following content exactly, word for word, and write nothing else.\n\n{audio_token}",
            "st": "Translate the following content into {tgt_lang}, and write nothing else.\n\n{audio_token}",
        }
        texts = ["hallo welt", "ein zweiter satz"]
        out = build_distill_prompts(
            processor, texts, ["asr", "st"], ["de", "de"], template, "mirror", tgt_langs=["", "fr"]
        )
        decode = processor.tokenizer.decode
        audio_block = "<|audio_bos|><|audio|><|audio_eos|>"
        rows = []
        for i, text in enumerate(texts):
            student = decode(out["student_prompt_input_ids"][i][out["student_prompt_attention_mask"][i].bool()])
            teacher = decode(out["teacher_prompt_input_ids"][i][out["teacher_prompt_attention_mask"][i].bool()])
            assert student.replace(audio_block, text) == teacher
            rows.append(student)
        assert "Repeat the following content" in rows[0]
        assert "Translate the following content into French" in rows[1]
        assert out["distill_translate"].tolist() == [False, True]

    def test_student_prompt_is_a_prefix_of_the_gold_training_sequence(self):
        """The rollout starts from exactly the context MA training conditions on."""
        from melt.training.data.audio.lhotse.helpers import apply_chat_template_to_texts
        from melt.training.self_distill import build_distill_prompts

        processor = self._processor()
        tokenizer = processor.tokenizer
        (full,) = apply_chat_template_to_texts(
            ["hello world"], ["asr"], ["en"], tokenizer=tokenizer, audio_token="<|audio|>",
            prompt_template="{audio_token}", prompt_template_selection="custom",
        )
        gold = tokenizer(processor._surround_bos_eos_mm_tokens(full))["input_ids"]
        out = build_distill_prompts(processor, ["hello world"], ["asr"], ["en"], "{audio_token}")
        prompt = out["student_prompt_input_ids"][0].tolist()

        assert gold[: len(prompt)] == prompt

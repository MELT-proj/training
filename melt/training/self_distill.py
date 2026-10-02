"""Online self-distillation for modality alignment (AZeroS-style SIFT, on-policy).

AZeroS (arXiv 2601.06086) trains the adapter on targets the frozen decoder
generates when it is handed the *transcript* as text, with no instruction, and
stores those targets offline. This module does the same thing online, which
removes the per-decoder dataset:

* **teacher** = the frozen text decoder on the text path: the chat-templated
  transcript goes straight into ``model.text_decoder``. No adapter, no audio,
  and no copy of the weights -- it is the very module the student decodes with.
* **student** = the full speech LLM on the audio path, instruction-free prompt
  (``{audio_token}`` only). Only the adapter trains.

Each micro-batch draws one response per utterance -- from the student
(on-policy, probability ``lmbda``) or from the teacher (the offline AZeroS
target, regenerated on the fly) -- and scores it under both prompts. Because
both share one tokenizer and one decoder, the response tokens sit at the same
positions on both paths and the per-token loss is a full-vocabulary divergence
between two distributions, not a sampled scalar reward. The divergence is TRL's
generalized JSD (``GKDTrainer.generalized_jsd_loss``): ``beta=0`` is forward
KL(teacher || student), ``beta=1`` reverse KL(student || teacher). With
``lmbda=0`` and ``loss="ce"`` the objective is exactly AZeroS's (hard-label
cross-entropy on a teacher response), which is the control this method has to
beat.

What this module deliberately does not do: train the decoder or the encoder
(the teacher would drift with the student, see ``check_adapter_only``), run
under FSDP/DeepSpeed (generation and the teacher forward call submodules
directly), or run with LoRA on the decoder (the teacher would no longer be the
original LLM).
"""

import random
from collections import defaultdict
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ..logging_utils import get_logger
from .data.audio.lhotse import (
    FallbackDataset,
    MELTDataCollator,
    SpeechToTextDataset,
    get_train_dataloader_from_config,
)
from .data.audio.lhotse.helpers import _get_config_value, apply_chat_template_to_texts
from .trainer import MELTTrainer


logger = get_logger(__name__)

# Batch keys this module adds on top of the standard MELT batch.
DISTILL_PROMPT_KEYS = (
    "student_prompt_input_ids",
    "student_prompt_attention_mask",
    "teacher_prompt_input_ids",
    "teacher_prompt_attention_mask",
)

# Keys of the standard (gold-transcript) training batch that the distillation
# loss does not read. Dropped explicitly so nothing gold-derived can leak into
# the objective by accident.
_GOLD_ONLY_KEYS = ("input_ids", "attention_mask", "labels", "audio_lengths")


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SelfDistillConfig:
    """The ``distill:`` section of a training config.

    Attributes:
        lmbda: Probability that a micro-batch's responses are sampled from the
            student (on-policy). ``0`` always uses teacher responses (online
            AZeroS); ``1`` is fully on-policy. Same meaning as GKD's ``lmbda``.
        loss: ``"jsd"`` for TRL's generalized JSD over the full vocabulary, or
            ``"ce"`` for hard-label cross-entropy on the response tokens.
            ``"ce"`` requires ``lmbda == 0``: cross-entropy on the student's own
            samples only sharpens the student and carries no teacher signal.
        beta: JSD interpolation, ``0`` = forward KL(teacher || student),
            ``1`` = reverse KL(student || teacher). Ignored by ``"ce"``.
        max_new_tokens: Response length budget for every rollout, training and
            eval. A truncated response is still distilled on its prefix.
        temperature: Sampling temperature for training rollouts; ``0`` decodes
            greedily. Eval rollouts are always greedy.
        top_p: Nucleus mass for training rollouts. ``1.0`` samples from the full
            distribution, which is what the on-policy objective assumes.
        eval_alignment_gap: Replace ``eval_loss`` with the forward KL of the
            student from the teacher on greedy teacher responses (see
            ``MELTSelfDistillTrainer.prediction_step``).
    """

    lmbda: float = 1.0
    loss: str = "jsd"
    beta: float = 1.0
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    eval_alignment_gap: bool = True

    def __post_init__(self):
        if self.loss not in ("jsd", "ce"):
            raise ValueError(f"distill.loss must be 'jsd' or 'ce', got {self.loss!r}")
        if not 0.0 <= self.lmbda <= 1.0:
            raise ValueError(f"distill.lmbda must be in [0, 1], got {self.lmbda}")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"distill.beta must be in [0, 1], got {self.beta}")
        if self.loss == "ce" and self.lmbda != 0.0:
            raise ValueError(
                "distill.loss='ce' needs distill.lmbda=0: cross-entropy on the student's own "
                "samples is self-training, not distillation. Use loss='jsd' for on-policy runs."
            )
        if self.max_new_tokens < 1:
            raise ValueError(f"distill.max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.temperature < 0.0:
            raise ValueError(f"distill.temperature must be >= 0, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"distill.top_p must be in (0, 1], got {self.top_p}")

    @classmethod
    def from_config(cls, section: DictConfig | dict | None) -> "SelfDistillConfig":
        """Build from the ``distill:`` config section; unknown keys are an error.

        A misspelt knob that silently kept its default is how an arm stops being
        the ablation it claims to be, so typos fail here rather than at the end.
        """
        if section is None:
            return cls()
        values = OmegaConf.to_container(section, resolve=True) if isinstance(section, DictConfig) else dict(section)
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown distill.* key(s) {unknown}; expected a subset of {sorted(known)}")
        return cls(**values)


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------


def build_distill_prompts(
    processor,
    texts: list[str],
    tasks: list[str],
    langs: list[str],
    student_prompt_template: str | dict[str, str],
) -> dict[str, torch.Tensor]:
    """Tokenise the student and teacher generation prompts for one batch.

    Both prompts go through the same chat template and the same tokenizer call,
    and end with the same assistant header, so a response appended to either is
    tokenised identically. They differ only in the user turn: the student's is
    the instruction-free audio template (``{audio_token}``), the teacher's is
    the transcript. Tokenised and left-padded exactly as ``MELTDataCollator``
    tokenises the eval generation prompt, so the student rollout sees the same
    ids an evaluation would.

    Args:
        processor: The run's ``MELTProcessor``.
        texts: Transcripts, one per kept utterance, in batch order.
        tasks: Task tags, aligned with *texts*.
        langs: Language codes, aligned with *texts*.
        student_prompt_template: The training ``data.prompt_template``.

    Returns:
        The four ``DISTILL_PROMPT_KEYS`` tensors.
    """
    tokenizer = processor.tokenizer
    _, student_prompts = apply_chat_template_to_texts(
        texts,
        tasks,
        langs,
        tokenizer=tokenizer,
        audio_token=processor.audio_token,
        prompt_template=student_prompt_template,
        prompt_template_selection="custom",
        return_prompts=True,
    )
    teacher_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for text in texts
    ]
    student = tokenizer(
        [processor._surround_bos_eos_mm_tokens(p) for p in student_prompts],
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
    teacher = tokenizer(teacher_prompts, padding=True, padding_side="left", return_tensors="pt")
    return {
        "student_prompt_input_ids": student["input_ids"],
        "student_prompt_attention_mask": student["attention_mask"],
        "teacher_prompt_input_ids": teacher["input_ids"],
        "teacher_prompt_attention_mask": teacher["attention_mask"],
    }


def _require_instruction_free_setup(config) -> None:
    """The student prompt must be the fixed, chat-templated custom template."""
    if not bool(_get_config_value(config, "apply_chat_template", False)):
        raise ValueError("Self-distillation needs data.apply_chat_template: true (the teacher is a chat model).")
    if str(_get_config_value(config, "prompt_template_selection", "random")) != "custom":
        raise ValueError(
            "Self-distillation needs data.prompt_template_selection: custom. The student prompt is "
            "the fixed instruction-free template; a randomly drawn task instruction is what AZeroS "
            "shows hurts generalisation (SIT vs SIFT)."
        )


class SelfDistillDataset(SpeechToTextDataset):
    """``SpeechToTextDataset`` plus the student and teacher generation prompts.

    The parent builds the usual batch (audio features, gold-transcript ids and
    labels). The transcripts it keeps -- after skipping cuts whose audio or text
    failed -- are only visible inside ``_apply_chat_template``, so they are
    captured there and turned into the two prompt tensors afterwards.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _require_instruction_free_setup(self.config)
        self._batch_meta: tuple[list[str], list[str], list[str]] | None = None

    def _apply_chat_template(self, texts, tasks, langs, src_langs=None, tgt_langs=None):
        self._batch_meta = (list(texts), list(tasks), list(langs))
        return super()._apply_chat_template(texts, tasks, langs, src_langs, tgt_langs)

    def __getitem__(self, cuts):
        self._batch_meta = None
        batch = super().__getitem__(cuts)
        if batch is None:
            return None
        texts, tasks, langs = self._batch_meta
        batch.update(build_distill_prompts(self.processor, texts, tasks, langs, self.prompt_template))
        return batch


class SelfDistillEvalCollator(MELTDataCollator):
    """``MELTDataCollator`` plus the prompts the alignment-gap metric needs.

    The eval generation prompt (for WER) follows ``validation_ds`` as usual and
    may carry an instruction; the distillation prompts always use the
    *training* template, so the metric scores the objective that was trained.
    """

    def __init__(self, processor, config, student_prompt_template, is_train: bool = False):
        super().__init__(processor=processor, config=config, is_train=is_train)
        self.student_prompt_template = student_prompt_template

    def __call__(self, items: list[dict]) -> dict:
        batch = super().__call__(items)
        valid = [it for it in items if not it.get("__invalid__", False)]
        batch.update(
            build_distill_prompts(
                self.processor,
                [it["text"] for it in valid],
                [it.get("task", "asr") for it in valid],
                [it.get("lang", "") for it in valid],
                self.student_prompt_template,
            )
        )
        return batch


# ----------------------------------------------------------------------------
# Rollouts and loss
# ----------------------------------------------------------------------------


def check_adapter_only(model: torch.nn.Module) -> None:
    """Raise unless the adapter holds every trainable parameter (and has some).

    The teacher *is* the student's decoder, so training the decoder would move
    the target with the student; training the encoder would leave the method's
    "adapter alone aligns the modalities" claim untested. LoRA is excluded for
    the first reason.
    """
    adapter_ids = {id(p) for p in model.audio_stack.adapter.parameters()}
    stray = [name for name, p in model.named_parameters() if p.requires_grad and id(p) not in adapter_ids]
    if stray:
        raise ValueError(
            f"Self-distillation trains the adapter only, but {len(stray)} other parameter(s) are trainable, "
            f"e.g. {stray[:3]}. Set model.encoder.freeze and model.decoder.freeze to true and disable LoRA."
        )
    if not any(p.requires_grad for p in model.audio_stack.adapter.parameters()):
        raise ValueError("Self-distillation needs a trainable adapter (model.adapter.freeze: false).")


def check_adapter_only_gradients(model: torch.nn.Module) -> None:
    """Raise if any non-adapter parameter holds a gradient, or the adapter holds none."""
    adapter_ids = {id(p) for p in model.audio_stack.adapter.parameters()}
    stray = [name for name, p in model.named_parameters() if p.grad is not None and id(p) not in adapter_ids]
    if stray:
        raise RuntimeError(f"Non-adapter parameters received gradients: {stray[:5]}")
    if all(p.grad is None for p in model.audio_stack.adapter.parameters()):
        raise RuntimeError("The adapter received no gradient from the distillation loss.")


def distill_vocab_limit(processor) -> int:
    """First token id MELT added to the decoder's vocabulary.

    The audio placeholder tokens, and any embedding rows past them, were never
    in the LLM's own output distribution: they are randomly initialised rows of
    a tied, frozen head. Slicing logits to ``[:limit]`` (and suppressing those
    ids in generation) is what makes the teacher exactly the original LLM, and
    keeps the student from being scored on mass it can only put on placeholders.
    """
    limit = min(processor.audio_token_id, processor.audio_bos_token_id, processor.audio_eos_token_id)
    # The stop token must be inside the teacher's distribution. The pad id may lie
    # past the limit (Qwen's is a token MELT adds): it only fills masked positions.
    eos = getattr(processor.tokenizer, "eos_token_id", None)
    if eos is not None and eos >= limit:
        raise ValueError(f"tokenizer.eos_token_id={eos} lies past MELT's added tokens (limit {limit}).")
    return limit


def response_mask(response: torch.Tensor, eos_token_ids: torch.Tensor) -> torch.Tensor:
    """Mask of real response tokens: everything up to and including the first EOS.

    ``generate()`` pads finished rows with the pad id; those positions, and any
    token after the first EOS, are excluded. A row that never emitted EOS was
    truncated at ``max_new_tokens`` and is kept whole.
    """
    is_eos = torch.isin(response, eos_token_ids).long()
    eos_before = is_eos.cumsum(dim=1) - is_eos
    return eos_before == 0


def response_logits(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    response_ids: torch.Tensor,
    response_attention: torch.Tensor,
    vocab_limit: int,
    **model_kwargs,
) -> torch.Tensor:
    """Logits that predict each response token, shape ``(batch, response_len, vocab_limit)``.

    The response is appended to the left-padded prompt, so it occupies the last
    ``L`` positions of every row -- also after MELT injects the audio frames,
    because ``_inject_tensor`` left-pads the merged sequence. ``logits_to_keep=L+1``
    therefore returns the last prompt position plus the response, and dropping
    the final position leaves exactly one predicting logit per response token.
    It also skips the LM head over the audio frames and the prompt.
    """
    response_len = response_ids.shape[1]
    out = model(
        input_ids=torch.cat([prompt_ids, response_ids], dim=1),
        attention_mask=torch.cat([prompt_mask, response_attention], dim=1),
        logits_to_keep=response_len + 1,
        use_cache=False,
        **model_kwargs,
    )
    logits = out.logits[:, :-1, :vocab_limit]
    if logits.shape[1] != response_len:
        raise RuntimeError(f"Expected {response_len} response logits, got {logits.shape[1]}")
    return logits


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_ids: torch.Tensor,
    mask: torch.Tensor,
    cfg: SelfDistillConfig,
) -> torch.Tensor:
    """Token-mean distillation loss over the real response tokens.

    Logits are gathered to the ``(N, vocab)`` real tokens before the divergence,
    so padding never materialises a full-vocabulary log-softmax.
    """
    student = student_logits[mask].float()
    if cfg.loss == "ce":
        return F.cross_entropy(student, response_ids[mask])

    teacher = teacher_logits[mask].float()
    # labels=None with "batchmean" divides by the first dimension, i.e. the
    # number of real tokens: a per-token mean, summed over the vocabulary.
    return _generalized_jsd_loss()(student, teacher, labels=None, beta=cfg.beta, reduction="batchmean")


def _generalized_jsd_loss():
    """TRL's GKD divergence, imported lazily.

    Only this objective, not the standard training path, depends on TRL; see
    pyproject.toml's ``distill`` extra for the pin and why it is 0.29.x.
    """
    from trl.experimental.gkd import GKDTrainer

    return GKDTrainer.generalized_jsd_loss


# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------


class MELTSelfDistillTrainer(MELTTrainer):
    """``MELTTrainer`` whose objective is online self-distillation.

    Data loading, evaluation, optimizer groups, checkpointing and logging are the
    parent's. Only the loss changes, plus two checks the method depends on: the
    adapter is the only trainable module (at construction), and the only one
    that actually receives a gradient (after the first backward).
    """

    def __init__(self, model, args, config: DictConfig, processor, **kwargs):
        self.distill = SelfDistillConfig.from_config(config.get("distill"))
        _require_instruction_free_setup(config.data)
        super().__init__(model=model, args=args, config=config, processor=processor, **kwargs)

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            raise ValueError(
                "Self-distillation runs under DDP only: rollouts and the teacher forward call "
                "submodules directly, which sharded parameters do not support."
            )
        if self._memory_preallocation:
            raise ValueError(
                "run.memory_preallocation measures the gold-transcript forward, not the distillation "
                "step (rollout + two forwards), so its numbers would mislead. Set it to false."
            )
        check_adapter_only(self.model)
        # Import now, so a missing TRL fails at startup and not at step 1.
        _generalized_jsd_loss()

        # compute_loss returns a per-micro-batch token mean. MELT's forward takes
        # **kwargs, so HF would otherwise assume the loss is already normalised
        # by num_items_in_batch (counted from the *gold* labels) and skip the
        # division by the accumulation steps.
        self.model_accepts_loss_kwargs = False

        self._vocab_limit = distill_vocab_limit(processor)
        eos = self.model.text_decoder.generation_config.eos_token_id
        eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        if processor.tokenizer.eos_token_id is not None:
            eos.append(processor.tokenizer.eos_token_id)
        self._eos_token_ids = sorted({int(e) for e in eos if e is not None})
        self._pad_token_id = processor.tokenizer.pad_token_id
        self._rng = random.Random(int(args.seed))
        self._gradients_checked = False
        self._distill_stats: dict[str, list[float]] = defaultdict(list)

        if self._eval_collator is not None:
            self._eval_collator = SelfDistillEvalCollator(
                processor=processor,
                config=self._eval_collator.config,
                student_prompt_template=_get_config_value(config.data, "prompt_template", None),
            )

        logger.info(
            "Self-distillation: %s (eos ids %s, vocab limit %d)",
            self.distill,
            self._eos_token_ids,
            self._vocab_limit,
        )

    # -- data ----------------------------------------------------------------

    def get_train_dataloader(self) -> DataLoader:
        """``MELTTrainer.get_train_dataloader`` with the prompt-emitting dataset."""
        dataset = FallbackDataset(
            SelfDistillDataset(
                processor=self.processor,
                config=self.config.data,
                is_train=True,
                return_labels=True,
                return_langs=True,
            )
        )
        dataloader = get_train_dataloader_from_config(
            data_config=self.config.data,
            dataset=dataset,
            global_rank=self._global_rank,
            world_size=self._world_size,
        )
        self._train_dataloader_ref = dataloader
        if self._lhotse_resume_from is not None:
            self._restore_sampler_state(dataloader)
        return dataloader

    # -- rollouts ------------------------------------------------------------

    def _generation_kwargs_for(self, greedy: bool, logits_dim: int) -> dict:
        kwargs = {
            "max_new_tokens": self.distill.max_new_tokens,
            "eos_token_id": self._eos_token_ids,
            "pad_token_id": self._pad_token_id,
            "use_cache": True,
            # Same support as the loss: see distill_vocab_limit.
            "suppress_tokens": list(range(self._vocab_limit, logits_dim)),
        }
        if greedy or self.distill.temperature == 0.0:
            # Explicit, because Llama's generation_config ships do_sample=True.
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            # top_k=0 disables transformers' default top-k of 50, which would
            # otherwise truncate the "on-policy" distribution.
            kwargs.update(do_sample=True, temperature=self.distill.temperature, top_p=self.distill.top_p, top_k=0)
        return kwargs

    @torch.no_grad()
    def _rollout(self, inputs: dict, from_student: bool, greedy: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one response per utterance; return ``(ids, mask)``, pad ids where masked."""
        unwrapped = self.model
        gen_kwargs = self._generation_kwargs_for(greedy, unwrapped.text_decoder.get_output_embeddings().weight.shape[0])
        with self.accelerator.autocast():
            if from_student:
                response = unwrapped.generate(
                    input_ids=inputs["student_prompt_input_ids"],
                    attention_mask=inputs["student_prompt_attention_mask"],
                    input_features=self._cast_to_audio_dtype(inputs["input_features"]),
                    features_attention_mask=inputs.get("features_attention_mask"),
                    **gen_kwargs,
                )
            else:
                prompt_ids = inputs["teacher_prompt_input_ids"]
                response = unwrapped.text_decoder.generate(
                    input_ids=prompt_ids,
                    attention_mask=inputs["teacher_prompt_attention_mask"],
                    **gen_kwargs,
                )
                # Generating from input_ids returns prompt + response; from
                # inputs_embeds (the student path) only the response.
                response = response[:, prompt_ids.shape[1]:]
        mask = response_mask(response, torch.tensor(self._eos_token_ids, device=response.device))
        return response.masked_fill(~mask, self._pad_token_id), mask

    def _student_and_teacher_logits(self, model, inputs: dict, response: torch.Tensor, mask: torch.Tensor):
        attention = mask.long()
        # Through `model` (the DDP wrapper) so the gradient is all-reduced.
        student = response_logits(
            model,
            inputs["student_prompt_input_ids"],
            inputs["student_prompt_attention_mask"],
            response,
            attention,
            self._vocab_limit,
            input_features=inputs["input_features"],
            features_attention_mask=inputs.get("features_attention_mask"),
        )
        with torch.no_grad(), self.accelerator.autocast():
            teacher = response_logits(
                self.model.text_decoder,
                inputs["teacher_prompt_input_ids"],
                inputs["teacher_prompt_attention_mask"],
                response,
                attention,
                self._vocab_limit,
            )
        return student, teacher

    # -- training ------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        for key in _GOLD_ONLY_KEYS:
            inputs.pop(key, None)
        from_student = self._rng.random() < self.distill.lmbda
        response, mask = self._rollout(inputs, from_student=from_student)
        student, teacher = self._student_and_teacher_logits(model, inputs, response, mask)
        loss = distillation_loss(student, teacher, response, mask, self.distill)
        self._record_stats(student, teacher, response, mask, from_student)
        return (loss, None) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if not self._gradients_checked:
            check_adapter_only_gradients(self.model)
            self._gradients_checked = True
            logger.info("Gradient check passed: only adapter parameters received gradients.")
        return loss

    @torch.no_grad()
    def _record_stats(self, student, teacher, response, mask, from_student: bool) -> None:
        """Per-micro-batch diagnostics, averaged and reduced at log time.

        ``teacher_logp`` is the mean log-probability the teacher gives the
        response tokens. On on-policy batches it is the reward-hacking monitor:
        it can only go up by the student writing text the transcript makes more
        plausible, and a collapse to generic boilerplate shows up as it rising
        while ``response_len`` falls.
        """
        tokens = response[mask].unsqueeze(-1)
        s, t = student[mask].float(), teacher[mask].float()
        student_logp = (s.gather(-1, tokens).squeeze(-1) - s.logsumexp(-1)).mean()
        teacher_logp = (t.gather(-1, tokens).squeeze(-1) - t.logsumexp(-1)).mean()
        lengths = mask.sum(dim=1).float()
        stats = self._distill_stats
        stats["on_policy_frac"].append(float(from_student))
        stats["response_len"].append(lengths.mean().item())
        stats["truncated_frac"].append((lengths == response.shape[1]).float().mean().item())
        stats["student_logp" if from_student else "student_logp_on_teacher"].append(student_logp.item())
        stats["teacher_logp" if from_student else "teacher_logp_on_teacher"].append(teacher_logp.item())

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Add ``distill/*`` means, reduced across ranks, to training rows.

        The collective runs unconditionally on every rank, as in the parent's
        reductions: gating it on local state would deadlock.
        """
        local = dict(self._distill_stats)
        self._distill_stats = defaultdict(list)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: list[dict | None] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]
        merged: dict[str, list[float]] = defaultdict(list)
        for rank_stats in gathered:
            for key, values in (rank_stats or {}).items():
                merged[key].extend(values)
        if "loss" in logs:
            logs.update({f"distill/{k}": sum(v) / len(v) for k, v in merged.items() if v})
        super().log(logs, start_time)

    # -- evaluation ----------------------------------------------------------

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        """WER as usual, and ``eval_loss`` := the student's alignment gap.

        The alignment gap is the per-token forward KL(teacher || student) on
        the teacher's own greedy response to the transcript -- how far the audio
        path is from reading the utterance as the text path does. It is defined
        for any MELT checkpoint, so it can be scored on the gold-transcript MA
        arms too, which is what makes it the comparison metric. WER keeps using
        ``validation_ds``'s prompt, which for this method should carry a
        transcription instruction (an instruction-free model answers rather
        than transcribes).
        """
        distill_inputs = {key: inputs.pop(key) for key in DISTILL_PROMPT_KEYS if key in inputs}
        features = {key: inputs[key] for key in ("input_features", "features_attention_mask") if key in inputs}
        loss, predictions, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if not self.distill.eval_alignment_gap or not distill_inputs:
            return loss, predictions, labels

        batch = self._prepare_inputs({**distill_inputs, **features})
        response, mask = self._rollout(batch, from_student=False, greedy=True)
        with torch.no_grad(), self.accelerator.autocast():
            student, teacher = self._student_and_teacher_logits(model, batch, response, mask)
        gap = distillation_loss(student, teacher, response, mask, SelfDistillConfig(loss="jsd", beta=0.0, lmbda=0.0))
        return gap.detach(), predictions, labels


__all__ = [
    "DISTILL_PROMPT_KEYS",
    "MELTSelfDistillTrainer",
    "SelfDistillConfig",
    "SelfDistillDataset",
    "SelfDistillEvalCollator",
    "build_distill_prompts",
    "check_adapter_only",
    "check_adapter_only_gradients",
    "distill_vocab_limit",
    "distillation_loss",
    "response_logits",
    "response_mask",
]

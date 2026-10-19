"""Tests for projects/self-distill/sft/build_sft_mix.py.

The builder spreads one total number of hours over two sets of entries (HR5: a fixed
cap per entry; EU24: a common cap solved so the totals agree) and writes the result
as mux weights. The arithmetic is tested on its own (water-filling inside an entry,
the bisection for the cap), then the pieces that would silently train the wrong
thing: the text field each source is tagged with, the audit that reads cuts, and the
weights as they come out of a rendered file.

Manifests are plain JSONL written by hand, and the process pool is replaced by an
inline executor: nyx enforces strict commit accounting, so forking inside a test is
the expensive part, not the arithmetic.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects" / "self-distill" / "sft"))

import build_sft_mix as bsm  # noqa: E402


class _InlineExecutor:
    """Stands in for ProcessPoolExecutor so tests never fork."""

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, iterable, chunksize=1):
        return map(fn, iterable)


@pytest.fixture
def inline_pool(monkeypatch):
    monkeypatch.setattr(bsm.ctc, "ProcessPoolExecutor", _InlineExecutor)


# ---------------------------------------------------------------------------
# water_fill: the split of an entry's hours over its corpora
# ---------------------------------------------------------------------------


def test_water_fill_without_a_binding_cap_is_proportional_to_sqrt_of_the_hours():
    given = bsm.water_fill(100.0, [400.0, 100.0, 25.0])  # sqrt: 20, 10, 5

    assert given == pytest.approx([100 * 20 / 35, 100 * 10 / 35, 100 * 5 / 35])


def test_water_fill_never_asks_a_corpus_for_more_than_it_holds():
    # The proportional share of the 25 h corpus is 3000 * 5 / 205 = 73 h, but it only holds 25.
    given = bsm.water_fill(3000.0, [10_000.0, 25.0, 10_000.0])

    assert given[1] == pytest.approx(25.0)
    assert given[0] == given[2] == pytest.approx(1487.5)
    assert sum(given) == pytest.approx(3000.0)


def test_water_fill_redistributes_in_rounds_when_a_second_corpus_becomes_binding():
    # Round 1 caps the 4 h corpus (share 19.8 h). The 100 h corpus is under its cap at 99.1 h, but
    # the surplus pushes it to 100.5 h, so round 2 caps it too.
    capacity = [4.0, 100.0, 10_000.0]
    given = bsm.water_fill(1110.0, capacity)

    assert given == pytest.approx([4.0, 100.0, 1006.0])
    # Corpora that are not capped sit at one common level of given / sqrt(held).
    uncapped = bsm.water_fill(300.0, [10_000.0, 2_500.0, 400.0])
    assert [g / c**0.5 for g, c in zip(uncapped, [10_000.0, 2_500.0, 400.0])] == pytest.approx([300 / 170] * 3)


def test_water_fill_gives_everything_when_the_budget_covers_it():
    assert bsm.water_fill(500.0, [10.0, 20.0]) == [10.0, 20.0]
    assert bsm.water_fill(30.0, [10.0, 20.0]) == [10.0, 20.0]


def test_water_fill_alpha_is_the_exponent_of_the_split():
    assert bsm.water_fill(30.0, [100.0, 200.0], alpha=1.0) == pytest.approx([10.0, 20.0])
    assert bsm.water_fill(30.0, [100.0, 200.0], alpha=0.0) == pytest.approx([15.0, 15.0])


def test_water_fill_leaves_an_empty_corpus_empty_and_rejects_nonsense():
    assert bsm.water_fill(5.0, [0.0, 10.0, 10.0])[0] == 0.0
    with pytest.raises(ValueError):
        bsm.water_fill(-1.0, [1.0])
    with pytest.raises(ValueError):
        bsm.water_fill(1.0, [-1.0, 2.0])


def test_water_fill_random_inputs_keep_their_invariants():
    rng = random.Random(7)
    for _ in range(200):
        capacity = [rng.choice([0.5, 3.0, 40.0, 900.0, 25_000.0]) for _ in range(rng.randint(1, 7))]
        budget = rng.uniform(0.1, 1.2) * sum(capacity)
        given = bsm.water_fill(budget, capacity)
        assert sum(given) == pytest.approx(min(budget, sum(capacity)))
        assert all(0.0 <= g <= c + 1e-9 for g, c in zip(given, capacity))


# ---------------------------------------------------------------------------
# solve_common_cap: the EU24 cap
# ---------------------------------------------------------------------------


def _exact_cap(available: list[float], target: float) -> float:
    """Closed form: sort, fill the smallest entries whole, spread the rest evenly."""
    rest = sorted(available)
    used = 0.0
    for i, a in enumerate(rest):
        cap = (target - used) / (len(rest) - i)
        if a >= cap:
            return cap
        used += a
    return rest[-1]


def test_solve_common_cap_matches_the_closed_form():
    rng = random.Random(3)
    for _ in range(100):
        available = [
            rng.choice([0.6, 2.0, 15.0, 90.0, 400.0, 30_000.0]) * rng.uniform(0.5, 1.5)
            for _ in range(rng.randint(2, 40))
        ]
        target = rng.uniform(0.05, 0.95) * sum(available)
        cap = bsm.solve_common_cap(available, target)
        assert cap == pytest.approx(_exact_cap(available, target), rel=1e-6)
        assert sum(min(cap, a) for a in available) == pytest.approx(target, abs=1e-6)


def test_solve_common_cap_entries_below_the_cap_use_all_their_data():
    available = [3.0, 12.0, 500.0, 9_000.0]
    cap = bsm.solve_common_cap(available, 3.0 + 12.0 + 2 * 100.0)

    assert cap == pytest.approx(100.0)
    assert [min(cap, a) for a in available] == pytest.approx([3.0, 12.0, 100.0, 100.0])


def test_solve_common_cap_refuses_a_target_nothing_can_reach():
    assert bsm.solve_common_cap([5.0, 7.0], 12.0) == 7.0  # everything, exactly
    with pytest.raises(ValueError, match="exceeds"):
        bsm.solve_common_cap([5.0, 7.0], 12.5)


# ---------------------------------------------------------------------------
# The catalogue: what each condition contains and what each source is tagged with
# ---------------------------------------------------------------------------


def _by_key(entries):
    return {e.key: e for e in entries}


def test_hr5_has_the_thirteen_entries():
    keys = [e.key for e in bsm.hr5_entries()]

    assert keys == (
        [f"asr:{lang}" for lang in ("en", "de", "fr", "es", "it")]
        + [f"st:{lang}-en" for lang in ("de", "fr", "es", "it")]
        + [f"st:en-{lang}" for lang in ("de", "fr", "es", "it")]
    )


def test_eu24_covers_every_language_and_every_direction_with_data():
    entries = bsm.eu24_entries()
    asr = [e.src for e in entries if e.task == "asr"]
    x_to_en = {e.src for e in entries if e.task == "st" and e.tgt == "en"}
    en_to_x = {e.tgt for e in entries if e.task == "st" and e.src == "en"}

    assert asr == list(bsm.EU24) and len(asr) == 24
    # yodas-granary ast covers 20 languages, CoVoST2 adds Slovenian.
    assert x_to_en == set(bsm.EU24) - {"en", "ga", "mt"} and len(x_to_en) == 21
    assert en_to_x == {"de", "et", "lv", "sl", "sv", "es", "fr", "it", "nl", "pl", "pt", "ro"}


def test_text_fields_follow_the_corpus_and_the_task():
    asr = _by_key(bsm.eu24_entries())
    field = {s.rel: s.text_field for e in asr.values() for s in e.sources}

    assert field["cv22_sidon/de/train"] == "custom.metadata.sentence"
    assert field["mls_sidon/german/train"] == "custom.pnc_text"
    assert field["voxpopuli/de/train"] == "custom.pnc_text"
    assert field["fleurs/de_de/train"] == "custom.pnc_text"
    assert field["yodas-granary/German/asr_only"] is None
    # The same ast leaf is read two ways: the supervision for ASR, the translation for X->en.
    ast_for_asr = next(s for s in asr["asr:de"].sources if s.rel == "yodas-granary/German/ast")
    ast_for_st = next(s for s in asr["st:de-en"].sources if s.rel == "yodas-granary/German/ast")
    assert ast_for_asr.text_field is None
    assert ast_for_st.text_field == "custom.translation_en"
    # CoVoST2 and Europarl-ST keep the translation in the supervision.
    assert all(s.text_field is None for s in asr["st:de-en"].sources if s.corpus == "covost2")
    assert all(s.text_field is None for s in asr["st:en-de"].sources)


def test_leaf_tags_carry_the_task_the_languages_and_the_locale_spelling():
    entries = _by_key(bsm.eu24_entries())
    ga = next(s for s in entries["asr:ga"].sources if s.rel == "cv22_sidon/ga-IE/train")
    sv_en = next(s for s in entries["st:sv-en"].sources if s.corpus == "covost2")
    en_sv = next(s for s in entries["st:en-sv"].sources if s.corpus == "covost2")
    de = next(s for s in entries["asr:de"].sources if s.rel == "cv22_sidon/de/train")

    assert entries["asr:ga"].leaf_tags(ga) == {
        "task": "asr", "lang": "ga", "text_field": "custom.metadata.sentence", "region_code": "ga-IE",
    }  # fmt: skip
    assert entries["st:sv-en"].leaf_tags(sv_en) == {
        "task": "st",
        "src_lang": "sv",
        "tgt_lang": "en",
        "src_region_code": "sv-SE",
    }
    assert entries["st:en-sv"].leaf_tags(en_sv) == {
        "task": "st",
        "src_lang": "en",
        "tgt_lang": "sv",
        "tgt_region_code": "sv-SE",
    }
    assert "region_code" not in entries["asr:de"].leaf_tags(de)  # the directory is the language code
    assert entries["st:de-en"].group_tags() == {"task": "st", "src_lang": "de", "tgt_lang": "en"}


def test_excluded_sources_never_reach_an_entry():
    rels = {s.rel for e in bsm.eu24_entries() + bsm.hr5_entries() for s in e.sources}

    assert set(bsm.EXCLUDED) and not rels & set(bsm.EXCLUDED)
    assert "voxpopuli/hr/train" in {s.rel for s in bsm.asr_sources("hr", skip=frozenset())}


def test_fleurs_leaf_without_pnc_text_is_read_from_the_supervision_on_purpose():
    ga = {s.rel: s for s in bsm.asr_sources("ga")}

    assert ga["fleurs/ga_ie/train"].text_field is None and ga["fleurs/ga_ie/train"].note
    assert ga["cv22_sidon/ga-IE/train"].text_field == "custom.metadata.sentence"
    assert next(s for s in bsm.asr_sources("pt") if s.corpus == "fleurs").text_field == "custom.pnc_text"


def test_generic_converter_voxpopuli_leaves_are_allowed_their_blank_cuts_and_the_others_are_not():
    allowed = {
        s.rel: s.max_blank
        for lang in ("cs", "sl", "de", "en")
        for s in bsm.asr_sources(lang)
        if s.corpus == "voxpopuli"
    }

    assert allowed["voxpopuli/cs/train"] == bsm.GENERIC_VOXPOPULI_MAX_BLANK == allowed["voxpopuli/sl/train"]
    assert allowed["voxpopuli/de/train"] == allowed["voxpopuli/en/train"] == bsm.AUDIT_MAX_BLANK


def test_validation_is_one_named_set_per_language_and_per_direction():
    sets = bsm.validation_sets()
    names = [s.name for s in sets]
    by_name = {s.name: s for s in sets}

    assert len(names) == len(set(names)) == 24 + 10 + 5 + 8
    assert [s.src for s in sets if s.task == "asr"] == list(bsm.EU24)
    assert by_name["asr_de"].rel == "cv22_sidon/de/validation"
    assert by_name["asr_ga"].rel == "cv22_sidon/ga-IE/validation"
    assert by_name["asr_hr"].rel == "fleurs/hr_hr/validation"  # CV22 has no Croatian
    assert by_name["st_sv_en_covost2"].rel == "covost2/sv-SE_en/dev"
    assert by_name["st_en_de_covost2"].rel == "covost2/en_de/dev"
    assert by_name["st_en_de_europarl"].rel == "europarl_st/en_de/validation"
    # Nothing is validated on a split a training source reads.
    train = {s.rel for e in bsm.eu24_entries() + bsm.hr5_entries() for s in e.sources}
    assert not train & {s.rel for s in sets}


def test_every_language_the_prompts_will_name_is_known_to_the_loader():
    helpers = pytest.importorskip("melt.training.data.audio.lhotse.helpers")
    langs = {e.src for e in bsm.eu24_entries()} | {e.tgt for e in bsm.eu24_entries() if e.tgt}
    langs |= {s.src for s in bsm.validation_sets()} | {s.tgt for s in bsm.validation_sets() if s.tgt}

    assert langs - {None} <= set(helpers.LANGUAGE_ISO_TO_NAME)


# ---------------------------------------------------------------------------
# Planning and weights, on synthetic measurements
# ---------------------------------------------------------------------------


def _measures(entries, hours_of=None, mean_cut=None):
    """Synthetic hours per source: a recognisable number per corpus, mean cut per corpus."""
    hours_of = hours_of or {"cv22_sidon": 300.0, "mls_sidon": 2_000.0, "voxpopuli": 120.0, "yodas-granary": 9_000.0,
                            "fleurs": 9.0, "covost2": 150.0, "europarl_st": 75.0}  # fmt: skip
    mean_cut = mean_cut or {"cv22_sidon": 5.5, "mls_sidon": 15.0, "voxpopuli": 10.0, "yodas-granary": 9.0,
                            "fleurs": 11.0, "covost2": 5.0, "europarl_st": 8.5}  # fmt: skip
    out = {}
    for e in entries:
        for s in e.sources:
            h = hours_of[s.corpus]
            out[s.rel] = bsm.Measure(h, round(h * 3600 / mean_cut[s.corpus]), h)
    return out


def test_an_entry_draws_the_smaller_of_the_cap_and_what_it_holds():
    entries = _by_key(bsm.hr5_entries())
    measures = _measures(entries.values())

    big = bsm.plan_entry(entries["asr:de"], 600.0, measures)
    small = bsm.plan_entry(entries["st:en-es"], 600.0, measures)

    assert big.hours == 600.0 and big.at_limit and sum(d.hours for d in big.draws) == pytest.approx(600.0)
    assert small.hours == small.available == 75.0 and not small.at_limit
    assert all(d.hours <= d.available + 1e-9 for p in (big, small) for d in p.draws)


def test_both_conditions_draw_the_same_total_and_entries_below_the_cap_use_everything():
    hr5, eu24 = bsm.hr5_entries(), bsm.eu24_entries()
    measures = _measures(hr5 + eu24)
    hr5_plans = bsm.plan_condition(hr5, bsm.HR5_CAP_HOURS, measures)
    total = sum(p.hours for p in hr5_plans)

    cap = bsm.solve_common_cap(bsm.entry_availability(eu24, measures), total)
    eu24_plans = bsm.plan_condition(eu24, cap, measures)

    assert sum(p.hours for p in eu24_plans) == pytest.approx(total, abs=1e-6)
    assert cap < bsm.HR5_CAP_HOURS
    for p in eu24_plans:
        assert p.hours == pytest.approx(min(cap, p.available))
        if not p.at_limit:
            assert [d.hours for d in p.draws] == pytest.approx([d.available for d in p.draws])


def test_mux_weights_deliver_the_hours_of_every_source_not_the_hour_shares():
    entries = bsm.eu24_entries()
    measures = _measures(entries)
    plans = bsm.plan_condition(entries, 150.0, measures)
    total = sum(p.hours for p in plans)

    group_w, leaf_w = bsm.mux_weights(plans)
    mean_cut = [[d.mean_cut for d in p.draws] for p in plans]
    audio = bsm.cmw.implied_hour_shares(group_w, leaf_w, mean_cut)

    assert sum(group_w) == pytest.approx(1.0) and all(sum(w) == pytest.approx(1.0) for w in leaf_w)
    for p, shares in zip(plans, audio):
        for d, share in zip(p.draws, shares):
            assert share == pytest.approx(d.hours / total, rel=1e-6)
    # A weight is a probability of drawing a CUT: proportional to hours / mean cut, not to hours.
    for p, weights in zip(plans, leaf_w):
        cuts = [d.hours / d.mean_cut for d in p.draws]
        assert weights == pytest.approx([c / sum(cuts) for c in cuts])
    german = next(i for i, p in enumerate(plans) if p.entry.key == "asr:de")
    hour_shares = [d.hours / plans[german].hours for d in plans[german].draws]
    assert leaf_w[german] != pytest.approx(hour_shares, rel=0.05)  # the cuts of the sources differ several-fold


def test_entry_order_is_asr_then_x_to_en_then_en_to_x_largest_first():
    entries = bsm.build_entries(("de", "cs"), ("de", "cs"), ("de", "sv"))
    measures = _measures(entries)
    plans = bsm.plan_condition(entries, 200.0, measures)

    order = [plans[i].entry.key for i in bsm.entry_order(plans)]

    # de and cs are both capped at 200 h, so ties fall back to the key; en->de (225 h held) is
    # capped at 200 h and en->sv holds 150 h, so the entry drawing more comes first.
    assert order == ["asr:cs", "asr:de", "st:cs-en", "st:de-en", "st:en-de", "st:en-sv"]


# ---------------------------------------------------------------------------
# Rendering: the file that comes out
# ---------------------------------------------------------------------------


def _render(entries, limit=150.0, name="SFT-TOY"):
    measures = _measures(entries)
    eval_sets = bsm.validation_sets()[:2] + [s for s in bsm.validation_sets() if s.name.startswith("st_en_de_")]
    for s in eval_sets:
        measures[s.rel] = bsm.Measure(20.0, 14_000, 20.0)
    plans = bsm.plan_condition(entries, limit, measures)
    template = (REPO_ROOT / "projects" / "self-distill" / "sft" / "SFT-template.yaml").read_text()
    text = bsm.render_config(template, name, ["intro line"], plans, "Cap note.", "[1.0, 2.0]", eval_sets, measures)
    return text, plans, measures, eval_sets


def test_a_rendered_config_parses_and_carries_the_groups_weights_and_totals():
    entries = bsm.build_entries(("de", "ga"), ("de",), ("de", "sv"))
    text, plans, measures, eval_sets = _render(entries)
    cfg = yaml.safe_load(text)
    train = cfg["data"]["train_ds"]

    assert "# @@HEADER@@" not in text and text.startswith("# SFT-TOY --")
    assert f"TOTAL HOURS: {sum(p.hours for p in plans):,.1f} h over {len(plans)} entries" in text
    assert cfg["run"]["exp_name"] == "SFT-TOY"
    assert [g["tags"].get("lang") or f"{g['tags']['src_lang']}-{g['tags']['tgt_lang']}" for g in train["input_cfg"]] == [
        "de", "ga", "de-en", "en-de", "en-sv",
    ]  # fmt: skip
    assert all(g["type"] == "group" and g["input_cfg"] for g in train["input_cfg"])
    assert sum(g["weight"] for g in train["input_cfg"]) == pytest.approx(1.0, abs=1e-8)
    for g in train["input_cfg"]:
        assert sum(c["weight"] for c in g["input_cfg"]) == pytest.approx(1.0, abs=1e-8)
        assert all(c["shar_path"].startswith("${oc.env:LOCAL_DATASETS_DIR}/") for c in g["input_cfg"])
    assert train["total_hours"] == pytest.approx(sum(p.hours for p in plans), abs=0.01)
    assert train["total_cuts"] == bsm.total_cuts_drawn(plans)
    assert train["bucket_duration_bins"] == [1.0, 2.0]
    val = cfg["data"]["validation_ds"]
    assert [v["name"] for v in val["input_cfg"]] == [s.name for s in eval_sets]
    assert val["total_hours"] == pytest.approx(20.0 * len(eval_sets))


def test_the_rendered_weights_give_every_source_the_hours_written_beside_it():
    entries = bsm.build_entries(("de", "cs"), ("de",), ("sv",))
    text, plans, measures, _ = _render(entries)
    train = yaml.safe_load(text)["data"]["train_ds"]
    root = "${oc.env:LOCAL_DATASETS_DIR}/"

    audio: dict[tuple[str, str], float] = {}
    for g, i in zip(train["input_cfg"], bsm.entry_order(plans)):  # the file lists entries in entry_order
        for c in g["input_cfg"]:
            m = measures[c["shar_path"].removeprefix(root)]
            audio[(plans[i].entry.key, c["shar_path"])] = g["weight"] * c["weight"] * m.mean_cut
    total = sum(audio.values())
    for plan in plans:
        for d in plan.draws:
            share = audio[(plan.entry.key, root + d.source.rel)] / total
            assert share == pytest.approx(d.hours / sum(p.hours for p in plans), rel=1e-6)


def test_the_rendered_tags_match_the_catalogue_leaf_by_leaf():
    entries = bsm.build_entries(("de", "ga"), ("de",), ("sv",))
    text, plans, _, _ = _render(entries)
    train = yaml.safe_load(text)["data"]["train_ds"]

    for g, i in zip(train["input_cfg"], bsm.entry_order(plans)):
        plan = plans[i]
        assert g["tags"] == plan.entry.group_tags()
        for c, d in zip(g["input_cfg"], plan.draws):
            assert c["tags"] == plan.entry.leaf_tags(d.source)


def test_the_template_pins_the_recipe_the_study_asks_for():
    entries = bsm.build_entries(("de",), (), ())
    cfg = yaml.safe_load(_render(entries)[0])
    trainer, data, opt = cfg["trainer"], cfg["data"], cfg["optimization"]

    # PyYAML reads 6e-6 as a string (OmegaConf, which the trainer uses, reads it as a float).
    assert (float(opt["encoder_lr"]), float(opt["adapter_lr"]), float(opt["decoder_lr"])) == (6e-6, 2e-4, 2e-5)
    assert (opt["adam_beta1"], opt["adam_beta2"]) == (0.9, 0.95)
    assert trainer["lr_scheduler_type"] == "warmup_stable_decay" and trainer["warmup_steps"] == 0.05
    # num_decay_steps is the launcher's (max_steps / 5): the scheduler refuses to build without it.
    assert trainer["lr_scheduler_kwargs"] == {"min_lr_ratio": 0.1, "decay_type": "cosine"}
    assert trainer["gradient_accumulation_steps"] == 4 and trainer["bf16"] is True
    assert trainer["predict_with_generate"] is True and trainer["generation_max_length"] == 256
    assert trainer["eval_on_start"] is True and trainer["max_grad_norm"] == 1.0
    assert (data["apply_chat_template"], data["prompt_template_selection"], data["chat_template_config"]) == (
        True,
        "random",
        "chatml",
    )
    assert data["strict_text_field"] is True and data["validation_ds"]["strict_text_field"] is True
    train = data["train_ds"]
    assert (train["max_duration"], train["min_duration"], train["batch_duration"]) == (60.0, 0.5, 60)
    assert train["quadratic_duration"] is None and (train["seed"], train["shard_seed"]) == (1337, 1337)
    assert data["validation_ds"]["max_samples"] == 20
    model = cfg["model"]
    assert not any(model[part]["freeze"] for part in ("encoder", "decoder", "adapter"))


def _flatten(node, prefix=""):
    """A nested mapping as {dotted.key: value}, without the data blocks the builder fills in."""
    out = {}
    for key, value in node.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, dotted + "."))
        else:
            out[dotted] = value
    return out


def test_the_sd_template_differs_from_the_sft_template_only_where_the_recipe_does():
    sft = yaml.safe_load((REPO_ROOT / "projects" / "self-distill" / "sft" / "SFT-template.yaml").read_text())
    sd = yaml.safe_load((REPO_ROOT / "projects" / "self-distill" / "sft" / "SD-template.yaml").read_text())
    a, b = _flatten(sft), _flatten(sd)
    changed = {k for k in a.keys() | b.keys() if a.get(k, "<absent>") != b.get(k, "<absent>")}

    assert changed == {
        "run.exp_name",
        "run.memory_preallocation",  # the distillation trainer refuses the preallocation pass
        "model.encoder.freeze",  # only the adapter trains: the decoder is the teacher
        "model.decoder.freeze",
        "data.prompt_template_selection",  # one fixed instruction per task, not a random draw
        "data.validation_ds.prompt_template.asr",  # a single repeat string, not the SFT per-task mapping
        "data.validation_ds.prompt_template.st",
        "data.validation_ds.prompt_template",
    }
    assert (b["model.encoder.freeze"], b["model.decoder.freeze"], b["model.adapter.freeze"]) == (True, True, False)
    assert b["run.memory_preallocation"] is False and b["data.prompt_template_selection"] == "custom"
    assert b["data.validation_ds.prompt_template"].startswith("Repeat the following content exactly")


def test_the_sd_condition_is_the_asr_half_of_eu24_with_asr_validation_and_its_own_header():
    entries = [e for e in bsm.eu24_entries() if e.task == "asr"]
    assert [e.src for e in entries] == list(bsm.EU24) and {e.task for e in entries} == {"asr"}

    toy = bsm.build_entries(("de", "ga"), (), ())
    measures = _measures(toy)
    eval_sets = [ev for ev in bsm.validation_sets() if ev.task == "asr"][:3]
    for ev in eval_sets:
        measures[ev.rel] = bsm.Measure(20.0, 14_000, 20.0)
    plans = bsm.plan_condition(toy, 150.0, measures)
    total = sum(p.hours for p in plans)
    template = (REPO_ROOT / "projects" / "self-distill" / "sft" / "SD-template.yaml").read_text()
    text = bsm.render_config(
        template, "SD-TOY", bsm.sd24_intro(150.0, total), plans, "Cap note.", "[1.0, 2.0]", eval_sets, measures, None,
        bsm.sd24_title("SD-TOY"), bsm.SD24_PROVENANCE, bsm.SD24_VALIDATION,
    )  # fmt: skip
    cfg = yaml.safe_load(text)

    assert text.startswith("# SD-TOY -- ") and "adapter-only self-distillation" in text
    assert "all three components trainable" not in text and "HR5 and EU24 draw the same total" not in text
    assert "ASR only, 20 clips per language" in text
    assert cfg["data"]["train_ds"]["total_hours"] == pytest.approx(total, abs=0.01)
    assert {g["tags"]["task"] for g in cfg["data"]["train_ds"]["input_cfg"]} == {"asr"}
    assert {v["tags"]["task"] for v in cfg["data"]["validation_ds"]["input_cfg"]} == {"asr"}


def test_sd24_hours_renders_the_condition_alone_at_the_requested_total(tmp_path, monkeypatch, capsys):
    entries = [e for e in bsm.eu24_entries() if e.task == "asr"]
    measures = _measures(entries)
    for ev in bsm.validation_sets():
        measures[ev.rel] = bsm.Measure(20.0, 14_000, 20.0)
    monkeypatch.setattr(bsm, "measure_all", lambda *a, **k: (measures, {}))
    monkeypatch.setattr(bsm, "measure_bins", lambda *a, **k: "[1.0, 2.0]")
    argv = ["build_sft_mix.py", "--datasets-root", str(tmp_path), "--out-dir", str(tmp_path), "--skip-audit",
            "--sd24-hours", "500"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)

    assert bsm.main() == 0
    assert sorted(f.name for f in tmp_path.iterdir()) == ["SD-EU24-500h.yaml"]  # HR5 and EU24 are not rendered
    cfg = yaml.safe_load((tmp_path / "SD-EU24-500h.yaml").read_text())
    assert cfg["run"]["exp_name"] == "SD-EU24-500h"
    assert cfg["data"]["train_ds"]["total_hours"] == pytest.approx(500.0, abs=0.01)
    assert len(cfg["data"]["train_ds"]["input_cfg"]) == 24 and len(cfg["data"]["validation_ds"]["input_cfg"]) == 24
    assert (cfg["model"]["encoder"]["freeze"], cfg["model"]["decoder"]["freeze"]) == (True, True)
    assert "common cap c" in capsys.readouterr().out


def test_both_conditions_share_everything_but_the_mix():
    entries = bsm.build_entries(("de",), (), ())
    first, second = _render(entries, limit=100.0)[0], _render(entries, limit=50.0)[0]

    def outside_train(text):
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("  validation_ds:"))
        return "\n".join(lines[start:])

    assert outside_train(first) == outside_train(second)


# ---------------------------------------------------------------------------
# resolve_text mirrors get_text_from_cut(strict=True)
# ---------------------------------------------------------------------------

RECORDS = {
    "cv22": {"supervisions": [{"text": ""}], "custom": {"metadata": {"sentence": "Der See."}}},
    "mls": {"supervisions": [{"text": "der see"}], "custom": {"pnc_text": "Der See."}},
    "plain": {"supervisions": [{"text": "  hello  "}]},
    "custom_text": {"supervisions": [{"text": ""}], "custom": {"text": "fallback"}},
    "no_custom_block": {"supervisions": [{"text": "nuair a bhionn"}]},
    "textless": {"supervisions": [{"text": ""}]},
    "blank_field": {"supervisions": [{"text": "x"}], "custom": {"pnc_text": "   "}},
    "translation_missing": {"supervisions": [{"text": "Hallo"}], "custom": {"translation_en": None}},
    "translation": {"supervisions": [{"text": "Hallo"}], "custom": {"translation_en": "Hello"}},
}
FIELDS = [None, "custom.metadata.sentence", "custom.pnc_text", "custom.translation_en"]


def _cut(record: dict):
    from lhotse import Recording, SupervisionSegment
    from lhotse.audio import AudioSource
    from lhotse.cut import MonoCut

    rec = Recording(
        id="r", sources=[AudioSource(type="file", channels=[0], source="/dev/null")],
        sampling_rate=16000, num_samples=16000, duration=1.0,
    )  # fmt: skip
    sup = [
        SupervisionSegment(id="s", recording_id="r", start=0, duration=1.0, text=s.get("text"))
        for s in record["supervisions"]
    ]
    return MonoCut(
        id="c", start=0, duration=1.0, channel=0, recording=rec, supervisions=sup, custom=record.get("custom")
    )


@pytest.mark.parametrize("name", sorted(RECORDS))
@pytest.mark.parametrize("field", FIELDS)
def test_resolve_text_agrees_with_the_loaders_get_text_from_cut(name, field):
    pytest.importorskip("lhotse")
    from melt.training.data.audio.lhotse.helpers import get_text_from_cut

    record = RECORDS[name]
    bucket, text = bsm.resolve_text(record, field)
    cut = _cut(record)
    try:
        real = get_text_from_cut(cut, field or "text", strict=True)
    except ValueError:
        real, expected_bucket = None, "mismatch"
    else:
        expected_bucket = "ok" if real else "blank"

    assert bucket == expected_bucket
    assert text == real


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


def _write_leaf(root: Path, rel: str, records: list[dict], name: str = "cuts.000000.jsonl") -> None:
    leaf = root / rel
    leaf.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"id": f"c{i}", "duration": 4.0, **r}) for i, r in enumerate(records)]
    (leaf / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_audit_counts_ok_blank_and_mismatch_the_way_strict_mode_would(tmp_path):
    ok = {"supervisions": [{"text": ""}], "custom": {"metadata": {"sentence": "Der See."}}}
    textless = {"supervisions": [{"text": ""}]}
    mismatch = {"supervisions": [{"text": "unpunctuated"}]}
    _write_leaf(tmp_path, "cv22_sidon/xx/train", [ok] * 96 + [textless] * 3 + [mismatch])

    audit = bsm.audit_source(tmp_path, "cv22_sidon/xx/train", "custom.metadata.sentence", "asr")

    assert (audit.cuts, audit.ok, audit.blank, audit.mismatch) == (100, 96, 3, 1)
    assert any("no text anywhere" in p for p in audit.problems())
    assert any("nothing at 'custom.metadata.sentence'" in p for p in audit.problems())


def test_the_audit_tolerates_blank_cuts_only_up_to_the_sources_allowance(tmp_path):
    ok = {"supervisions": [{"text": ""}], "custom": {"pnc_text": "Der See."}}
    textless = {"supervisions": [{"text": ""}]}
    _write_leaf(tmp_path, "voxpopuli/xx/train", [ok] * 97 + [textless] * 3)

    strict = bsm.audit_source(tmp_path, "voxpopuli/xx/train", "custom.pnc_text", "asr")
    generic = bsm.audit_source(
        tmp_path, "voxpopuli/xx/train", "custom.pnc_text", "asr", bsm.GENERIC_VOXPOPULI_MAX_BLANK
    )

    assert strict.problems() and not generic.problems()


def test_the_audit_flags_an_st_source_whose_target_is_the_transcript(tmp_path):
    good = {"supervisions": [{"text": "Hallo Welt"}], "custom": {"translation_en": "Hello world"}}
    bad = {"supervisions": [{"text": "Hallo Welt"}], "custom": {"translation_en": "hallo welt"}}
    _write_leaf(tmp_path, "yodas-granary/German/ast", [good] * 10)
    _write_leaf(tmp_path, "yodas-granary/Dutch/ast", [bad] * 10)

    fine = bsm.audit_source(tmp_path, "yodas-granary/German/ast", bsm.TRANSLATION_FIELD, "st")
    copy = bsm.audit_source(tmp_path, "yodas-granary/Dutch/ast", bsm.TRANSLATION_FIELD, "st")

    assert not fine.problems()
    assert any("equal their source text" in p for p in copy.problems())


def test_the_audit_reads_the_supervision_of_a_covost_style_leaf_against_its_sentence(tmp_path):
    translated = {
        "supervisions": [{"text": "The early bird gets the worm."}],
        "custom": {"sentence": "Morgenstund hat Gold im Mund."},
    }
    _write_leaf(tmp_path, "covost2/de_en/train", [translated] * 5)

    audit = bsm.audit_source(tmp_path, "covost2/de_en/train", None, "st")

    assert audit.ok == 5 and audit.compared == 5 and audit.same == 0 and not audit.problems()


# ---------------------------------------------------------------------------
# Measurement and its caches
# ---------------------------------------------------------------------------


def test_window_hours_counts_only_cuts_inside_the_train_loaders_duration_window():
    hist = {"20": 10, "100": 10, "6000": 10, "9000": 10}  # 0.2 s, 1 s, 60 s, 90 s at 0.01 s resolution
    entry = {"res": 0.01, "hist": hist}

    assert bsm._window_hours(entry) == pytest.approx((10 * 1.0 + 10 * 60.0) / 3600.0)


def test_measure_all_measures_once_mirrors_into_the_hours_cache_and_notices_changes(tmp_path, inline_pool, capsys):
    root, hist_cache, hours_cache = tmp_path / "shar", tmp_path / "hist.json", tmp_path / "hours.json"
    _write_leaf(root, "cv22_sidon/xx/train", [{"duration": 2.0}] * 3 + [{"duration": 100.0}])
    _write_leaf(root, "cv22_sidon/xx/validation", [{"duration": 5.0}] * 2)
    rels = ["cv22_sidon/xx/train", "cv22_sidon/xx/validation"]

    measures, cache = bsm.measure_all(root, rels, hist_cache, hours_cache, jobs=1)
    train = measures[rels[0]]

    assert train.cuts == 4 and train.hours == pytest.approx(106.0 / 3600.0)
    assert train.window_hours == pytest.approx(6.0 / 3600.0)  # the 100 s cut is outside the window
    assert json.loads(hours_cache.read_text())[str(root / rels[0])]["cuts"] == 4
    assert set(json.loads(hist_cache.read_text())) == {str(root / r) for r in rels}

    capsys.readouterr()
    bsm.measure_all(root, rels, hist_cache, hours_cache, jobs=1)
    assert "Measuring" not in capsys.readouterr().out  # served from the cache

    manifest = root / rels[1] / "cuts.000000.jsonl"
    os.utime(manifest, (manifest.stat().st_atime, manifest.stat().st_mtime + 100))
    bsm.measure_all(root, rels, hist_cache, hours_cache, jobs=1)
    assert "manifests changed since last measurement" in capsys.readouterr().out


def test_measure_all_refuses_to_measure_when_told_not_to_and_names_missing_directories(tmp_path, inline_pool):
    root = tmp_path / "shar"
    _write_leaf(root, "cv22_sidon/xx/train", [{"duration": 2.0}])

    with pytest.raises(SystemExit, match="do not exist"):
        bsm.measure_all(root, ["cv22_sidon/nope/train"], tmp_path / "h.json", tmp_path / "m.json", jobs=1)
    with pytest.raises(SystemExit, match="--no-measure"):
        bsm.measure_all(root, ["cv22_sidon/xx/train"], tmp_path / "h.json", tmp_path / "m.json", jobs=1, measure=False)

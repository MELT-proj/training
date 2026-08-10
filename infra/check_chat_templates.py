#!/usr/bin/env python3
"""Report each backbone's chat template and the MELT config that matches it.

Two things this catches before a run, both of which are silent otherwise.

**A base model that ships a chat template.** Qwen 2.5 base does, complete with a
"You are a helpful assistant." system turn. If the base arm of a base-vs-instruct
comparison silently gets chat formatting, the comparison measures formatting
rather than instruction tuning. Decide deliberately which format each arm uses,
and record the decision.

**A boundary mismatch.** Label masking locates ``assistant_start`` /
``assistant_end`` as literal strings. When they are absent — a Llama tokenizer
under the ``chatml`` config — the span is never found and *every* token ends up
masked, so the run trains on nothing without raising.

Separately, note that Qwen 3 and 3.5 inject an empty ``<think>`` block after the
assistant header, and ``enable_thinking=False`` does not remove it. Masking is
inclusive of the boundaries, so that block is part of the training target no
matter which config is chosen; it is reported here as a warning, not an error.

    python3 infra/check_chat_templates.py \\
        Qwen/Qwen3.5-2B-Base Qwen/Qwen3.5-2B utter-project/EuroLLM-1.7B

With no arguments it checks the campaign's six backbones.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from melt.training.data.chat_templates import (  # noqa: E402
    CHAT_TEMPLATE_CONFIGS,
    validate_chat_template_config,
)


CAMPAIGN_BACKBONES = [
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3.5-2B",
    "utter-project/EuroLLM-1.7B",
    "utter-project/EuroLLM-1.7B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
]

# Which of those are the *base* arm. This cannot be inferred from the id:
# `Qwen/Qwen3.5-2B` is the instruct model while `Qwen/Qwen3.5-2B-Base` is base,
# and `Qwen/Qwen2.5-1.5B` is base with no marker at all. A wrong guess here
# hides the very thing the check exists to surface, so it is declared.
KNOWN_BASE = {
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "utter-project/EuroLLM-1.7B",
    "meta-llama/Llama-3.2-1B",
}

PROBE = [
    {"role": "user", "content": "__melt_probe_user__"},
    {"role": "assistant", "content": "__melt_probe_assistant__"},
]


def main() -> int:
    from transformers import AutoTokenizer

    models = sys.argv[1:] or CAMPAIGN_BACKBONES
    if os.environ.get("HF_HUB_OFFLINE") is None:
        print("(set HF_HUB_OFFLINE=1 to check the cache without network)\n")

    problems = 0
    for model_id in models:
        print("=" * 72)
        print(model_id)
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  COULD NOT LOAD: {type(exc).__name__}: {str(exc)[:160]}")
            problems += 1
            continue

        template = getattr(tok, "chat_template", None)
        known_base = model_id in KNOWN_BASE

        print(f"  has chat_template : {bool(template)}")
        if template and known_base:
            problems += 1
            print(
                "  ATTENTION: this is a BASE checkpoint and it ships a chat\n"
                "  template. If it is applied, the base-vs-instruct arm becomes\n"
                "  a comparison of formatting rather than of instruction tuning.\n"
                "  Decide explicitly which format this arm uses, and record it."
            )
        elif template and model_id not in CAMPAIGN_BACKBONES:
            print(
                "  (not a declared campaign backbone — if this is a base model, "
                "add it to KNOWN_BASE so the check is not silently skipped)"
            )

        if not template:
            print("  -> no template; MA/IFT must run unformatted for this model.")
            continue

        try:
            rendered = tok.apply_chat_template(PROBE, tokenize=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  render failed: {type(exc).__name__}")
            problems += 1
            continue

        print(f"  rendered          : {rendered!r}")

        matches = []
        for name in CHAT_TEMPLATE_CONFIGS:
            try:
                validate_chat_template_config(tok, CHAT_TEMPLATE_CONFIGS[name], name)
                matches.append(name)
            except ValueError:
                continue

        if matches:
            print(f"  chat_template_config: {' or '.join(matches)}")
        else:
            print(
                "  chat_template_config: NONE MATCH — add an entry to "
                "CHAT_TEMPLATE_CONFIGS before training this backbone, or label "
                "masking will silently train on the wrong tokens."
            )
            problems += 1

    print("=" * 72)
    if problems:
        print(f"{problems} model(s) need attention before launching.")
        return 1
    print("All models resolved to a matching chat_template_config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

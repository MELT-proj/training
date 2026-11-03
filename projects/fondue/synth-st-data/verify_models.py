"""Offline check that EuroLLM-9B-Instruct (vLLM) and MetricX-24-Hybrid-XXL (bf16) load and run.

Run inside synth.sif with HF_HUB_OFFLINE=1. Prints driver/CUDA info first.
"""

import os
import time

import torch


EURO = "utter-project/EuroLLM-9B-Instruct"
SRC = [
    "The European Parliament voted on Tuesday in Strasbourg.",
    "Please take two tablets of ibuprofen after meals.",
]


def main() -> None:
    print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
    print("gpu", torch.cuda.get_device_name(0), "n", torch.cuda.device_count())
    os.system("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")

    from vllm import LLM, SamplingParams

    t = time.time()
    llm = LLM(EURO, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.45)
    print("vllm load s", time.time() - t)
    msgs = [
        [
            {
                "role": "user",
                "content": f"Translate the following English text into Latvian. Output only the translation.\n\n{s}",
            }
        ]
        for s in SRC
    ]
    outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=256))
    mt = [o.outputs[0].text.strip() for o in outs]
    print("MT", mt)
    del llm
    torch.cuda.empty_cache()

    from run_mt_qe import load_qe, metricx_scores

    model, tok, _ = load_qe("/dev/null", "verify", "bf16")
    print("MetricX", metricx_scores(model, tok, list(zip(SRC, mt)), 8))


if __name__ == "__main__":
    main()

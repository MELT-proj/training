"""Offline check that EuroLLM-9B-Instruct (vLLM) and XCOMET-XXL load and run.

Run inside synth.sif with HF_HUB_OFFLINE=1. Prints driver/CUDA info first.
"""

import os
import time

import torch


EURO = "utter-project/EuroLLM-9B-Instruct"
XCOMET = "Unbabel/XCOMET-XXL"
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

    from comet import load_from_checkpoint
    from huggingface_hub import snapshot_download

    path = snapshot_download(XCOMET, local_files_only=True)
    t = time.time()
    model = load_from_checkpoint(os.path.join(path, "checkpoints/model.ckpt"))
    print("comet load s", time.time() - t)
    out = model.predict([{"src": s, "mt": m} for s, m in zip(SRC, mt)], batch_size=8, gpus=1)
    print("XCOMET", out.scores, [list(md) for md in out.metadata.error_spans])


if __name__ == "__main__":
    main()

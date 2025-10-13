# interim-speech-lm

This repo contains the code to train and evaluate SpeechLM.

Training TODOs:

- [x] basic training with python script
- [x] using accelerate for distributed training on single-node, multi-gpu exps (DDP and FSDP) 
- [ ] support a multi-node training setup
- [x] support training restarts from checkpoints
- [x] use different learning rates for speech encoder, text decoder, and adapter
- [x] add hint about text length in pretraining examples
- [x] handle string/transcript normalization 
- [ ] move training config into yaml files
- [ ] add WER/CER computation using Whisper normalization recipe in `compute_metrics`

Eval TODOs:

- [x] have a basic decoding function to see if the model can do at least ASR
- [x] fix caching for decoding 
- [x] support HF's generate API, so that we can use any sampling strategy
- [ ] code a PoC eval suite using WER/CER on standard datasets, e.g., Librispeech Clean/Other
- [x] build a mini mini mini gradio demo
- [ ] include some more eval benchmarks


MISC:

- [x] code basic synthetic data generation using open-weight TTS and IFT datasets


## Commands

IWSLT:

```
sbatch --partition=h100 --qos=gpu-h100 --job-name=MELT_qwen05 ./bash/train_slurm.sh config/iwslt/w2v_qwen2.5-0.5.yaml config/accelerate/zero1.yaml
```



#### Useful commands

```bash
VENV_PATH=/mnt/scratch-artemis/giuseppe/venvs/melt/bin/activate \
  TMPDIR=/mnt/scratch-artemis/giuseppe/melt-data/tmp \
  HF_HOME=/mnt/scratch-artemis/giuseppe/melt-data/hf_cache \
  OUTPUT_DIR=/mnt/scratch-artemis/giuseppe/melt-data/outputs \
  LOCAL_DATASETS_DIR=/mnt/scratch-nyx/giuseppe/melt/iwslt-2026/shar \
  bash ./projects/iwslt26-metric/run_train.sh config/accelerate/fsdp2.yaml \
  --config projects/iwslt26-metric/config.yaml \
  --trainer.gradient-accumulation-steps 4 \
  --trainer.output_dir /mnt/scratch-artemis/giuseppe/melt-data/outputs/MELT_QE_v1.1
```
[![Tests](https://img.shields.io/github/actions/workflow/status/MELT-proj/training/ci.yml?label=tests&logo=github)](https://github.com/MELT-proj/training/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)

# MELT Training

MELT is a training and modeling stack built on Hugging Face components with Lhotse-based speech dataloading.

## Milestones
- [**2025/07**] v1 of the model was accepted and published as system paper at IWLST 2025. [Link](https://aclanthology.org/2025.iwslt-1.36/) 

## Getting Started
For setup and usage details, see the folder-specific guides:
- docs/run_training.md – end-to-end training and launch details
- docs/hpc_runbook.md – operating runs on MN5 (air-gapped): code sync, image build, submit
- docs/lhotse_dataloading.md – data preparation and Shar/Lhotse notes
- docs/generation_eval.md – how eval WER/CER is decoded, and the knobs that bound it
- config/README.md – configuration structure and examples
- infra/README.md – environment, runners, and container notes

## Running the tests

```bash
pytest tests/                      # unit tests
LOCAL_DATASETS_DIR=/path/to/shar pytest tests/   # also runs the 5 that need real Shar data
```

CI does **not** re-run on every push to a pull request. It runs once when the PR
opens; after that, ask for it:

- comment **`/test`** on the pull request, or
- `gh workflow run ci.yml --ref <branch>`

A `/test` run is not attached to the PR's head commit, so it appears under the
[Actions tab](https://github.com/MELT-proj/training/actions) rather than as a
check on the PR; the comment gets a 👀 reaction when it starts.

## Main Components
- Training orchestrator: custom MELT Trainer built on HF Trainer
- Modeling choices: Hugging Face encoder/decoder classes with adapter support
- Data loading: Lhotse Shar pipelines for speech datasets

## Citation

```bibtex
@inproceedings{attanasio-etal-2025-instituto,
    title = "Instituto de Telecomunica{\c{c}}{\~o}es at {IWSLT} 2025: Aligning Small-Scale Speech and Language Models for Speech-to-Text Learning",
    author = "Attanasio, Giuseppe  and
      Sannigrahi, Sonal  and
      Peters, Ben  and
      Filipe Torres Martins, Andr{\'e}",
    editor = "Salesky, Elizabeth  and
      Federico, Marcello  and
      Anastasopoulos, Antonis",
    booktitle = "Proceedings of the 22nd International Conference on Spoken Language Translation (IWSLT 2025)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria (in-person and online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.iwslt-1.36/",
    doi = "10.18653/v1/2025.iwslt-1.36",
    pages = "347--353",
    ISBN = "979-8-89176-272-5",
    abstract = "This paper presents Instituto de Telecomunica{\c{c}}{\~o}es{'}s submission to the IWSLT 2025 Shared Task on Instruction Following Speech Processing. We submit results for the Short Track, i.e., speech recognition, translation, and spoken question answering. Our model is a unified speech-to-text model that integrates a pretrained continuous speech encoder and text decoder through a first phase of modality alignment and a second phase of instruction fine-tuning. Crucially, we focus on using small-scale language model backbones ({\ensuremath{<}} 2B) and restrict to high-quality, CC-BY data along with synthetic data generation to supplement existing resources."
}
```
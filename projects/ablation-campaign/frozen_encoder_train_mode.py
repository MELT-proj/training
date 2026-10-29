#!/usr/bin/env python3
"""How far do a frozen encoder's TRAIN-mode features sit from its EVAL-mode ones?

HF's Trainer calls ``model.train()`` on the whole model and MELT never puts a frozen
encoder back into eval mode, so layerdrop, dropout and (wav2vec2 family) SpecAugment fire
on the features the adapter trains on, while every evaluation sees clean ones. This
measures that gap per encoder, on real speech, on CPU -- no GPU, no training.

For each case it takes the encoder's last hidden state (what the adapter receives) for a
few real cuts, once in eval mode and ``--passes`` times in train mode under different
seeds, and reports the mean per-frame cosine similarity and the relative L2 error between
them. Whisper is the control: layerdrop 0, dropout 0, SpecAugment off, so its gap is
exactly zero. Run inside the training container (transformers, lhotse):

    LOCAL_DATASETS_DIR=... HF_HUB_OFFLINE=1 python3 frozen_encoder_train_mode.py

Measured 2026-09-26 on four cv22 English validation cuts (4.1-7.1 s), five passes each,
fp32 on CPU -- see plan/board.md that day. A different corpus or more cuts moves the
digits, not the ordering.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from lhotse import CutSet
from transformers import AutoFeatureExtractor, AutoModel

# (hub name, from_pretrained kwargs, label). The "pinned off" rows are what
# ABL-MA-700-asr.yaml's `model.encoder.apply_spec_augment: false` does.
CASES = [
    ("facebook/w2v-bert-2.0", {}, "w2v-BERT 2.0, as shipped"),
    ("openai/whisper-large-v3", {}, "Whisper-large-v3, as shipped"),
    ("utter-project/mHuBERT-147", {}, "mHuBERT-147, as shipped (spec-augment ON)"),
    ("utter-project/mHuBERT-147", {"apply_spec_augment": False}, "mHuBERT-147, spec-augment pinned off"),
    ("facebook/mms-1b", {}, "MMS-1b, as shipped (spec-augment ON)"),
    ("facebook/mms-1b", {"apply_spec_augment": False}, "MMS-1b, spec-augment pinned off"),
]


def load_audio(shar_dir: str, n_cuts: int) -> list[np.ndarray]:
    wavs = []
    for cut in CutSet.from_shar(in_dir=shar_dir):
        if 4.0 <= cut.duration <= 12.0:
            wavs.append(cut.resample(16000).load_audio()[0].astype(np.float32))
        if len(wavs) == n_cuts:
            break
    print(f"[data] {len(wavs)} cuts from {shar_dir}: {[round(len(w) / 16000, 1) for w in wavs]} s", flush=True)
    return wavs


def encode(model, feature_extractor, wav: np.ndarray) -> torch.Tensor:
    """The encoder's last hidden state for one utterance, valid frames only."""
    kind = type(feature_extractor).__name__
    if kind == "WhisperFeatureExtractor":
        feats = feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features
        out = model.get_encoder()(feats).last_hidden_state
        return out[0, : int(np.ceil(len(wav) / 16000 * 50))]  # Whisper pads to a 30 s window
    if kind == "Wav2Vec2FeatureExtractor":
        values = feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_values
        return model(values).last_hidden_state[0]
    feats = feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features
    return model(feats).last_hidden_state[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--shar",
        default=os.path.join(os.environ.get("LOCAL_DATASETS_DIR", ""), "cv22_sidon", "en", "validation"),
        help="a lhotse Shar directory of speech (default: cv22_sidon English validation)",
    )
    ap.add_argument("--cuts", type=int, default=4)
    ap.add_argument("--passes", type=int, default=5, help="seeded train-mode passes per cut")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    wavs = load_audio(args.shar, args.cuts)
    print(f"{'case':46s} {'cos(train,eval)':>16s} {'rel L2 err':>12s}   passes x cuts", flush=True)
    for name, kwargs, label in CASES:
        started = time.time()
        feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        # float32: Whisper's checkpoint would otherwise load as float16 and refuse float32 input
        model = AutoModel.from_pretrained(name, dtype=torch.float32, **kwargs)
        for p in model.parameters():  # frozen, exactly as MELT holds it
            p.requires_grad = False

        passes = 1 if "whisper" in name else args.passes  # nothing stochastic to sample in Whisper
        cosines, rel_errors = [], []
        with torch.no_grad():
            for wav in wavs:
                model.eval()
                reference = encode(model, feature_extractor, wav)
                model.train()  # what Trainer.training_step does, and MELT leaves alone
                for k in range(passes):
                    torch.manual_seed(1000 + k)
                    np.random.seed(1000 + k)
                    sample = encode(model, feature_extractor, wav)
                    n = min(reference.shape[0], sample.shape[0])
                    a, b = reference[:n].float(), sample[:n].float()
                    cosines.append(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item())
                    rel_errors.append(((a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-8)).mean().item())
        cfg = model.config
        print(
            f"{label:46s} {np.mean(cosines):16.4f} {np.mean(rel_errors):12.4f}   {passes} x {len(wavs)}"
            f"   [layerdrop={getattr(cfg, 'layerdrop', getattr(cfg, 'encoder_layerdrop', None))},"
            f" spec_aug={getattr(cfg, 'apply_spec_augment', None)}, {time.time() - started:.0f}s]",
            flush=True,
        )
        del model


if __name__ == "__main__":
    main()

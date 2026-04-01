import argparse
import json
import logging
import os
import torch
import soundfile as sf
from tqdm import tqdm
from melt.modeling import MELTForSequenceClassification, MELTProcessor
from datasets import Audio, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MELT inference on IWSLT26 dev set")

    # Model loading — mutually exclusive paths: full checkpoint vs base + adapter
    model_group = parser.add_argument_group("Model loading")
    model_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a full model checkpoint folder (used when no adapter is provided)",
    )
    model_group.add_argument(
        "--base_checkpoint",
        type=str,
        default=None,
        help="Path to the base model checkpoint (required when --adapter_checkpoint is set)",
    )
    model_group.add_argument(
        "--adapter_checkpoint",
        type=str,
        default=None,
        help="Path to a PEFT adapter checkpoint to apply on top of --base_checkpoint",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the JSONL results file",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/mnt/scratch-artemis/sonal/IWSLT26/data/audios/dev/",
        help="Directory containing the dev audio files",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="de",
        help="Target language for the translation prompt",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Log results and distribution artefacts to W&B (project: metric-iwslt26)",
    )
    return parser.parse_args()


def validate_args(args):
    using_adapter = args.adapter_checkpoint is not None
    if using_adapter and args.base_checkpoint is None:
        raise ValueError("--base_checkpoint is required when --adapter_checkpoint is set")
    if not using_adapter and args.checkpoint is None and args.base_checkpoint is None:
        raise ValueError("Provide either --checkpoint or --base_checkpoint (+ optional --adapter_checkpoint)")
    if args.checkpoint and (args.base_checkpoint or args.adapter_checkpoint):
        raise ValueError("--checkpoint is mutually exclusive with --base_checkpoint / --adapter_checkpoint")


def load_model(args, device):
    using_adapter = args.adapter_checkpoint is not None

    if using_adapter:
        from peft import PeftModel

        logger.info(f"Loading base model from: {args.base_checkpoint}")
        model = MELTForSequenceClassification.from_pretrained(args.base_checkpoint)
        logger.info(f"Applying PEFT adapter from: {args.adapter_checkpoint}")
        model = PeftModel.from_pretrained(model, args.adapter_checkpoint)
        model = model.merge_and_unload()
        logger.info("Adapter merged into base model")
        processor_path = args.adapter_checkpoint
    else:
        checkpoint = args.checkpoint or args.base_checkpoint
        logger.info(f"Loading full model from: {checkpoint}")
        model = MELTForSequenceClassification.from_pretrained(checkpoint)
        processor_path = checkpoint

    model.eval()
    model = model.to(device)
    logger.info(f"Model on {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, processor_path


def log_to_wandb(results, run_name):
    import wandb

    run = wandb.init(project="metric-iwslt26", name=run_name)

    scores = [r["score"] for r in results]
    trues = [r["true"] for r in results]

    # Results table
    table = wandb.Table(
        columns=["doc_id", "tgt_text", "score", "true"],
        data=[[r["doc_id"], r["tgt_text"], r["score"], r["true"]] for r in results],
    )
    run.log({"results": table})

    # Distribution histograms
    run.log({
        "hist/true_labels": wandb.Histogram(trues),
        "hist/predicted_scores": wandb.Histogram(scores),
    })

    logger.info(f"W&B run: {run.url}")
    run.finish()


def main():
    args = parse_args()
    validate_args(args)

    logger.info("Loading dataset...")
    dataset = load_dataset("maikezu/iwslt2026-metrics-shared-train-dev", split="dev")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    logger.info(f"Dataset loaded: {len(dataset)} examples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor_path = load_model(args, device)

    logger.info(f"Loading processor from: {processor_path}")
    processor = MELTProcessor.from_pretrained(processor_path)

    results = []

    for instance in tqdm(dataset, desc="Running inference"):
        doc_id = instance["doc_id"]
        true = instance["score"]
        audio_path = os.path.join(args.audio_dir, doc_id + ".wav")

        src_audio_array, sample_rate = sf.read(audio_path, dtype="float32")
        if src_audio_array.ndim > 1:
            src_audio_array = src_audio_array.mean(axis=1)

        tgt_text = instance["tgt_text"]
        text = (
            f"{processor.audio_token} Score how well this {args.tgt_lang} translation"
            f" matches the audio: {tgt_text}. Return a float between 0 and 1."
        )

        inputs = processor(text=text, audio=src_audio_array, return_tensors="pt", padding=False)
        inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits.squeeze().item()

        results.append({
            "doc_id": doc_id,
            "tgt_text": tgt_text,
            "score": score / 2.5,
            "true": true / 100,
        })

    logger.info(f"Writing {len(results)} results to: {args.output}")
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    if args.wandb:
        run_name = os.path.basename(args.adapter_checkpoint or args.checkpoint or args.base_checkpoint)
        logger.info(f"Logging artefacts to W&B (run: {run_name})...")
        log_to_wandb(results, run_name)

    logger.info("Done.")


if __name__ == "__main__":
    main()

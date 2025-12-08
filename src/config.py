from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


@dataclass
class TrainingArgsConfig:
    output_dir: str = "$SCRATCH/speech_lm/models/w2v-qwen25-05_align-iwslt"
    seed: int = 42
    do_train: bool = True
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    per_device_eval_batch_size: int = 2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    num_train_epochs: float = 1
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 2000
    logging_strategy: str = "steps"
    logging_steps: int = 25
    report_to: str | list[str] = "wandb"
    do_eval: bool = True
    evaluation_strategy: str = "steps"
    eval_steps: int = 3000
    save_strategy: str = "steps"
    save_total_limit: int = 5
    save_steps: int = 1000
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    bf16: bool = True
    group_by_length: bool = True


@dataclass
class Config:
    datasets: list[str] = field(default_factory=lambda: ["librispeech", "peoples_speech", "mls", "cv16.1"])
    selected_langs: list[str] = field(default_factory=lambda: ["en", "de", "it", "zh-CN", "zh-HK", "zh-TW"])
    dataset_workers: int = 8
    audio_encoder: str = "facebook/w2v-bert-2.0"
    text_decoder: str = "Qwen/Qwen2.5-0.5B"
    training_args: TrainingArgsConfig = field(default_factory=TrainingArgsConfig)
    encoder_lr: float = 6e-6
    decoder_lr: float = 2e-5
    adapter_lr: float = 2e-4
    min_lr_scale: float = 0.1
    use_flash_attention_2: bool = True
    val_samples_per_language: int = 3000
    encoder_params: dict[str, Any] = field(
        default_factory=lambda: {
            "add_adapter": False,
            "adapter_kernel_size": 3,
            "adapter_stride": 2,
            "num_adapter_layers": 2,
        }
    )
    decoder_params: dict[str, Any] = field(default_factory=dict)
    max_duration: int = 60
    min_chars: int = 3
    attn_implementation: str | None = None
    freeze_encoder: bool = False
    freeze_decoder: bool = False
    freeze_adapter: bool = False
    seed: int | None = None
    collator_args: dict[str, Any] = field(default_factory=dict)
    add_pre_adapter: bool = False
    num_pre_adapter_layers: int = 3
    ckpt: str | None = None


def override_dataclass(obj: Any, overrides: dict[str, Any]) -> Any:
    """Recursively override a dataclass (or nested dicts) with values from a dict."""

    for key, value in overrides.items():
        target_key = "evaluation_strategy" if key == "eval_strategy" else key
        if not hasattr(obj, target_key):
            continue

        current = getattr(obj, target_key)

        if is_dataclass(current) and isinstance(value, dict):
            override_dataclass(current, value)
        elif isinstance(current, dict) and isinstance(value, dict):
            current.update(value)
        else:
            setattr(obj, target_key, value)

    return obj


__all__ = [
    "asdict",
    "Config",
    "TrainingArgsConfig",
    "override_dataclass",
]

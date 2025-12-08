import logging
import os
import time
from dataclasses import asdict

import torch
import tyro
import yaml
from accelerate.logging import get_logger
from data_utils.utils import get_dataset
from datasets import Dataset, concatenate_datasets

import ddp
import transformers
from config import Config, override_dataclass
from melt import MELTConfig, MELTForConditionalGeneration, MELTProcessor
from training_utils import AudioTextDataCollator, MELTTrainer, count_trainable_parameters, filter_data
from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint


# from datasets import disable_caching
# disable_caching()
# datasets.config.IN_MEMORY_MAX_SIZE = 16_000_000_000

# Setup logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = get_logger(__name__)

# set because torch _inductor backend warning said it was a good idea
torch.set_float32_matmul_precision("high")


def prepare_model(
    cfg: Config, targs: TrainingArguments, processor: MELTProcessor
) -> (MELTForConditionalGeneration, str | None):
    # Prepare model configs
    audio_config = AutoConfig.from_pretrained(cfg.audio_encoder, **cfg.encoder_params)
    text_config = AutoConfig.from_pretrained(cfg.text_decoder, **cfg.decoder_params)
    config = MELTConfig(audio_encoder_config=audio_config, text_decoder_config=text_config)

    # The audi_bos_token is important in our forward and might change depending on the tokenizer used
    config.audio_bos_token_id = processor.tokenizer.convert_tokens_to_ids(["<|audio_bos|>"])[0]

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(targs.output_dir) and targs.do_train and not targs.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(targs.output_dir)
        if last_checkpoint is None and len(os.listdir(targs.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({targs.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and targs.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # TODO: these should be probably optimized with hparam tuning
    if cfg.ckpt is not None:
        logger.info(f"Loading model from checkpoint: {cfg.ckpt}")
        model = MELTForConditionalGeneration.from_pretrained(cfg.ckpt)
    else:
        model = MELTForConditionalGeneration(config)
        # resize the embedding matrix due to possibly new special tokens
        model.decoder.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False, pad_to_multiple_of=8)

    if cfg.freeze_adapter:
        logger.info("Freezing the adapter")
        model.freeze_adapter()
    if cfg.freeze_encoder:
        logger.info("Freezing the encoder")
        model.freeze_encoder()
    if cfg.freeze_decoder:
        logger.info("Freezing the decoder")
        model.freeze_decoder()

    return model, last_checkpoint


def main(cfg: Config = Config(), config_file: str | None = None, dry_run: bool = False):
    # Override defaults with values from YAML if provided
    if config_file is not None:
        with open(config_file) as f:
            overrides = yaml.safe_load(f) or {}
        cfg = override_dataclass(cfg, overrides)

    targs = TrainingArguments(**asdict(cfg.training_args))
    targs.output_dir = os.path.expandvars(targs.output_dir)
    targs.dataloader_num_workers = cfg.dataset_workers
    seed = cfg.seed if cfg.seed is not None else targs.seed

    # Basic setup
    set_seed(seed)

    if ddp.is_distributed():
        rank = ddp.get_global_rank()
        world_size = ddp.get_world_size()
        logger.info(f"Distributed setup: rank {rank} out of {world_size}")
        world_size = ddp.get_world_size()
        _local_world_size = ddp.get_local_world_size()
        local_rank = ddp.get_local_rank()
        is_local_master = ddp.is_local_master()
        is_global_master = ddp.is_global_master()
        is_distributed = ddp.is_distributed()
        logging.info(
            f"world_size: {world_size}, local_world_size: {ddp.get_local_world_size()}"
            f" local_rank: {local_rank}, group_rank: {ddp.get_group_rank()}"
            f" is_local_master: {is_local_master}, is_global_master: {is_global_master}"
            f" is_distributed: {is_distributed}"
        )
        # DDP blows up logging, so this is an attempt to suppress it to only logs from the master process
        logging.basicConfig(level=logging.INFO if is_local_master else logging.ERROR)
        # os.environ["TORCH_LOGS"] = "ERROR" if is_local_master else "WARNING"
        transformers.logging.set_verbosity(logging.WARNING if is_local_master else logging.ERROR)
        # hf_datasets.logging.set_verbosity(logging.WARNING if is_local_master else logging.ERROR)
    else:
        logger.info("Not in a distributed setup")

    ##########################
    ## DATA LOADING
    ##########################
    train_datasets = []
    val_datasets = []
    for dataset_name in cfg.datasets:
        logger.info(f"Loading dataset: {dataset_name}")
        data = get_dataset(
            dataset_name,
            splits=["train", "validation"],
            max_duration=cfg.max_duration,
            samples_validation=cfg.val_samples_per_language,
            selected_langs=cfg.selected_langs,
        )

        def add_column(ds, col_name):
            tmp = Dataset.from_dict({col_name: [""] * len(ds)})
            return concatenate_datasets([ds, tmp], axis=1)

        preamble_col = cfg.collator_args.get("preamble_col", None)
        if preamble_col is not None and preamble_col not in data["train"].column_names:
            logger.info(f"Adding empty preamble to the dataset: {preamble_col}")
            data["train"] = add_column(data["train"], preamble_col)
            data["validation"] = add_column(data["validation"], preamble_col)
            data["train"] = data["train"].filter(lambda x: isinstance(x, str), input_columns=[preamble_col])
            data["validation"] = data["validation"].filter(lambda x: isinstance(x, str), input_columns=[preamble_col])

        logger.info(f"Loaded dataset {dataset_name}", main_process_only=False)

        if dry_run:
            logger.info("Dry run, using a subset of the dataset")
            data["train"] = data["train"].select(range(1000))

        # filter a subset if it's a dry run
        train_datasets.append(data["train"])
        if targs.do_eval:
            val_datasets.append(data["validation"])

    datasets_count = len(train_datasets)
    logger.info(f"Loaded {datasets_count} training datasets")
    # At this stage, datasets are loaded with columns: audio, text, lang

    processor = MELTProcessor(
        feature_extractor=AutoFeatureExtractor.from_pretrained(cfg.audio_encoder),
        tokenizer=AutoTokenizer.from_pretrained(cfg.text_decoder, use_fast=True),
    )  # the processor takes care if special tokens are added to the tokenizer

    train_dataset = train_datasets[0] if datasets_count == 1 else concatenate_datasets(train_datasets)
    val_dataset = None
    if targs.do_eval:
        val_dataset = val_datasets[0] if datasets_count == 1 else concatenate_datasets(val_datasets)

    ### Preprocess the datasets and set transforms
    logger.info("Starting preprocessing... Tracking time")
    stime = time.time()

    # Remove every sample that has a duration longer than the max_duration
    # with targs.main_process_first():
    train_dataset = filter_data(train_dataset, cfg.max_duration, cfg.min_chars)
    if targs.do_eval:
        val_dataset = filter_data(val_dataset, cfg.max_duration, cfg.min_chars)

    logger.info(f"Preprocessing took {time.time() - stime:.2f} seconds")
    data_collator = AudioTextDataCollator(processor, **cfg.collator_args)
    logger.info(f"Number of rows: {len(train_dataset)}")

    ##########################
    ## MODEL PREPARATION
    ##########################
    model, last_checkpoint = prepare_model(cfg, targs, processor)
    logger.info("Model prepared!")

    # Print the number of learnable parameters
    trainable_params, trainable_str = count_trainable_parameters(model, return_int=True)
    logger.info(f"Total number of learnable parameters: {trainable_params} ({trainable_str})")

    ############################
    ## VALUES FOR THE OPTIMIZER
    ############################
    # These values are used within the custom trainer to setup optimizer and scheduler
    targs.encoder_lr = cfg.encoder_lr
    targs.decoder_lr = cfg.decoder_lr
    targs.adapter_lr = cfg.adapter_lr
    targs.min_lr_scale = cfg.min_lr_scale

    ##########################
    ## TRAINING
    ##########################

    trainer = MELTTrainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    if targs.resume_from_checkpoint is not None:
        checkpoint = targs.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    else:
        checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    logger.info(f"Train_result: {train_result}")

    ##########################
    ## SAVING COMPONENTS
    ##########################

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    # if is_dist and ddp.get_global_rank() == 0:
    #     torch.cuda.memory._dump_snapshot("profile.pkl")
    #     torch.cuda.memory._record_memory_history(enabled=None)

    trainer.save_model()
    processor.save_pretrained(targs.output_dir)


if __name__ == "__main__":
    tyro.cli(main)

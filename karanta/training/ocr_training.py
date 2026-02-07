import os
import time
import logging
import torch
import math
import wandb
import transformers

from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Dict
from contextlib import nullcontext
from torch.utils.data import DataLoader, random_split
from datasets import concatenate_datasets

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    InitProcessGroupKwargs,
    set_seed,
    DataLoaderConfiguration,
    DeepSpeedPlugin,
    ProfileKwargs,
)
from transformers import (
    AutoConfig,
    AutoModel,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    get_scheduler,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


import sys

sys.path.append("/home/oogundep/karanta-ocr/")
from karanta.training.utils import (
    ArgumentParserPlus,
    get_last_checkpoint_path,
    clean_last_n_checkpoints,
    save_with_accelerate,
)
from karanta.training.ocr_training_args import (
    ExperimentArguments,
    ModelArguments,
    DatasetArguments,
)
from karanta.training.data import LocalDataset, DataCollator
from karanta.training.muon_optimizer import SingleDeviceMuonWithAuxAdam

load_dotenv()
logger = get_logger(__name__)
logger.setLevel(logging.INFO)


def evaluate_model(
    model: torch.nn.Module,
    eval_dataloader: Dict[str, DataLoader] | DataLoader,
    accelerator,
    is_profile: bool = False,
    cm=nullcontext(),
):
    model.eval()
    eval_metrics = {}

    if isinstance(eval_dataloader, DataLoader):
        test_dict = {}
        test_dict["eval_split"] = eval_dataloader
        eval_dataloader = test_dict

    with torch.no_grad():
        with cm as prof:
            for eval_set, eval_set_dataloader in eval_dataloader.items():
                total_loss = 0.0
                num_batches = 0

                for batch in tqdm(eval_set_dataloader, desc=f"Evaluating {eval_set}"):
                    if batch is None:
                        continue

                    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    batch_loss = outputs.loss.item()

                    if torch.isnan(outputs.loss).sum().item() > 0:
                        continue
                    else:
                        num_batches += 1
                        total_loss += batch_loss

                eval_loss = total_loss / num_batches
                eval_metrics[f"{eval_set}_loss"] = eval_loss

            # Average eval loss across all datasets
            eval_loss = sum(eval_metrics.values()) / len(eval_metrics)
            eval_metrics["eval_loss"] = eval_loss

    if is_profile and prof is not None:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    return eval_metrics


def main(args: ArgumentParserPlus):
    exp_args, model_args, data_args = args[0], args[1], args[2]

    exp_args.output_dir = os.path.join(exp_args.output_dir, exp_args.exp_name)

    exp_args.run_name = f"{exp_args.exp_name}__{exp_args.seed}__{int(time.time())}"
    if exp_args.push_to_hub:
        exp_args.run_name = f"{exp_args.run_name}__hub"

        if exp_args.hf_entity is None:
            exp_args.hf_entity = "taresco"
        if exp_args.hf_repo_id is None:
            exp_args.hf_repo_id = f"{exp_args.hf_entity}/{exp_args.exp_name}"
        if exp_args.hf_repo_revision is None:
            exp_args.hf_repo_revision = exp_args.run_name

        exp_args.hf_repo_url = f"https://huggingface.co/{exp_args.hf_repo_id}/tree/{exp_args.hf_repo_revision}"

    accelerator_log_kwargs = {}

    if exp_args.with_tracking:
        accelerator_log_kwargs["log_with"] = exp_args.reports_to
        accelerator_log_kwargs["project_dir"] = exp_args.output_dir

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=exp_args.timeout))
    dataloader_config = DataLoaderConfiguration()
    dataloader_config.use_seedable_sampler = True

    if exp_args.is_profile:
        profile_kwargs = ProfileKwargs(
            activities=["cpu", "cuda"],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        )
        all_kwargs = [timeout_kwargs, profile_kwargs]
        exp_args.num_train_epochs = 1  # when profiling, only do one epoch
    else:
        all_kwargs = [timeout_kwargs]

    if exp_args.use_deepspeed:
        deepspeed_config = {
            "bf16": {"enabled": "auto"},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "zero_optimization": {
                "stage": 2,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "steps_per_print": 1e5,
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "wall_clock_breakdown": False,
        }

        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=deepspeed_config,
            zero_stage=2,
            gradient_accumulation_steps=exp_args.gradient_accumulation_steps,
            gradient_clipping="auto",
            offload_param_device="cpu",
        )

        accelerator = Accelerator(
            gradient_accumulation_steps=exp_args.gradient_accumulation_steps,
            dataloader_config=dataloader_config,
            deepspeed_plugin=deepspeed_plugin,
            **accelerator_log_kwargs,
            kwargs_handlers=all_kwargs,
            mixed_precision="bf16",
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=exp_args.gradient_accumulation_steps,
            dataloader_config=dataloader_config,
            **accelerator_log_kwargs,
            kwargs_handlers=all_kwargs,
            mixed_precision="bf16",
        )

    if exp_args.is_profile:
        cm = accelerator.profile()
    else:
        cm = nullcontext()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if exp_args.seed:
        set_seed(exp_args.seed)

    # Create output directory in the main process
    if accelerator.is_main_process:
        if exp_args.output_dir is not None:
            os.makedirs(exp_args.output_dir, exist_ok=True)

    # Initialize the training dataset
    with accelerator.main_process_first():
        train_dataset = []
        for i, data_mix in enumerate(data_args.dataset_train):
            logger.info(
                f"Creating training dataset {i + 1} from: {data_mix.get('root_dir', None)}"
            )
            pipeline_mix = data_mix.get("pipeline", None)

            ld = LocalDataset(
                root_dir=Path(data_mix["root_dir"]),
                pdf_dir_name=data_mix["pdf_dir_name"],
                json_dir_name=data_mix["json_dir_name"],
                pipeline_steps=pipeline_mix,
                cache_folder_name=data_args.data_cache_folder_name,
                num_samples=data_args.num_samples,
            )

            logger.info(f"Found {len(ld)} samples")

            if len(ld) > 0:
                train_dataset.append(ld.dataset)

        # Combine all training datasets
        train_dataset = (
            concatenate_datasets(train_dataset)
            if len(train_dataset) > 1
            else train_dataset[0]
        )
        logger.info(f"Total training samples: {len(train_dataset)}")
        data_collator = DataCollator(
            model_args.model_name_or_path, max_token_len=data_args.max_length
        )
        train_dataset = train_dataset.shuffle(seed=exp_args.seed)

        if not data_args.dataset_eval:
            train_dataset, eval_dataset = random_split(train_dataset, [0.999, 0.001])
            eval_dl = DataLoader(
                eval_dataset,
                collate_fn=data_collator,
                batch_size=exp_args.per_device_eval_batch_size,
                shuffle=True,
                drop_last=True,
            )
            eval_dataloaders = eval_dl

        else:
            # Initialize the evaluation dataset
            eval_dataloaders = {}
            for i, data_mix in enumerate(data_args.dataset_eval):
                logger.info(
                    f"Creating eval dataset {i + 1} from: {data_mix.get('root_dir', None)}"
                )
                pipeline_mix = data_mix.get("pipeline", None)
                dataset = LocalDataset(
                    root_dir=Path(data_mix["root_dir"]),
                    pdf_dir_name=data_mix["pdf_dir_name"],
                    json_dir_name=data_mix["json_dir_name"],
                    pipeline_steps=pipeline_mix,
                )
                logger.info(f"Found {len(dataset)} samples")

                eval_dl = DataLoader(
                    dataset,
                    collate_fn=data_collator,
                    batch_size=exp_args.per_device_eval_batch_size,
                    shuffle=True,
                )
                eval_dataloaders[data_mix.get("name", str(i))] = eval_dl

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=exp_args.per_device_train_batch_size,
        num_workers=data_args.dataloader_num_workers,
        drop_last=True,
        pin_memory=True
    )
    accelerator.wait_for_everyone()

    # Initialize Model Config if not provided
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You need to specify either a model name or a model configuration name"
        )

    # Initialize the model
    if model_args.model_name_or_path:
        if "Qwen2.5-VL" in model_args.model_name_or_path:
            model_class = Qwen2_5_VLForConditionalGeneration
        elif "Qwen2-VL" in model_args.model_name_or_path:
            model_class = Qwen2VLForConditionalGeneration
        elif "Qwen3-VL" in model_args.model_name_or_path:
            model_class = Qwen3VLForConditionalGeneration

        if model_args.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}  # force data-parallel training.

            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                revision=model_args.model_revision,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                trust_remote_code=model_args.trust_remote_code,
                quantization_config=quantization_config if model_args.use_qlora else {},
                device_map=device_map,
                attn_implementation="flash_attention_2"
                if exp_args.use_flash_attention
                else "eager",
            )
        else:
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                revision=model_args.model_revision,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                trust_remote_code=model_args.trust_remote_code,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
                if exp_args.use_flash_attention
                else "eager",
            )
    else:
        logger.info("Training from scratch, no weights or quantization needed")
        model = AutoModel.from_config(config)

    if model_args.use_lora:
        if model_args.use_qlora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=model_args.gradient_checkpointing
            )

        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=[
                "q_proj",
                "o_proj",
                "v_proj",
                "k_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif model_args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    # Compiling the model if specified
    if model_args.torch_compile:
        logger.info(
            f"Compiling model with torch.compile (backend={model_args.torch_compile_backend}, mode={model_args.torch_compile_mode})"
        )
        model = torch.compile(
            model,
            backend=model_args.torch_compile_backend,
            mode=model_args.torch_compile_mode,
            fullgraph=model_args.torch_compile_fullgraph,
            dynamic=model_args.torch_compile_dynamic,
        )
        logger.info("Model compilation complete")

    # Set up optimizer
    if exp_args.optimizer == "adamw_torch":
        no_decay = [
            "bias",
            "LayerNorm.weight",
            "layer_norm.weight",
            "ln_f.weight",
            "norm.weight",
        ]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": exp_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=float(exp_args.learning_rate),
        )
    elif exp_args.optimizer == "muon":
        # Separate parameters for Muon (hidden matrices) and Adam (embeddings, scalars, head)
        hidden_matrix_params = [
            p
            for n, p in model.named_parameters()
            if p.ndim >= 2 and "embed" not in n and "lm_head" not in n
        ]
        embed_params = [p for n, p in model.named_parameters() if "embed" in n]
        scalar_params = [p for p in model.parameters() if p.ndim < 2]
        head_params = [p for n, p in model.named_parameters() if "lm_head" in n]

        # Create Adam groups with different learning rates
        adam_groups = [
            dict(
                params=head_params,
                lr=float(exp_args.learning_rate) * 0.8,
                use_muon=False,
            ),
            dict(
                params=embed_params,
                lr=float(exp_args.learning_rate) * 12.0,
                use_muon=False,
            ),
            dict(
                params=scalar_params,
                lr=float(exp_args.learning_rate) * 0.8,
                use_muon=False,
            ),
        ]

        # Add Adam hyperparameters to groups
        for g in adam_groups:
            g["betas"] = (0.8, 0.95)
            g["eps"] = float(1e-10)
            g["weight_decay"] = 1e-10

        # Create Muon group
        muon_group = dict(
            params=hidden_matrix_params,
            lr=float(exp_args.learning_rate),
            momentum=0.95,
            weight_decay=1e-10,
            use_muon=True,
        )

        # Combine all groups
        param_groups = [*adam_groups, muon_group]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        raise NotImplementedError(
            f"Optimizer {exp_args.optimizer} not supported in custom loop"
        )

    # Total training steps calculation
    samples_per_step = (
        exp_args.per_device_train_batch_size * exp_args.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / samples_per_step)
    max_train_steps = int(
        math.ceil(exp_args.num_train_epochs * num_update_steps_per_epoch)
    )
    max_train_samples = int(math.ceil(exp_args.num_train_epochs * len(train_dataset)))

    # Set up scheduler
    lr_scheduler = get_scheduler(
        name=exp_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * exp_args.warmup_ratio),
        num_training_steps=max_train_steps,
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / exp_args.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    exp_args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = exp_args.checkpointing_steps
    if checkpointing_steps and str(checkpointing_steps).lower() != "epoch":
        checkpointing_steps = int(checkpointing_steps)
    elif (
        checkpointing_steps
        and isinstance(checkpointing_steps, str)
        and checkpointing_steps.lower() == "epoch"
    ):
        checkpointing_steps = num_update_steps_per_epoch

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if exp_args.with_tracking:
        experiment_config = vars(exp_args)
        dataset_args = vars(data_args)
        modelling_args = vars(model_args)

        all_config = {**experiment_config, **dataset_args, **modelling_args}

        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]

        if exp_args.wandb_entity is None:
            raise ValueError("Please provide a wandb entity.")

        accelerator.init_trackers(
            exp_args.wandb_project_name,
            all_config,
            init_kwargs={
                "wandb": {
                    "name": exp_args.run_name,
                    "entity": exp_args.wandb_entity,
                    "tags": [exp_args.exp_name],
                }
            },
        )
        _wandb_tracker = accelerator.get_tracker("wandb")

    # Train!
    total_batch_size = (
        exp_args.per_device_train_batch_size
        * accelerator.num_processes
        * exp_args.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {exp_args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {exp_args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {exp_args.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )

    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    last_checkpoint_path = get_last_checkpoint_path(exp_args, True)
    if last_checkpoint_path:
        accelerator.print(f"Resumed from checkpoint: {last_checkpoint_path}")
        accelerator.load_state(last_checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        last_checkpoint_path = os.path.basename(last_checkpoint_path)
        training_difference = os.path.splitext(last_checkpoint_path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * exp_args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // exp_args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    if accelerator.is_main_process and exp_args.report_to == "wandb":
        wandb.watch(model, log="all", log_freq=exp_args.logging_steps)

    # Evaluate before training
    # if accelerator.is_main_process:
    #     eval_metrics = evaluate_model(
    #         model, eval_dataloaders, accelerator, is_profile=exp_args.is_profile, cm=cm
    #     )
    #     accelerator.log(eval_metrics, step=completed_steps)
    #     logger.info(f"Evaluation results at step {completed_steps} is  {eval_metrics}")

    accelerator.wait_for_everyone()

    logger.info(f"Starting from epoch {starting_epoch} and step {completed_steps}.")
    logger.info(
        f"Total training steps: {max_train_steps}, Total samples to process: {max_train_samples}"
    )

    progress_bar.update(completed_steps)
    start_time = time.time()

    for epoch in range(starting_epoch, exp_args.num_train_epochs):
        model.train()
        train_dataloader.set_epoch(epoch)
        total_loss = 0
        local_total_tokens = 0
        total_token_including_padding = 0

        if last_checkpoint_path is not None and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, num_batches=resume_step
            )
        else:
            active_dataloader = train_dataloader

        with cm as prof:
            for step, batch in enumerate(active_dataloader):
                if batch is None:
                    # in case of an empty batch, we create a dummy loss and call backward
                    # so all ranks still perform a backward step (and hit the same collectives).
                    dummy_loss = torch.tensor(
                        0.0, device=accelerator.device, requires_grad=True
                    )
                    accelerator.backward(dummy_loss)
                    continue

                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                local_total_tokens += batch["attention_mask"].sum()
                total_token_including_padding += batch["attention_mask"].numel()

                with accelerator.accumulate(model):
                    output = model(**batch)
                    loss = output.loss

                    total_loss += loss.detach().float()
                    accelerator.backward(loss)

                    # Gradient clipping is not needed when using deepspeed
                    if not exp_args.use_deepspeed:
                        if accelerator.sync_gradients and exp_args.max_norm > 0:
                            accelerator.clip_grad_norm_(
                                model.parameters(), exp_args.max_norm
                            )

                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                    if (
                        completed_steps > 0
                        and exp_args.eval_steps
                        and completed_steps % exp_args.eval_steps == 0
                    ):
                        if accelerator.is_main_process:
                            eval_metrics = evaluate_model(
                                model, eval_dataloaders, accelerator
                            )
                            accelerator.log(eval_metrics, step=completed_steps)
                            logger.info(
                                f"Evaluation results at step {completed_steps} is  {eval_metrics}"
                            )
                        accelerator.wait_for_everyone()
                        model.train()

                    if (
                        exp_args.is_profile
                        and exp_args.profile_steps
                        and completed_steps % exp_args.profile_steps == 0
                    ):
                        if accelerator.is_main_process:
                            # write profile results to output dir
                            if exp_args.output_dir is not None:
                                with open(
                                    os.path.join(
                                        exp_args.output_dir,
                                        f"profile_step_{completed_steps}.txt",
                                    ),
                                    "w",
                                ) as f:
                                    f.write(
                                        prof.key_averages().table(
                                            sort_by="cpu_time_total", row_limit=15
                                        )
                                    )
                                    f.write("\n\n")
                                    f.write(
                                        prof.key_averages().table(
                                            sort_by="cuda_time_total", row_limit=15
                                        )
                                    )
                                    f.write("\n\n")
                                    f.write(
                                        prof.key_averages().table(
                                            sort_by="cuda_memory_usage", row_limit=15
                                        )
                                    )
                                    f.write("\n\n")
                        break

                    if (
                        exp_args.logging_steps
                        and completed_steps % exp_args.logging_steps == 0
                    ):
                        avg_loss = (
                            accelerator.gather(total_loss).mean().item()
                            / exp_args.gradient_accumulation_steps
                            / exp_args.logging_steps
                        )
                        total_tokens = (
                            accelerator.gather(local_total_tokens).sum().item()
                        )
                        total_tokens_including_padding = accelerator.reduce(
                            torch.tensor(
                                total_token_including_padding, device=accelerator.device
                            ),
                            reduction="sum",
                        ).item()
                        training_metrics_to_log = {
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "train_loss": avg_loss,
                            "total_tokens": total_tokens,
                            "per_device_tps": total_tokens
                            / accelerator.num_processes
                            / (time.time() - start_time),
                            "total_tokens_including_padding": total_tokens_including_padding,
                            "per_device_tps_including_padding": total_tokens_including_padding
                            / accelerator.num_processes
                            / (time.time() - start_time),
                        }

                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, TPS: {total_tokens / (time.time() - start_time)}"
                        )

                        if exp_args.with_tracking:
                            accelerator.log(
                                training_metrics_to_log, step=completed_steps
                            )

                        total_loss = 0

                if completed_steps > 0 and completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if exp_args.output_dir is not None:
                        output_dir = os.path.join(exp_args.output_dir, output_dir)

                    accelerator.save_state(output_dir)
                    accelerator.wait_for_everyone()

    if exp_args.output_dir is not None:
        save_with_accelerate(
            accelerator,
            model,
            exp_args.output_dir,
            model_args.use_lora,
        )

    # remove all checkpoints to save space
    if accelerator.is_main_process:
        clean_last_n_checkpoints(exp_args.output_dir, keep_last_n_checkpoints=2)

    # Final evaluation
    if accelerator.is_main_process:
        final_metrics = evaluate_model(model, eval_dataloaders, accelerator)
        logger.info(f"Final evaluation metrics: {final_metrics}")

    accelerator.wait_for_everyone()

    if exp_args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParserPlus((ExperimentArguments, ModelArguments, DatasetArguments))
    args = parser.parse()
    main(args)

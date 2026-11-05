"""Shared argument parsing, data loading, and LoRA training utilities."""

import json
import os
import random

import datasets
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def add_training_arguments(parser):
    parser.add_argument("--base_model", required=True, help="Base model to fine-tune")
    parser.add_argument("--sft_data_path", required=True, help="Rank1 training JSON")
    parser.add_argument("--output_model_path", required=True, help="Output directory")
    parser.add_argument("--logging_steps", default=20, type=int)
    parser.add_argument("--per_device_train_batch_size", default=16, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--lora_r", default=32, type=int)
    parser.add_argument("--lora_alpha", default=64, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--seed", default=23, type=int)


def load_training_dataset(path, convert_row):
    with open(path, encoding="utf-8") as input_file:
        rows = json.load(input_file)
    return datasets.Dataset.from_list([convert_row(row) for row in rows])


def find_all_linear_names(model):
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            module_names.add(names[0] if len(names) == 1 else names[-1])
    module_names.discard("lm_head")
    return list(module_names)


def train_lora_adapter(args, train_data, variant):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=None,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            task_type="CAUSAL_LM",
            target_modules=find_all_linear_names(model),
            inference_mode=False,
        ),
    )
    model.train()

    model_output_dir = os.path.join(
        args.output_model_path,
        f"{os.path.basename(args.base_model)}_{variant}",
    )
    effective_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    steps_per_epoch = len(train_data) / effective_batch_size

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        do_train=True,
        save_strategy="steps",
        save_steps=max(1, int(steps_per_epoch // 5)),
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        seed=args.seed,
        disable_tqdm=False,
        dataloader_pin_memory=False,
    )
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_data)
    trainer.train()
    trainer.save_model(model_output_dir)

from huggingface_hub import login
import os
import gc
import wandb
import torch
from merge_lora import merge_and_save_model
from config import get_default_config
from model import get_base_model
from transformers import (
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import argparse


#bookkeeping
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_samples', dest='num_samples', metavar='num_samples', required=False, help = 'Number of samples to use to finetune the model')

os.environ["WANDB_PROJECT"] = "python-autocomplete-deepseek"
wandb.login()
login()


def train(model, train_dataset, eval_dataset, tokenizer, config):
    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        r=config.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim=config.optim,
        save_steps=config.save_steps,
        evaluation_strategy = config.eval_strategy,
        eval_steps = config.eval_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        bf16=config.bf16,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        group_by_length=config.group_by_length,
        lr_scheduler_type=config.lr_scheduler_type,
        report_to="wandb"
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset = eval_dataset,
        peft_config=peft_config,
        dataset_text_field="code",
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=config.packing,
    )

    trainer.train()
    wandb.run.finish()

    trainer.model.save_pretrained(config.new_model)
    del trainer

def flush_artifacts(model, tokenizer, train_dataset, eval_dataset):
    del model
    del tokenizer
    del train_dataset
    del eval_dataset

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    """
        Login to huggingface and Wandb, get the base model, then train it with SFTTrainer
    """
    args = parser.parse_args()
    num_samples = args.num_samples
    default_config = get_default_config()
    model, tokenizer, train_dataset, eval_dataset = get_base_model(default_config, num_samples)
    
    #train and save model
    train(model, train_dataset, eval_dataset, tokenizer, default_config)
    flush_artifacts(model, tokenizer, train_dataset, eval_dataset)
    
    #don't need to keep the above model, since we saved it locally. The below will reload it and the base model and merge them.
    merge_and_save_model(default_config)


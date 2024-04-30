from transformers import Trainer, TrainingArguments
import os
from huggingface_hub import login
import wandb
from model import init_model
import argparse

#bookkeeping
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_samples', dest='num_samples', metavar='num_samples', required=False, help = 'Number of samples to use to finetune the model')


#bookkeeping
wandb.login()
login()
os.environ["WANDB_PROJECT"] = "python-gpt2"

def train(num_samples):
    model, tokenizer, tokenized_datasets, data_collator = init_model(num_samples)

    args = TrainingArguments(
    output_dir = "python-gpt2",
    per_device_train_batch_size= 32,
    per_device_eval_batch_size = 32,
    evaluation_strategy="steps",
    eval_steps=25,
    logging_steps=25,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps = 50,
    fp16 = True,
    report_to="wandb"
)

    trainer = Trainer(
        model = model,
        tokenizer=tokenizer,
        args = args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"]
    )

    trainer.train()
    wandb.run.finish()
    trainer.push_to_hub()

if __name__ == "__main__":
    args = parser.parse_args()
    num_samples = args.num_samples
    train(num_samples)




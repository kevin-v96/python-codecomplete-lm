import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

def get_tokenizer(config):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    return tokenizer


def init_dataset(config, num_samples):
    dataset = load_dataset(config.dataset_name)

    #Shuffle the dataset and slice it
    #Bigger slice for training
    train_dataset = dataset['train'].shuffle(seed=42).select(range(num_samples))
    eval_dataset = dataset['train'].shuffle(seed=42).select(range(25))
    return train_dataset, eval_dataset

def get_base_model(config, num_samples = 1000):
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    #we use bitsandbytes to load the model in 4-bit. 
    #This massively decreases its memory footprint and allows us to fine-train it on a free tier Colab notebook
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map=config.device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = get_tokenizer(config)
    train_dataset, eval_dataset = init_dataset(config, num_samples)

    return model, tokenizer, train_dataset, eval_dataset
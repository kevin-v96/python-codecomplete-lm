import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict

context_length = 256 # what is the maximum context length for predictions?
torch.manual_seed(42)

def tokenize(tokenizer, element):
    outputs = tokenizer(
    element["code"],
    truncation = True,
    max_length = context_length,
    return_overflowing_tokens = True,
    return_length = True
)
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)

    return {
        "input_ids": input_batch
    }

def get_tokenized_dataset(num_samples = 50000):
    raw_dataset = load_dataset("ArtifactAI/arxiv_python_research_code")
    ds_train = raw_dataset["train"]
    raw_datasets = DatasetDict(
        {
            "train": ds_train.shuffle().select(range(num_samples)),
            "valid": ds_train.shuffle().select(range(500))
        }
    )
    tokenized_datasets = raw_datasets.map(
    tokenize, batched = True, remove_columns = raw_datasets["train"].column_names
)
    return tokenized_datasets


def get_data_collator(tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
    return data_collator
   


def init_model(num_samples):
   tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
   config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size = len(tokenizer),
    n_ctx = context_length,
    bos_token_id = tokenizer.bos_token_id,
    eos_token_id = tokenizer.eos_token_id
)
   model = GPT2LMHeadModel(config)
   model_size = sum(t.numel() for t in model.parameters())
   print(f"GPT2 size: {model_size/1000**2:.1f}M parameters")

   data_collator = get_data_collator(tokenizer)
   tokenized_datasets = get_tokenized_dataset(num_samples)
   return model, tokenizer, tokenized_datasets, data_collator
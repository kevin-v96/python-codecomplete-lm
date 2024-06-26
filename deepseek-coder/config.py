from types import SimpleNamespace
import copy

config = SimpleNamespace(
  # The model that you want to train from the Hugging Face hub
model_name = "deepseek-ai/deepseek-coder-1.3b-base",

# The instruction dataset to use
dataset_name = "ArtifactAI/arxiv_python_research_code",

# Fine-tuned model name
new_model = "deepseek-coder-1.3b-python-peft",

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64,

# Alpha parameter for LoRA scaling
lora_alpha = 16,

# Dropout probability for LoRA layers
lora_dropout = 0.1,

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True,

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16",

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4",

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False,

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results",

# Number of training epochs
num_train_epochs = 15,

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False,
bf16 = False,

# Batch size per GPU for training
per_device_train_batch_size = 4,

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4,

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1,

# Enable gradient checkpointing
gradient_checkpointing = True,

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.001,

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-3,

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001,

# Optimizer to use
optim = "paged_adamw_32bit",

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = 'cosine_with_restarts',

# Number of training steps (overrides num_train_epochs)
max_steps = -1,

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03,

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True,

# Save checkpoint every X updates steps
save_steps = 25,

# Log every X updates steps
logging_steps = 25,

#eval every X steps
eval_steps = 25,

#evaluation strategy
eval_strategy = 'steps',

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None,

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False,

# Load the entire model on the GPU 0
device_map = {"": 0},
)

def get_default_config():
    new_config = copy.copy(config)
    return new_config
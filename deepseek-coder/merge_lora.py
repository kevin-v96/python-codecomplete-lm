import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

def merge_and_save_model(config):
    """
    Reload model in FP16 and merge it with LoRA weights, upload it to HuggingFace
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=config.device_map,
    )
    model = PeftModel.from_pretrained(base_model, config.new_model)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.push_to_hub(config.new_model, use_temp_dir=False)
    tokenizer.push_to_hub(config.new_model, use_temp_dir=False)

# Finetuned DeepSeek-Coder-1.3B
This is a PEFT (Parameter-efficient Fine-Tune) of the [DeepSeek-Coder-1.3B model](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) on the [ArtifactAI/arxiv_python_research_code](https://huggingface.co/datasets/ArtifactAI/arxiv_python_research_code) dataset.

### Final Model
You can fine the model I finetuned and uploaded [here](https://huggingface.co/MadMarx37/deepseek-coder-1.3b-python-peft).

### Hardware requirements
The original DeepSeek-Coder model is FP16, which means it is half-precision. It has 1.35B parameters.
Calculating memory usage (with some caveats) can be done with the equation $`memory_usage = num_parameters * bytes_per_param`$
In our case, that is $`memory_usage = 1350000000 * 2`$ (since 16bits/param = 16 / 8 = 2 bytes/param)

Thus, our the base model needs $`2700000000 bytes / (1024^3 bytes/gigabyte) =~ 2.5GBs`$
This is usually for inference, and for training, gradients and optimisers mean we have overhead that requires more memory. Consider 3-4x this for training since we're on the lower end of model size.

### Quantization
quantize.ipynb has code to quantize the model to the q5_k_m (5-bit) model, which is easier to use with consumer GPUs, requiring a maximum of 3.5GB VRAM (see [here](https://huggingface.co/TheBloke/deepseek-coder-1.3b-base-GGUF))

Quantized models in the GGUF format can be loaded and used in Python as such:
```
from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained("MadMarx37/deepseek-coder-1.3b-python-peft-GGUF", model_file="deepseek-coder-1.3b-python-peft.Q5_K_M.gguf", model_type="deepseek", gpu_layers=50)

print(llm("AI is going to"))
```

The quantization saw some drop in quality, likely dues to some issues with mismatch of special tokens in the tokenizer. In the interest of time, I've kept it as an error to rectify later.

### Finetuning metrics
You can find the process I followed for finetuning including the hyperparameter sweep in the Jupyter Notebook. 
Alternatively, you can fine a report about the same (including training and eval plots) [here](https://api.wandb.ai/links/kevinv3796/nzt0ndnq).

## Using the model
Run `pip install -r requirements.txt` to install the required modules.

### Tuning your own
If you want to tune your own model, you can run `python train.py`. The default number of samples from the above dataset used
to finetune the model is 1000, but you can provide your own number with `python train.py -n <number of samples>`

### Prediction
You can run prediction by running `python predict.py <prompt>`, which will run prediction using the model I trained.
If you want to use the model you trained instead, you can replace the `model_name` attribute in `predict.py` and proceed as above.
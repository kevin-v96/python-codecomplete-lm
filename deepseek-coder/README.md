# Finetuned DeepSeek-Coder-1.3B
This is a PEFT (Parameter-efficient Fine-Tune) of the [DeepSeek-Coder-1.3B model](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) on the [ArtifactAI/arxiv_python_research_code](https://huggingface.co/datasets/ArtifactAI/arxiv_python_research_code) dataset.

### Final Model
You can fine the model I finetuned and uploaded [here](https://huggingface.co/MadMarx37/deepseek-coder-1.3b-python-peft).

### Hardware requirements
The original DeepSeek-Coder model is FP16, which means it is half-precision. It has 1.35B parameters.
Calculating memory usage (with some caveats) can be done with the equation $`memory_usage = num_parameters * bytes_per_param`$
In our case, that is $`memory_usage = 1350000000 * 2`$ (since 16bits/param = 16 / 8 = 2 bytes/param)

Thus, our the base model needs $`2700000000 bytes / (1024^3 bytes/gigabyte) =~ 2.5GBs`$
This is usually for inference, and for training, gradients and optimisers mean we have overhead that requires more memory.

### Finetuning metrics
You can find the process I followed for finetuning including the hyperparameter sweep in the Jupyter Notebook. 
Alternatively, you can fine a report about the same (including training and eval plots) [here](https://api.wandb.ai/links/kevinv3796/nzt0ndnq).

### Tuning your own
If you want to tune your own model, you can run `python train.py`. The default number of samples from the above dataset used
to finetune the model is 1000, but you can provide your own number with `python train.py -n <number of samples>`

### Prediction
You can run prediction by running `python predict.py <prompt>`, which will run prediction using the model I trained.
If you want to use the model you trained instead, you can replace the `model_name` attribute in `predict.py` and proceed as above.
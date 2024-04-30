# Finetuned DeepSeek-Coder-1.3B
This is a PEFT (Parameter-efficient Fine-Tune) of the [DeepSeek-Coder-1.3B model](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) on the [ArtifactAI/arxiv_python_research_code](https://huggingface.co/datasets/ArtifactAI/arxiv_python_research_code) dataset.

### Final Model
You can fine the model I finetuned and uploaded [here](https://huggingface.co/MadMarx37/deepseek-coder-1.3b-python-peft).

### Finetuning metrics
You can find the process I followed for finetuning including the hyperparameter sweep in the Jupyter Notebook. 
Alternatively, you can fine a report about the same (including training and eval plots) [here](https://api.wandb.ai/links/kevinv3796/nzt0ndnq).

### Tuning your own
If you want to tune your own model, you can run `python train.py`. The default number of samples from the above dataset used
to finetune the model is 1000, but you can provide your own number with `python train.py -n <number of samples>`

### Prediction
You can run prediction by running `python predict.py <prompt>`, which will run prediction using the model I trained.
If you want to use the model you trained instead, you can replace the `model_name` attribute in `predict.py` and proceed as above.
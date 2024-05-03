# python-codecomplete-lm
Some small LMs (less than or around 1B parameters) for next token prediction in Python

## Models
Models I tried (you can find instructions about each in their own folders):

### First model: DeepSeek Coder 1.3B Parameter-efficient fine-tune
The [evalplus](https://evalplus.github.io/leaderboard.html) leaderboard has a list of the best current LLMs for code. For our size requirements (close to ~1B), the [DeepSeek Coder 1.3B base model](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) was ideal. I followed the steps given [here](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html) to fine-tune models in a Parameter-efficient way using methods like LoRA (Low Rank Adapters that are aligned with and trained instead of the model weights, which are frozen) and QLoRA (LoRA that works on Quantized LLMs, like those loaded in 4bits with bitsandbytes). This took about ~2 hours of training. 

### Second model: GPT2 from scratch
To start, I trained a small GPT2 model (~124M parameters) from scratch using the data we have. This Huggingface [tutorial](https://huggingface.co/learn/nlp-course/chapter7/6) goes into how to do this. This is cumbersome to do in some ways - training a tokenizer (even the fast tokenizers from HF written in Rust) takes a few hours since tokenizer training is CPU-bound. 

So I used the tokenizer from the above model. This took a few hours off our total training time. Then I followed the standard training procedure outlined in the tutorial with different sizes of data, trying different dataset sizes to see what's realistic to train on a free Google Clab and get decent results.

### Third model: Decoder-only Transformer from scratch in PyTorch
I wrote this model in PyTorch using a standard decoder-only Transformer architecture for generation. 

## Eval

## Outputs

## References
- https://evalplus.github.io/leaderboard.html
- https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base
- https://huggingface.co/learn/nlp-course/chapter7/6
- https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html
- https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/training.ipynb
- https://huggingface.co/docs/transformers/training
- https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse
- https://pytorch.org/tutorials/beginner/transformer_tutorial.html 
- https://wandb.ai/matt24/vit-snacks-sweeps/reports/Hyperparameter-Search-for-HuggingFace-Transformer-Models--VmlldzoyMTUxNTg0
- https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html 
- https://github.com/lucidrains/x-transformers 
- https://huggingface.co/docs/datasets/en/use_with_pytorch 
- https://ai.plainenglish.io/creating-and-exploring-gpt-from-scratch-ffe84ac415a9
 
# python-codecomplete-lm
A small LM for next token prediction in Python

## Models
Models I tried (you can find instructions about each in their own folders):

### First model: GPT2 from scratch
To start, I trained a small GPT2 model (~124M parameters) from scratch using the data we have. This Huggingface [tutorial](https://huggingface.co/learn/nlp-course/chapter7/6) goes into how to do this. This is cumbersome to do in some ways - training a tokenizer (even the fast tokenizers from HF written in Rust) takes a few hours since tokenizer training is CPU-bound. 

So I used the Python [tokenizer](https://huggingface.co/huggingface-course/code-search-net-tokenizer) HF had trained and referenced in the above tutorial. This took a few hours off our total training time. Then I followed the standard training procedure outlined in the tutorial with different sizes of data, trying different dataset sizes to see what's realistic to train on a free Google Clab and get decent results.

### Second model: DeepSeek Coder 1.3B Parameter-efficient fine-tune
The [evalplus](https://evalplus.github.io/leaderboard.html) leaderboard has a list of the best current LLMs for code. For our size requirements (close to ~1B), the DeepSeek Coder 1.3B base model was ideal. I followed the steps given [here](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html) to fine-tune models in a Parameter-efficient way using methods like LoRA (Low Rank Adapters that are aligned with and trained instead of the model weights, which are frozen) and QLoRA (LoRA that works on Quantized LLMs, like those loaded in 4bits with bitsandbytes). 

### Third model: Decoder-only Transformer from scratch in PyTorch
I wrote this model in PyTorch using a standard decoder-only Transformer architecture for generation. 

## Eval

## Outputs

## References
- https://evalplus.github.io/leaderboard.html
- https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base
- https://huggingface.co/learn/nlp-course/chapter7/6
- https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html
- https://unsloth.ai/blog/llama3
- https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing
- https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/training.ipynb
- https://huggingface.co/docs/transformers/training
- https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse
- https://medium.com/@akriti.upadhyay/building-custom-gpt-with-pytorch-59e5ba8102d4 
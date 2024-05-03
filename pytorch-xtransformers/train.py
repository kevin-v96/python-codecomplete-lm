import torch 
from model import TransformerDecoder

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define the hyperparameters
vocab_size     = 10000
d_model        = 2048
num_heads      = 2
ff_hidden_layer  = 8*d_model
dropout        = 0.1
num_layers     = 20
context_length = 1000
batch_size     = 1
# Initialize the model
model = TransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_layer, dropout)

#train here
print(f"The model has {count_parameters(model):,} trainable parameters")
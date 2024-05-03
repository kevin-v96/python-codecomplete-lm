import torch

#get model
# model = load_pytorch_model 

# Create a tensor representing batch size and context length
input_tensor = torch.randint(0, vocab_size, (context_length, batch_size))

# Forward pass through the model
output = model(input_tensor)


print(output.shape)  

# To get the predicted word indices, we can use the `argmax` function
predicted_indices = output.argmax(dim=-1)

print(predicted_indices.shape)
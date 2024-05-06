import tiktoken
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import notebook_login
import wandb
from model import AutoregressiveWrapper, LanguageModel
import matplotlib.pyplot as plt
from config import get_config

wandb.login()
notebook_login()

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


enc = tiktoken.encoding_for_model("gpt-4")
eos_token_id = enc.eot_token
eos_token = enc.decode([eos_token_id])
config = get_config(eos_token_id, eos_token)

class CodeSnippetDataset(Dataset):
    def __init__(self, snippets, tokenizer):
        self.snippets = snippets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, idx):
        snippet = self.snippets[idx]

        # Tokenize the snippet
        input_ids = self.tokenizer.encode(snippet, allowed_special = "all")
        attention_mask = [1] * len(input_ids)  # Assume all tokens are attended to

        return input_ids, attention_mask

def collate_fn(batch, max_length):
    input_ids_list, attention_mask_list = [], []

    # Process each snippet in the batch
    for input_ids, attention_mask in batch:
        # Truncate input_ids and attention_mask based on the maximum sequence length in the batch
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]

        # Pad input_ids and attention_mask to max_length
        padding_length = max_length - len(input_ids)
        padded_input_ids = input_ids + [config.padding_token_id] * padding_length
        padded_attention_mask = attention_mask + [0] * padding_length

        input_ids_list.append(padded_input_ids)
        attention_mask_list.append(padded_attention_mask)

    # Convert lists to tensors
    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_mask_list, dtype=torch.long)

    return input_ids_tensor, attention_mask_tensor


def prep_dataloaders():
    small_data = load_dataset("ArtifactAI/arxiv_python_research_code", split=f"train[:{config.total_samples}]")
    train_data_pt = small_data.with_format("torch", device=get_device())
    split_data = train_data_pt.train_test_split(test_size=config.valid_size)
    code_snippets = split_data["train"]["code"]
    val_code_snippets = split_data["test"]["code"]


    # Create the dataset
    train_dataset = CodeSnippetDataset(code_snippets, enc)
    val_dataset = CodeSnippetDataset(val_code_snippets, enc)

    # Create a DataLoader with collate_fn to handle padding
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, config.max_length + 1))
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, config.max_length + 1))

    return train_dataloader, val_dataloader

from tqdm import tqdm
class Trainer:
    def __init__(self, model, tokenizer, optimizer=None):
        super().__init__()
        self.model = model
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        else:
            self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.loss_function = torch.nn.CrossEntropyLoss()

    def train(self, train_dataloader, val_dataloader, epochs):
        loss_per_batch = []
        for epoch in range(epochs):
            self.model.train()

            for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                input_ids, attention_mask = batch

                # Create the input and mask tensors
                input_ids, attention_mask = input_ids.to(get_device()), attention_mask.to(get_device())

                # Forward pass
                #input_ids and attention_mask dimensions are (batch_size, sequence_length) (8,512)
                outputs, targets = self.model(x=input_ids, mask=attention_mask)

                # Reshape targets to match the format expected by CrossEntropyLoss
                targets_flat = targets.reshape(-1)  # Flatten to shape [batch_size * sequence_length]

                # Flatten logits to shape [batch_size * sequence_length, num_classes]
                outputs_flat = outputs.reshape(-1, outputs.shape[2])

                # Compute loss
                loss = self.loss_function(outputs_flat, targets_flat)

                # Backpropagate the loss.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Clip the gradients. This is used to prevent exploding gradients.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)

                # Log batch-level loss
                wandb.log({'train/batch_loss': loss.item(), 'epoch': epoch + 1, 'batch': batch_idx + 1})

                # Append batch loss to list
                loss_per_batch.append(loss.item())

            # Evaluate the model on validation data
            val_loss = self.evaluate(val_dataloader)
            wandb.log({'val/loss': val_loss, 'epoch': epoch + 1})

        return loss_per_batch

    def evaluate(self, dataloader):
        self.model.eval()
        val_loss = 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = batch
                input_ids, attention_mask = input_ids.to(get_device()), attention_mask.to(get_device())

                # Forward pass
                outputs, targets = self.model(x=input_ids, mask=attention_mask)
                targets_flat = targets.reshape(-1)
                outputs_flat = outputs.reshape(-1, outputs.shape[2])

                # Compute loss
                loss = self.loss_function(outputs_flat, targets_flat)
                val_loss += loss.item()

        # Calculate average validation loss
        val_loss /= num_batches
        return val_loss
    

if __name__ == "__main__":
    wandb.init(project = "transformer_from_scratch", job_type = "train", config = config, tags = ["training","trial_run"])
    # Create the tokenizer
    tokenizer = enc
    number_of_tokens = tokenizer.n_vocab

    # Create the model
    model = AutoregressiveWrapper(LanguageModel(
        embedding_dimension=config.embedding_dimension,
        number_of_tokens=number_of_tokens,
        number_of_heads=config.number_of_heads,
        number_of_layers=config.number_of_layers,
        dropout_rate=config.dropout_rate,
        max_sequence_length=config.max_length
    )).to(get_device())

    train_dataloader, val_dataloader = prep_dataloaders()

    # Train the model
    trainer = Trainer(model, tokenizer)
    loss_per_epoch = trainer.train(train_dataloader, val_dataloader, epochs=config.epochs)

    # Plot the loss per epoch in log scale
    plt.plot(loss_per_epoch)
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    wandb.run.finish()
    model.save_checkpoint('./trained_model')
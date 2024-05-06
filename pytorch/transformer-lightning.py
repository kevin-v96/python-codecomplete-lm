import tiktoken
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from model import AutoregressiveWrapper, LanguageModel
from config import get_config
import lightning as L
from typing import Any
from torch.utils.data import random_split
import os
import pickle
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LitGPT(L.LightningModule):
    def __init__(self, autoregressive_model, config, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.example_input_array = torch.Tensor(32, 1, 28, 28)  # need to change this
        self.autoregressive_model = autoregressive_model
        self.loss_function = torch.nn.CrossEntropyLoss()
        for (
            key,
            value,
        ) in (
            config.__dict__.items()
        ):  # this assigns all of the config in this format: self.lr = config.lr
            setattr(self, key, value)
        self.save_hyperparameters()

    def evaluate_batch_loss(self, batch):
        input_ids, attention_mask = batch
        outputs, targets = self.model(x=input_ids, mask=attention_mask)
        # Reshape targets to match the format expected by CrossEntropyLoss
        targets_flat = targets.reshape(-1)  # Flatten to shape [batch_size * sequence_length]
        # Flatten logits to shape [batch_size * sequence_length, num_classes]
        outputs_flat = outputs.reshape(-1, outputs.shape[2])
        # Compute loss
        loss = self.loss_function(outputs_flat, targets_flat)
        return loss

    def training_step(self, batch, batch_idx):
        train_loss = self.evaluate_batch_loss(batch)
        self.log("train/loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.evaluate_batch_loss(batch)
        self.log("val/loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x, *args: Any, **kwargs: Any) -> Any:
        x = x.view(x.size(0), -1)
        z = self.autoregressive_model(x)
        return z


class CodeSnippetDataset(Dataset):
    def __init__(self, snippets, tokenizer):
        self.snippets = snippets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, idx):
        snippet = self.snippets[idx]

        # Tokenize the snippet
        input_ids = self.tokenizer.encode(snippet, allowed_special="all")
        attention_mask = [1] * len(input_ids)  # Assume all tokens are attended to

        return input_ids, attention_mask


class PyCodeDataModule(L.LightningDataModule):
    def __init__(self, tokenizer, config, data_dir: str = os.getcwd()):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.vocab = tokenizer.n_vocab
        for (
            key,
            value,
        ) in (
            config.__dict__.items()
        ):  # this assigns all of the config in this format: self.lr = config.lr
            setattr(self, key, value)

        self.save_hyperparameters()

    def prepare_data(self) -> None:
        """
        Prepares the data by downloading and tokenizing it. Runs once, on Rank 0.
        """
        small_data = load_dataset(
            "ArtifactAI/arxiv_python_research_code",
            split=f"train[:{self.total_samples}]",
        )
        train_data_pt = small_data.with_format("torch", device=get_device())
        code_data = train_data_pt["code"]
        tokenized_dataset = CodeSnippetDataset(code_data, self.tokenizer)
        # lighting recommends you save to disk and load in the setup function to be compatible with distributed training
        with open("saved_dataset.pkl", "wb") as f:
            pickle.dump(tokenized_dataset, f)

    def collate_fn(self, batch, max_length):
        """Collate function for dataloader"""
        input_ids_list, attention_mask_list = [], []

        # Process each snippet in the batch
        for input_ids, attention_mask in batch:
            # Truncate input_ids and attention_mask based on the maximum sequence length in the batch
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]

            # Pad input_ids and attention_mask to max_length
            padding_length = max_length - len(input_ids)
            padded_input_ids = input_ids + [self.config.padding_toke_id] * padding_length
            padded_attention_mask = attention_mask + [0] * padding_length

            input_ids_list.append(padded_input_ids)
            attention_mask_list.append(padded_attention_mask)

        # Convert lists to tensors
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask_list, dtype=torch.long)

        return input_ids_tensor, attention_mask_tensor

    def setup(self, stage: str) -> None:
        # load the dataset from disk
        # dataset = load_dataset_from_disk(...)
        # Assign train/val datasets for use in dataloaders
        with open("saved_dataset.pkl", "rb") as f:
            tokenized_dataset = pickle.load(f)
        valid_size = self.valid_size
        if stage == "fit":
            self.train_dataset, self.val_dataset = random_split(
                tokenized_dataset,
                [1 - valid_size, valid_size],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test":
            pass

        if stage == "predict":
            pass

    def train_dataloader(self) -> Any:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn
        )  # the batch size needs to be self.batch_size for tuner to work

    def val_dataloader(self) -> Any:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn
        )


if __name__ == "__main__":
    enc = tiktoken.encoding_for_model("gpt-4")
    eos_token_id = enc.eot_token
    eos_token = enc.decode([eos_token_id])
    config = get_config(eos_token_id, eos_token)
    litgpt = LitGPT(
        model=AutoregressiveWrapper(
            LanguageModel(
                embedding_dimension=config.embedding_dimension,
                number_of_tokens=enc.n_vocab,
                number_of_heads=config.number_of_heads,
                number_of_layers=config.number_of_layers,
                dropout_rate=config.dropout_rate,
                max_sequence_length=config.max_length,
            )
        ),
        config=config,
    )

    dm = PyCodeDataModule(tokenizer=enc, config=config)
    wandb_logger = WandbLogger(project="transformer-lightning")
    trainer = L.Trainer(
        profiler="simple",
        #logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="16-mixed",  # 32-true by default. Other options: bf16-mixed, 16-true, 64-true (high memory, more sensitive)
        gradient_clip_val=0.5,  # default 0,
        fast_dev_run=True, #turn this off when the model runs
        max_epochs=config.epochs,
    )
    tuner = Tuner(trainer)
    tuner.scale_batch_size(litgpt, mode = "binsearch")
    tuner.lr_find(litgpt)
    trainer.fit(model=litgpt, datamodule=dm)


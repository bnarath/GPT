import os
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from nano_gpt.data_prep import Data_Retriever
from nano_gpt.data_dict import ROOT

from typing import List, Tuple, Callable

logging.getLogger().setLevel(logging.INFO)
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

class Nano_GPT:
    """
    Retrieves data
    Encodes it

    """

    def __init__(
        self,
        lr: float = 1e-3,
        batch_size: int = 32,
        block_size: int = 8,
        num_epochs: int = 300,
        max_new_tokens: int = 100,
        eval_interval: int = 30,
        eval_iters: int = 20,
        embed_size: int = 32,
        head_size: int = 16,
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.block_size = block_size
        self.epochs = num_epochs
        self.max_new_tokens = max_new_tokens
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.embed_size = embed_size
        self.head_size = head_size

    def run(self):
        data_retriever = Data_Retriever()
        file_uri = os.path.join(ROOT, "input.txt")
        text = data_retriever.get_data(file_uri=file_uri)
        vocab, self.vocab_size = self._inspect_data(data=text, file_uri=file_uri)
        self.encoder, self.decoder = self.create_tokenizer_encoder_decoder(vocab)
        data = torch.tensor(self.encoder(text), dtype=torch.long)
        logging.info(
            f"Type of data {type(data)}, Shape is {data.shape}, Data Type is {data.dtype}"
        )
        train_data, val_data = self.train_val_split(data, train_size=0.9)
        logging.info(
            f"Data is split into train : size {len(train_data)}, val: size {len(val_data)}"
        )
        # logging.info("Inspecting a random training batch")
        # self._inspect_a_random_batch(train_data, batch_size=4, block_size=8)
        self.model = BigramLanguageModel(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            embed_size=self.embed_size,
            head_size=self.head_size,
        )
        self.model = self.model.to(device)
        # logging.info("Inspecting generation task")
        # self._inspect_generation_task(train_data)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.train(train_data, val_data)
        self._inspect_generation_task(val_data)

    def train(self, train_data: torch.Tensor, val_data: torch.Tensor):

        for step in range(self.epochs):
            # on regular intervals, calculate mean loss on train and val data
            if step % self.eval_interval == 0:
                losses = self.estimate_loss(train_data, val_data)
                logging.info(
                    f"STEP: {step}, TRAIN LOSS: {losses['train']:.4f}, VAL LOSS: {losses['val']:.4f}"
                )

            # sample a batch
            xb, yb = self.get_batch(train_data, self.batch_size, self.block_size)
            # loss evaluation
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            # if step % 1000:
            #     logging.info(f"STEP: {step} LOSS: {loss.item()}")
            #     self._inspect_generation_task(train_data)
        # logging.info(f"STEP: {step} LOSS: {loss.item()}")
        # self._inspect_generation_task(train_data)

    @torch.no_grad()
    def estimate_loss(self, train_data: torch.Tensor, val_data: torch.Tensor):
        mean_loss = {}
        self.model.eval()
        for d_t in ["train", "val"]:
            data = train_data if d_t == "train" else val_data
            losses = torch.zeros(self.eval_iters)
            for batch in range(self.eval_iters):
                xb, yb = self.get_batch(
                    data=data, batch_size=self.batch_size, block_size=self.block_size
                )
                _, loss = self.model(xb, yb)
                losses[batch] = loss.item()
            mean_loss[d_t] = losses.mean()
        self.model.train()
        return mean_loss

    def _get_file_size(self, bytes: int):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes < 1024:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0

    def _inspect_data(self, data: str, file_uri: str) -> Tuple[List[str], int]:
        bytes = os.path.getsize(file_uri)
        logging.info(f"File size: {self._get_file_size(bytes)}")
        logging.info(f"No of characters in the dataset : {len(data)}")
        # logging.info("\n")
        # logging.info("##############################################\n")
        # logging.info("First 100 chars")
        # logging.info(data[:100])
        # logging.info("##############################################\n")

        # get the vocabulary size
        chars = sorted(set(data))
        vocab_size = len(chars)
        logging.info(f"vocab_size = {vocab_size}")
        logging.info(f"vocab = {''.join(chars)}")
        return chars, vocab_size

    def create_tokenizer_encoder_decoder(self, vocab: List[str]) -> Tuple[Callable]:
        stoi = {s: i for i, s in enumerate(vocab)}
        itos = {i: s for i, s in enumerate(vocab)}
        encoder = lambda s: [stoi[c] for c in s]
        decoder = lambda l: "".join([itos[i] for i in l])
        return encoder, decoder

    def train_val_split(
        self, data: torch.Tensor, train_size: float = 0.9
    ) -> Tuple[torch.Tensor]:
        n = int(train_size * len(data))
        train_data = data[:n]
        val_data = data[n:]
        return train_data, val_data

    def get_batch(
        self, data: torch.Tensor, batch_size: int, block_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select random index (as many as batch size) from the text and takes a context window of block_size, produce context and target"""
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    def _inspect_a_random_batch(
        self,
        data: torch.Tensor,
        batch_size: int,
        block_size: int,
    ) -> None:
        xb, yb = self.get_batch(data, batch_size, block_size)
        logging.info("inputs:")
        logging.info(xb.shape)
        logging.info(xb)
        logging.info("targets:")
        logging.info(yb.shape)
        logging.info(yb)

        for b in range(batch_size):
            for t in range(block_size):
                context = xb[b, : t + 1]
                target = yb[b, t]
                logging.info(f"context {context.tolist()}, target {target.tolist()}")

    def _inspect_generation_task(self, data: torch.Tensor) -> None:
        xb, yb = self.get_batch(data, self.batch_size, self.block_size)
        logging.info(f"Passing to a BigramLanguageModel")
        out, loss = self.model(xb, yb)
        logging.info(out.shape)
        logging.info(loss)
        logging.info("Testing generation task")
        idx = torch.zeros((1, 1), dtype=torch.long)
        logging.info(
            self.decoder(
                self.model.generate(idx, max_new_tokens=self.max_new_tokens)[0].tolist()
            )
        )


class AttentionHead(nn.Module):
    """Single head of self attention"""

    def __init__(self, block_size: int = 8, embed_size: int = 32, head_size: int = 16):
        super().__init__()
        self.block_size = block_size
        self.embed_size = embed_size
        self.head_size = head_size
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )  # part of model state but not trainable, no gradients

    def forward(self, x):
        B, T, C = x.shape  # C is embed_size
        k = self.key(x)  # B, T, head_size
        q = self.query(x)  # B, T, head_size
        # compute attention score weights
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5  # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # B, T, T
        wei = F.softmax(wei, dim=-1)  # B, T, T
        wei = self.dropout(wei)
        # Get weighted aggregation of values
        v = self.value(x)  # B, T, head_size
        out = wei @ v  # B, T, head_size
        return out


class BigramLanguageModel(nn.Module):

    def __init__(
        self, vocab_size: int, block_size: int, embed_size: int, head_size: int
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_size = embed_size
        self.head_size = head_size
        # Embedding of each token
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.sa_head = AttentionHead(block_size, embed_size, head_size=embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)
        self.pos_embedding_table = nn.Embedding(block_size, embed_size)

    def forward(self, idx, targets=None):
        # idx - (batch, block)
        # token_embeddings - (batch, block, embed_size)
        # logits - (batch, block, vocab_size)
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)  # (batch, block, embed_size)
        pos_embeddings = self.pos_embedding_table(
            torch.arange(T, device=device)
        )  # (block, vocab_size)
        x = (
            token_embeddings + pos_embeddings
        )  # (batch, block, embed_size) - broad casting
        x = self.sa_head(x)  # batch, block,head_size
        logits = self.lm_head(x)  # (batch, block, vocab_size)

        B, T, C = logits.shape
        _logits = logits.view(B * T, C)
        if targets is None:
            loss = None
        else:
            targets = targets.view(
                B * T,
            )
            loss = F.cross_entropy(
                _logits, targets
            )  # Expects the logits in the shape of (Batch, Channel)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx is (B, T) -> produces (B, T+max_new_tokens)
        for i in range(max_new_tokens):
            # crop idx to the last block_size of tokens
            idx_cropped = idx[:, -self.block_size :]
            # get logits
            logits, _ = self(idx_cropped)  # self.forward
            # take only the last time stamp to append
            logits = logits[:, -1, :]  # (B, X, C) to (B, C)
            # apply softmax
            probs = F.softmax(logits, dim=-1)
            idx_next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append the new with existing
            idx = torch.cat((idx, idx_next_token), dim=1)  # (B, T+1)
        return idx

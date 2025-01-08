import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyper parameters
epoch = 5000
embd_dim = 50
head_num = 1
block_size = 30
batch_size = 30
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'


data_path = '/Users/qianhangjian/AI/gpt_scratch/shakespeare/data/input.txt'
with open(data_path, 'r') as f:
    raw_text = f.read()

chars = sorted(list(set(raw_text)))
vocab_size = len(chars)

itos = {i:ch for i, ch in enumerate(chars)}
stoi = {ch:i for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s] # convert string to list integers
decode = lambda l: [itos[i] for i in l] # convert integers to string

data = torch.tensor(encode(raw_text))
n = int(len(data)*0.9)
train_data = data[:n]
validation_data = data[n:]

def get_batch(mode):
    batch_database = train_data if mode == 'train' else validation_data
    idx = torch.randint(len(batch_database)-block_size, (batch_size,))
    x = torch.stack([batch_database[i:i+block_size] for i in idx])
    y = torch.stack([batch_database[i+1:i+block_size+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = embd_dim // head_num
        self.query_matrix = nn.Linear(embd_dim, head_size, bias=False)
        self.key_matrix = nn.Linear(embd_dim, head_size, bias=False)
        self.value_matrix = nn.Linear(embd_dim, head_size, bias=False)
        self.tril = torch.tril(torch.ones(block_size, block_size))
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T, C = x.shape

        q = self.query_matrix(x) # batch_size * block_size * head_size
        k = self.key_matrix(x) # batch_size * block_size * head_size
        v = self.value_matrix(x) # batch_size * block_size * head_size

        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # batch_size * block_size * head_size
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        out = weights @ v # batch_size * block_size * head_size
        return out



class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # embeddding look up
        self.token_embedding_table = nn.Embedding(vocab_size, embd_dim) 
        self.position_embedding_table = nn.Embedding(block_size, embd_dim)
        self.sa = Head()
        self.ln_f = nn.LayerNorm(embd_dim)
        self.lm_head = nn.Linear(embd_dim, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tokens_embd = self.token_embedding_table(idx) # batch_size * block_size * embd_size
        pos_embd = self.position_embedding_table(torch.arange(T)) # block_size * embd_size
        x = tokens_embd + pos_embd # may broad cast shape here(if input length is too small)
        x = self.sa(x) # batch_size * block_size * embd_size
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            logits = logits.view(batch_size*block_size, vocab_size)
            targets = targets.view(batch_size*block_size)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_tokens=500):
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:] # careful here, the input idx could be shorter than block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # only take the last element, or should we take the len(idx_cond) for shorter length input?
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx



if __name__ == '__main__':
    mode = 'train'
    if mode == 'train':
        model = GPTLanguageModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        for i in range(epoch):
            x, y = get_batch(mode)
            logits, loss = model(x, y)
            print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(''.join(decode(model.generate(context, 300)[0].tolist())))

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
context_length = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 #384/6 = 64-dimensional head
n_head = 6
n_layer = 6 #6 layers of the above
dropout = 0.2 #prevents overfitting

torch.manual_seed(42)

with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)
#create a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #take a string and return a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #take a list of characters and return a string
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd,n_embd), #projection layer for residual output
            nn.Dropout(dropout), #right before the residual connection back into the pathway
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = nn.MultiheadAttention(head_size, n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        #communication and computation, along with residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #each token reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings_table = nn.Embedding(context_length, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        #idx and targets are both (B,T) tensors of integers
        token_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embeddings_table(torch.arange(T, device = device)) #(T,C)
        x = token_emb + pos_emb
        x = self.blocks(x) #(B,T,C)
        x = self.ln_f(x)
        logits = self.lm_head(x) #(B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self,idx,max_new_tokens):
        #idx is a (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last context_length tokens
            idx_cond = idx[:, -context_length:]
            #get the predictions
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:, -1, :] #becomes (B,C)
            probs = F.softmax(logits, dim = -1) #(B,C)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #append to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
    
model = LanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
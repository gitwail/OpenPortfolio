import torch 
import torch.nn as nn 
from torch.nn import functional as F
import torch.optim as optim 
import matplotlib.pyplot as plt

# parameters
block_size=8
batch_size=4
num_epoch=10000
n_embd=384
dropout=0.2
num_head=6
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# seed
torch.manual_seed(1337)


# loading shakespear text 
with open('data/input.txt','r',encoding='utf-8') as f:
    text=f.read()


# here all the unique characters in the text
chars=sorted(set(list(text)))
vocab_size=len(chars)


# mapping integers to char and vice versa
stoi={s:i for i,s in enumerate(chars)}
itos={i:s for i,s in enumerate(chars)}


# encode and decode: text <---> integer
encode= lambda s:[stoi[c] for c in s]
decode=lambda i:''.join([itos[b] for b in i])


# storing data in a pytorch tensor
data=torch.tensor(encode(text),dtype=torch.long)

# split data into Train/validation sets 90%/10%
n=int(len(data)*0.9)
train_data=data[:n]
val_data=data[n:]

# getting batch of data (x,y) from splits

def get_batch(split):
    data=  train_data if split=='train' else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x,y


xb,yb=get_batch('train')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
# defining the transformer model
# (1) code a head self-attention
# (2) code a multihead
# (3) code a block

class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        self.head_size=head_size
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self,x):
        q=self.query(x)# (B,T,H)
        k=self.key(x)# (B,T,H)
        v=self.value(x)# (B,T,H)
        B,T,H=q.shape
        wei=q @ k.transpose(-2,-1)*(n_embd**-0.5)# (B,T,H) @ (B,H,T) --> (B,T,T)
        # masking future tokens
        wei=wei.masked_fill(self.tril[:T, :T],float('-inf'))
        wei=F.softmax(wei,dim=-1)
        out=wei @ v
        return out




class MultiHeadAttention(nn.Module):

    def __init__(self,num_head,head_size):
        super().__init__()

        self.heads=nn.ModuleList([Head(head_size) for n in range(num_head)])
        self.proj=nn.Linear(num_head*head_size,n_embd)
        self.droptout=nn.Dropout(dropout)

    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.droptout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,n_embd*4),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout)

        )

    def forward(self,x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd,num_head) -> None:
        super().__init__()
        head_size=n_embd//num_head
        self.sa=MultiHeadAttention(num_head,head_size)
        self.ffwd=FeedForward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)

    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ffwd(self.ln2(x))

        return x



class GPTLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_table=nn.Embedding(vocab_size,n_embd) 
        self.positional_embd=nn.Embedding(block_size,n_embd) 
        self.blocks=nn.Sequential(*[ Block(n_embd,num_head)])
        self.ln=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)



    def forward(self,x,target=None):
        B,T=x.shape
        emb=self.embedding_table(x) # (B,T,C)
        pos=self.positional_embd(torch.arange(T)) # (T,C)
        x=emb+pos # (B,T,C)
        x=self.blocks(x)
        x=self.ln(x)
        logits=self.lm_head(x)
        B, T, C = logits.shape


        if target is None:
            loss=None

        else:
            logits=logits.view(B*T,C)
            target=target.view(B*T)
            loss=F.cross_entropy(logits,target)

        return logits,loss
    

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


        

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))



            


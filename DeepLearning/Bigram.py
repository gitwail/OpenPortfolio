import torch 
import torch.nn as nn 
from torch.nn import functional as F
import torch.optim as optim 

# parameters
block_size=8
batch_size=4
num_epoch=10000
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



class BiagramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)
        
    def forward(self,idx, targets=None):

        logits=self.token_embedding_table(idx)
        if targets is None:
            loss=None
        else:
            # logits shape is (B,T,C) and target shape is (B,T)
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        
        return logits,loss
    
    def generate(self,idx,max_new_tokens=100):
        # idx is (B,T)
        for _ in range(max_new_tokens):
            logits,loss=self(idx) #logits is (B,T,C)
            logits=logits[:,-1,:] # (B,C) only last as context for now
            probs=F.softmax(logits,dim=-1) # (B,C)
            idx_next=torch.multinomial(probs,num_samples=1) #(B,1)
            idx=torch.cat((idx,idx_next),dim=1)
            
        return idx
# model testing 
model=BiagramLanguageModel(vocab_size)


# Generate tokens before training the model
print(decode(model.generate(torch.zeros((1,1),dtype=torch.long))[0].tolist()))

# training loop
optimizer=optim.AdamW(model.parameters(),lr=1e-3)


for i in range(num_epoch):
    xb,yb=get_batch('train')
    # forward pass
    logits,loss=model(xb,yb)

    #backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%1000==0:
        print("training loss:",loss)


# Generate tokens after training the model
print(decode(model.generate(torch.zeros((1,1),dtype=torch.long))[0].tolist()))

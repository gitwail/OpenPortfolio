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

# defining the transformer model
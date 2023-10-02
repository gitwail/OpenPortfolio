import torch 
import numpy as np

seed = 42
torch.manual_seed(seed)

class LSTMcell():
    def __init__(self,x_dim,hidden_dim,batch_size) -> None:
        # hidden layer
        self.ht_1=torch.rand((batch_size,hidden_dim))
        concat_dim=hidden_dim+x_dim

        
        # forget gate 
        self.Wf=torch.rand((hidden_dim, concat_dim), requires_grad=True)
        self.bf=torch.rand((hidden_dim,batch_size),requires_grad=True)

        # input gate layer
        self.Wi=torch.rand((hidden_dim, concat_dim), requires_grad=True)
        self.bi=torch.rand((hidden_dim,batch_size),requires_grad=True)

        # candidate layer
        self.Wc=torch.rand((hidden_dim, concat_dim), requires_grad=True)
        self.bc=torch.rand((hidden_dim,batch_size),requires_grad=True)

        #output layer
        self.Wo=torch.rand((hidden_dim, concat_dim), requires_grad=True)
        self.bo=torch.rand((hidden_dim,batch_size),requires_grad=True)

        # cell state
        self.ct_1=torch.rand((hidden_dim,batch_size))

    def forward(self,xt):
        Xt=torch.concat([self.ht_1,xt],axis=1)
        # forget gate
        ft=torch.sigmoid((self.Wf@Xt.T)+self.bf)
        # input gate 
        it=torch.sigmoid((self.Wi@Xt.T)+self.bi)
        # candidate state
        c_tild=torch.tanh((self.Wc@Xt.T)+self.bc)
        # state calculation ct
        ct=(ft*self.ct_1)+(it*c_tild)
        # output gate of ct
        ot=torch.sigmoid((self.Wo@Xt.T)+self.bo)
        # new hidden layer
        ht=ot*torch.tanh(ct)

        return ht.T,ct


xt=torch.rand((20,1,3))

cell=LSTMcell(x_dim=3,hidden_dim=4,batch_size=1)
for x in xt:
    ht,ct=cell.forward(x)
    print(ht)
    cell.ht_1=ht
    cell.ct_1=ct
   


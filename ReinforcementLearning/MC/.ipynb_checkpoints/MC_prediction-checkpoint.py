import sys
sys.path.append("../lib/myenv")
from gridworld import gridworld
import numpy as np 
import random
import matplotlib.pyplot as plt 


# sample from categorical distribution
def sample_categorical(probabilities):
    return random.choices(range(len(probabilities)), probabilities)[0]

# gridworld dimension
dim=4
# gamma discounting factor 
gamma=0.5

# delta for policy evaluation
delta=0

#env variable 
gw=gridworld(dim)


# The agent is placed at (0,0) and value function are initiliazed to a zero array
gw.reset()

################ policy evaluation ########################################

def policy(dim):
    """
    Initial policy with 1/4 chance for each action
    Input: dimension of gridworld
    output: random uniform policy
    """
    pi={}
    for i in range(dim):
        for j in range(dim):
            pi[(i,j)]=[0.25]*4

    return pi



# State value (V) is an array of dimension nxn where n is the gridworld size
V=np.random.rand(dim,dim)
# V=np.zeros((dim,dim))


#Transition matrix
Pss=np.ones((dim,dim))


# Initial policy
pi=policy(dim)


# The action mapping for human readibility
# 0:right
# 1:Left
# 2:UP
# 3:Down 

####################### Policy evaluation using montecarlo first visit #########################



def episode(pi,dim):
    # reset environement
    gw.reset()
    # Init observation
    o=(0,0,dim-1,dim-1)
    # track state termination
    terminated=False
    # list of rewards
    rewards=[]
    # State/action/reward list
    sar=[]
    #Initial reward
    r=-1
    # episode length
    l=0
    while True :
        l+=1
        a=sample_categorical(pi[o[:2]])
        print("Action: ",a)
        sar.append((o[:2],a,r))
        if terminated:
            break
        o,r,terminated,_,_,=gw.step(a)
        rewards.append(r)

  
    print("total rewards: ", np.sum(rewards))
    # print("state action reward: ",sar)
    print("length of episode: ",l)
    return sar

# montecarlo first visit algorithm

# Gain initialization
G=0

# state/gain/visit
dict_visit={(i,j):[] for i in range(dim) for j in range(dim)}
print(dict_visit)


num_episodes=5

for _ in range(num_episodes):
    #episode 
    sar=episode(pi,dim)
    # reverse the list sar list
    reversed_sar=list(reversed(sar))

    for i,e in enumerate(reversed_sar):
        s,a,r=e
        G=G+(gamma*r)
        Exist=sum([sar[0]==s for sar in reversed_sar[i+1:]])
        if Exist==0 :
            dict_visit[s].append(G)
                                


print("list visit: ",dict_visit)
for k,v in dict_visit.items():
    if len(v)!=0:
        V[k]=np.mean(v)
    else:
        V[k]=0


print(V)





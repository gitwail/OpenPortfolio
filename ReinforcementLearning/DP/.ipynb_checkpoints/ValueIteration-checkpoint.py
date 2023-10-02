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
dim=10
# gamma discounting factor 
gamma=1

# delta for policy evaluation
delta=0

#env variable 
gw=gridworld(dim)

# The action mapping for human readibility
# 0:right
# 1:Left
# 2:UP
# 3:Down 

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



# policy evaluation: Vk+1(s)=  sigma_a [pi(a|s)] sigma_s [r+ Pss' Vk(s)]
# The possible transition  Pss'=1 otherwise Pss'=0
# Vk+1(s)=  sigma_a [pi(a|s)] sigma_s [r+ Vk(s)]

    
def policy_evaluation(pi,V,iteration_num=1):
    delta=0
    #looping over all states
    for _ in range(iteration_num):
        Vc=V.copy()
        for i in range(dim):
            for j in range(dim):
                gw.reset()
                listupdate=[]
                gw.agent=(i,j)

                if (i,j)!=(0,0):
                   for a in range(4):
                        o,r,_,_,_,=gw.step(a)
                        k,l=o[:2]
                        listupdate.append(pi[(i,j)][a]*(r+(gamma*Vc[k,l])))
                    
                    #update Vk+1
                   Vc[i,j]=np.max(listupdate)
                
                #calculate delta
                # print("state: ",(i,j))
                # print("V",V[i,j])
                # print("Vc",Vc[i,j])
                # print("Vc-V",np.abs(Vc[i,j]-V[i,j]))
                # delta=max(delta,np.abs(V[i,j]-Vc[i,j]))
                # print("delta ",delta)

        
        V=Vc
        # if delta <0.1:
        #     break

    return V


def policy_improvement(Vpi):
# policy iteration 
    for i in range(dim):
                for j in range(dim):
                    listofall=[-100]*4
                    r=-1
                    if (i,j)==gw.target:
                        r=0
                    
                    # listofall=[Vpi[i+1,j],Vpi[i-1,j],Vpi[i,j-1],Vpi[i,j+1]]
                    if i+1<dim:
                        listofall[0]=pi[(i,j)][0]*(r+(gamma*Vpi[i+1,j]))
                    if j+1<dim:
                        listofall[3]=pi[(i,j)][3]*(r+(gamma*Vpi[i,j+1])) 
                    if i-1>dim:
                        listofall[1]=pi[(i,j)][1]*(r+(gamma*Vpi[i-1,j]))
                    if j-1>dim:
                        listofall[2]=pi[(i,j)][1]*(r+(gamma*Vpi[i,j-1]))


                    a=np.argmax(listofall)

                    pi[(i,j)]=[1 if i == a else 0 for i in range(4)]

    return pi




Vpi=policy_evaluation(pi,V,iteration_num=1)
pi=policy_improvement(Vpi)



print(np.array(Vpi))

print(pi)

# The loop below will test the policy iteration algorithm
# start position
gw.reset()
o=(0,0,dim-1,dim-1)
terminated=False
rewards=[]
while not terminated:

    a=sample_categorical(pi[o[:2]])
    print(a)
    o,r,terminated,_,_,=gw.step(a)
    rewards.append(r)
    gw.render(np.round(Vpi, 1),mode='human')
    print(o[:2])
    print("reward is: ",r)

print("total rewards: ", np.sum(rewards))


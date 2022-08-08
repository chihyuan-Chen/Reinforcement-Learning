"""
@author: Ivan
"""

import numpy as np
from matplotlib import pyplot as plt
import math

class Epsilon_GreedyBandit():
  def __init__(self,mu,mean):
    self.mu=mu
    self.mean=mean
    self.times=0

  def pull(self):
    return np.random.randn()+self.mu
    
  def update(self,xn): 
    self.times+=1 
    self.mean=(self.mean*(self.times-1)+xn)/self.times 

  def run(self,mu1,mu2,mu3,epsilon,N):
    bandits=[Epsilon_GreedyBandit(mu1,self.mean),Epsilon_GreedyBandit(mu2,self.mean),Epsilon_GreedyBandit(mu3,self.mean)]
    data=[]
    #run simulation for N times
    for i in range(N):
      #1. choose bandit
      p=np.random.random()
      if p<epsilon:
        j=np.random.choice(3)
      else:#take advantage of the biggest mean now
        j=np.argmax([b.mean for b in bandits])
      #2. pull it
      x=bandits[j].pull()
      bandits[j].update(x)
      data.append(x)
    cumul_average=np.cumsum(data)/(np.arange(N)+1)
    return cumul_average


class GreedyBandit():
  def __init__(self,mu,mean):
    self.mu=mu
    self.mean=mean
    self.times=0

  def pull(self):
    return np.random.randn()+self.mu
    
  def update(self,xn):
    self.times+=1 
    self.mean=(self.mean*(self.times-1)+xn)/self.times 

  def run(self,mu1,mu2,mu3,N):
    bandits=[GreedyBandit(mu1,self.mean),GreedyBandit(mu2,self.mean),GreedyBandit(mu3,self.mean)]
    data=[]
    #run simulation for N times
    for i in range(N):
      #1. choose bandit
      j=np.argmax([b.mean for b in bandits])
      #2. pull it
      x=bandits[j].pull()
      bandits[j].update(x)
      data.append(x)
    cumul_average=np.cumsum(data)/(np.arange(N)+1)
    return cumul_average


class Optimistic_Initial_Value():
  def __init__(self,mu,mean):
    self.mu=mu
    self.mean=mean
    self.times=1

  def pull(self):
    return np.random.randn()+self.mu
    
  def update(self,xn):
    self.times+=1 
    self.mean=(1-1.0/self.times)*self.mean+1.0/self.times*xn

  def run(self,mu1,mu2,mu3,N):
    bandits=[Optimistic_Initial_Value(mu1,self.mean),Optimistic_Initial_Value(mu2,self.mean),Optimistic_Initial_Value(mu3,self.mean)]
    data=[]
    #run simulation for N times
    for i in range(N):
      #1. choose bandit
      j=np.argmax([b.mean for b in bandits])
      #2. pull it
      x=bandits[j].pull()
      bandits[j].update(x)
      data.append(x)
    cumul_average=np.cumsum(data)/(np.arange(N)+1)
    return cumul_average


class SoftmaxBandit():
  def __init__(self,mu,mean):
    self.mu=mu
    self.mean=mean
    self.times=0

  def pull(self):
    return np.random.randn()+self.mu
    
  def update(self,xn):
    self.times+=1 
    self.mean=(self.mean*(self.times-1)+xn)/self.times 

  def run(self,mu1,mu2,mu3,N):
    bandits=[SoftmaxBandit(mu1,self.mean),SoftmaxBandit(mu2,self.mean),SoftmaxBandit(mu3,self.mean)]
    data=[]
    #run simulation for N times
    for i in range(N):
      #1. choose bandit
      exp=([math.exp(b.mean) for b in bandits]) 
      sum_exp=sum(exp)
      softmax=np.argmax([(i/sum_exp) for i in exp])
      #2. pull it
      x=bandits[softmax].pull()
      bandits[softmax].update(x)
      data.append(x)
    cumul_average=np.cumsum(data)/(np.arange(N)+1)
    return cumul_average


class UpperConfidenceBound1():
  def __init__(self,mu,mean):
    self.mu=mu
    self.mean=mean
    self.times=1

  def pull(self):
    return np.random.randn()+self.mu
    
  def update(self,xn):
    self.times+=1 
    self.mean=(self.mean*(self.times-1)+xn)/self.times 

  def run(self,mu1,mu2,mu3,N):
    bandits=[UpperConfidenceBound1(mu1,self.mean),UpperConfidenceBound1(mu2,self.mean),UpperConfidenceBound1(mu3,self.mean)]
    data=[]
    #run simulation for N times
    for i in range(N):
      #1. choose bandit
      j=np.argmax(self.mean+(np.sqrt(2*np.log(i+1)/self.times)))
      #2. pull it
      x=bandits[j].pull()
      bandits[j].update(x)
      data.append(x)
    cumul_average=np.cumsum(data)/(np.arange(N)+1)
    return cumul_average


class ThompsonSampling():
  def __init__(self,mu,mean):
    self.mu=mu
    self.mean=mean
    self.times=1

  def pull(self):
    return np.random.randn()+self.mu
    
  def update(self,xn):
    self.times+=1 
    self.mean=(self.mean*(self.times-1)+xn)/self.times 

  def run(self,mu1,mu2,mu3,N):
    bandits=[ThompsonSampling(mu1,self.mean),ThompsonSampling(mu2,self.mean),ThompsonSampling(mu3,self.mean)]
    data=[]
    #run simulation for N times
    for i in range(N):
      #1. choose bandit
      j=np.argmax(np.random.beta(1+self.mean,1-self.mean)) 
      #2. pull it
      x=bandits[j].pull()
      bandits[j].update(x)
      data.append(x)
    cumul_average=np.cumsum(data)/(np.arange(N)+1)
    return cumul_average


if __name__ == '__main__':
    epochs=100000
    Greedy=GreedyBandit(0,0.5)
    Epsilon_Greedy=Epsilon_GreedyBandit(0,0.5) 
    Softmax=SoftmaxBandit(0,0.5)
    OIV=Optimistic_Initial_Value(0,0.5)
    UCB1=UpperConfidenceBound1(0,0.5)
    Thompson_Sampling=ThompsonSampling(0,0.5)
    R1=Greedy.run(1.6,1.0,0.6,N=epochs)
    R2=Epsilon_Greedy.run(1.6,1.0,0.6,epsilon=0.1,N=epochs)
    R3=Softmax.run(1.6,1.0,0.6,N=epochs)
    R4=OIV.run(1.6,1.0,0.6,N=epochs)
    R5=UCB1.run(1.6,1.0,0.6,N=epochs)
    R6=Thompson_Sampling.run(1.6,1.0,0.6,N=epochs)
    #plt.title("Multi-Armed Bandit",fontsize=15)
    plt.plot(R1,label="Greedy",color='r')
    plt.plot(R2,label="Epsilon_Greedy",color='y')
    plt.plot(R3,label="Softmax",color='g')
    plt.plot(R4,label="OIV",color='b')
    plt.plot(R5,label="UCB1",color='m')
    plt.plot(R6,label="Thompson_Sampling",color='c')
    plt.title("Multi-Armed Bandit",fontsize=15)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Rate of Choosing Best Arm')
    plt.legend()
    plt.xscale('log')
"""
@author: Ivan
"""

import numpy as np
from matplotlib import pyplot as plt

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

#============ main ===============
epochs=100000
Bandit=GreedyBandit(0,0.5)
results=Bandit.run(1.6,1.0,0.6,N=epochs)
plt.plot(results)
plt.xscale('log')
print(results[-1])
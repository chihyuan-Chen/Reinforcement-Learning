"""
@author: Ivan
"""

import numpy as np
from matplotlib import pyplot as plt

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

#============ main ===============
epochs=100000
Bandit=Optimistic_Initial_Value(0,0.5)
results=Bandit.run(1.6,1.0,0.6,N=epochs)
plt.plot(results)
plt.xscale('log')
print(results[-1])
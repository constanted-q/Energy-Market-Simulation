import numpy as np
import matplotlib.pyplot as plt

def moving_average(a, n=3) :
 ret = np.cumsum(a, dtype=float,axis = 0)
 ret[n:] = ret[n:] - ret[:-n]
 return ret[n - 1:] / n


with open('1.npy', 'rb') as f:
 reward_list_n2c7 = np.load(f)
 reward_list_n2c7_avg = moving_average(reward_list_n2c7,20)
with open('1.npy', 'rb') as f:
 reward_list_n5c7 = np.load(f)
 reward_list_n5c7_avg = moving_average(reward_list_n5c7,20)


plt.figure(dpi = 400)
plt.title('2 clients, capacity = 7')
plt.plot(range(0,len(reward_list_n2c7[:,0])), reward_list_n2c7[:,0], color='orangered',linewidth = 1.5,alpha = 0.2)
plt.plot(range(0,len(reward_list_n2c7[:,1])), reward_list_n2c7[:,1], color='darkorange',linewidth = 1.5,alpha = 0.2)
plt.plot(range(0,len(reward_list_n2c7_avg[:,0])), reward_list_n2c7_avg[:,0], color='orangered',label = 'Client 1, G=4',linewidth = 1.5)
plt.plot(range(0,len(reward_list_n2c7_avg[:,1])), reward_list_n2c7_avg[:,1], color='darkorange',label = 'Client 2, G=5',linewidth = 1.5)
plt.ylabel('Average Reward')
plt.xlabel('Episodes')
plt.ylim(0, 1)
plt.legend()
plt.show()

plt.figure(dpi = 400)
plt.title('5 clients, capacity = 7')
plt.plot(range(0,len(reward_list_n5c7[:,0])), reward_list_n5c7[:,0], color='orangered',alpha = 0.2)
plt.plot(range(0,len(reward_list_n5c7[:,1])), reward_list_n5c7[:,1], color='darkorange',alpha = 0.2)
plt.plot(range(0,len(reward_list_n5c7[:,2])), reward_list_n5c7[:,2], color='gold',alpha = 0.2)
plt.plot(range(0,len(reward_list_n5c7[:,3])), reward_list_n5c7[:,3], color='lightseagreen',alpha = 0.2)
plt.plot(range(0,len(reward_list_n5c7[:,4])), reward_list_n5c7[:,4], color='deepskyblue',alpha = 0.2)
plt.plot(range(0,len(reward_list_n5c7_avg[:,0])), reward_list_n5c7_avg[:,0], color='orangered',label = 'Client 1, G=4')
plt.plot(range(0,len(reward_list_n5c7_avg[:,1])), reward_list_n5c7_avg[:,1], color='darkorange',label = 'Client 2, G=5',)
plt.plot(range(0,len(reward_list_n5c7_avg[:,2])), reward_list_n5c7_avg[:,2], color='gold',label = 'Client 3, G=6')
plt.plot(range(0,len(reward_list_n5c7_avg[:,3])), reward_list_n5c7_avg[:,3], color='lightseagreen',label = 'Client 4, G=7')
plt.plot(range(0,len(reward_list_n5c7_avg[:,4])), reward_list_n5c7_avg[:,4], color='deepskyblue',label = 'Client 5, G=8')
plt.ylabel('Average Reward')
plt.xlabel('Episodes')
plt.ylim(0, 0.6)
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import json

def MA(vec,sample):
    cumsum_vec = np.cumsum(np.insert(vec, 0, 0))
    ma_vec = (cumsum_vec[sample:] - cumsum_vec[:-sample]) / sample
    return ma_vec

with open("logs\Info2.json") as f:
    data = json.load(f)

cost = np.array(data[0]["cost"])
alpha = np.array(data[0]["alpha"])
beta = np.array(data[0]["beta"])


def get_points(a,b,c):
    points = []
    for i,e in enumerate(c):
        points.append([ a[int(np.floor(i/len(a)))], b[i%len(b)], e])
    return points
fig = plt.figure()
ax = plt.axes(projection='3d')

print(cost.shape,alpha.shape,beta.shape)
for step in range(323,len(cost)):
    points = np.array(get_points(alpha,beta,cost[step]))

    img = ax.plot_trisurf(points[:,0],points[:,1],points[:,2],cmap=cm.jet, vmin= 0, vmax = 1)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    ax.set_zlabel(r'$cost$')
    # ax.set_zlim3d(0, 1)
    ax.azim = step
    ax.elev = 38
    plt.savefig(f"Images/input{step}.png")
    ax.cla()
    print(step)

# for i in range(0,360,1):
#     ax.azim = i
#     print(i)

# step = []
# pred = []
# true = []
# acc = []
# cost = []
#
# for epoch in range(0,2):
#     cost.append(np.array(data[epoch]["cost"]))
#     step.append(8000*epoch+np.array(data[epoch]["step"]))
#     acc.append(np.array(data[epoch]["acc"]))
#     pred.append(np.array(data[epoch]["pred"]))
#     true.append(np.array(data[epoch]["true"]))
#
# cost = np.array([*cost],dtype="object").flatten()
# step = np.array([*step],dtype="object").flatten()
# acc = np.array([*acc],dtype="object").flatten()
# ma_cost = MA(cost,500)
#
# plt.plot(np.arange(len(cost)),cost,label="cost")
# plt.plot(np.arange(len(ma_cost)),ma_cost,label="MA-cost")
# plt.plot(step,acc,label="accuracy")
# plt.plot(step[0::997],acc[0::997],label="accuracy-peak")
#
#
# plt.show()

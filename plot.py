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

cost = np.array(data["cost"])
alpha = np.array(data["alpha"])
beta = np.array(data["beta"])


def get_points(a,b,c):
    p = []
    for i,e in enumerate(a):
        for j,q in enumerate(b[i]):
            p.append([e,q,c[i,j]])
    return np.array(p)

fig = plt.figure()
ax = plt.axes(projection='3d')

print(cost.shape,alpha.shape,beta.shape)
points = get_points(alpha,beta,cost)
print(points,points.shape)

ax.plot_trisurf(points[:,0], points[:,1], points[:,2], cmap=cm.jet, linewidth=0.1);
for i in range(0,360,1):
    ax.azim = i
    print(i)
    plt.savefig(f"Images/input{i}.png")

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

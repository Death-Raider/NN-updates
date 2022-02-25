import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import json

def MA(vec,sample):
    cumsum_vec = np.cumsum(np.insert(vec, 0, 0))
    ma_vec = (cumsum_vec[sample:] - cumsum_vec[:-sample]) / sample
    return ma_vec

with open("logs\Info.json") as f:
    data = json.load(f)

# cost = data["cost"]
#
# ma_cost = MA(cost,100)
# plt.plot(np.arange(len(ma_cost)),ma_cost)
# plt.show()

step = []
pred = []
true = []
acc = []
cost = []

for epoch in range(0,1):
    cost.append(np.array(data[epoch]["cost"]))
    step.append(8000*epoch+np.array(data[epoch]["step"]))
    acc.append(np.array(data[epoch]["acc"]))
    pred.append(np.array(data[epoch]["pred"]))
    true.append(np.array(data[epoch]["true"]))

    cost = np.array([*cost],dtype="object").flatten()
    step = np.array([*step],dtype="object").flatten()
    acc = np.array([*acc],dtype="object").flatten()
    ma_cost = MA(cost,500)

    plt.plot(np.arange(len(cost)),cost,label="cost")
    plt.plot(np.arange(len(ma_cost)),ma_cost,label="MA-cost")

    plt.plot(step,acc,label="accuracy")
    plt.plot(step[0::997],acc[0::997],label="accuracy-peak")

    plt.savefig("test")
    plt.show()

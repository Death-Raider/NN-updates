import matplotlib.pyplot as plt
import json
import numpy as np
with open("logs\Info.json") as f:
    data = json.load(f)
    data = np.array(data,dtype="object")
cost = []
step = []
pred = []
true = []
acc = []
for epoch in range(0,1):
    cost.append(np.array(data[epoch]["cost"]))
    step.append(8000*epoch+np.array(data[epoch]["step"]))
    acc.append(np.array(data[epoch]["acc"]))
    pred.append(np.array(data[epoch]["pred"]))
    true.append(np.array(data[epoch]["true"]))

cost = np.array([*cost],dtype="object").flatten()
step = np.array([*step],dtype="object").flatten()
acc = np.array([*acc],dtype="object").flatten()
cumsum_vec = np.cumsum(np.insert(cost, 0, 0))
ma_vec = (cumsum_vec[100:] - cumsum_vec[:-100]) / 100

# plt.plot(step,cost,label="cost")
plt.plot(step[:len(ma_vec)],ma_vec,label="MA-cost")
plt.plot(step,acc,label="accuracy")
plt.plot(step[0::997],acc[0::997],label="accuracy-peak")
# plt.scatter(pred[0][:len(cost)],cost[:])
plt.legend()
plt.savefig("test")
plt.show()

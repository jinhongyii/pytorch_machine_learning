import os

import torch
from numba import jit

@jit
def loadDataSet(fileName):
    dataMat1 = []
    labelMat = []
    os.system("type " + fileName + " > result.txt")
    fr = open("result.txt")
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        dataMat1.append(list(map(int, curLine[:-1])))
        labelMat.append([int(curLine[-1])])
    return dataMat1, labelMat

xArr, yArr = loadDataSet("cm.txt")
x = torch.Tensor(xArr)
y = torch.Tensor(yArr)

net=torch.nn.Sequential(
    torch.nn.Linear(13,32),
    torch.nn.ReLU(),
    torch.nn.Linear(32,32),
    torch.nn.ReLU(),
    torch.nn.Linear(32,1)

)
print(net)  # net architecture
optimizer = torch.optim.Adam(net.parameters())
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

for epoch in range(30000):
    prediction = net(x)  # input x and predict based on x
    loss = loss_func(prediction, y)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

torch.save(net.state_dict(), 'net_param.pkl')

testdata,y=loadDataSet("cmtest.txt")
testdata=torch.Tensor(testdata)
y=torch.Tensor(y)
print(loss_func(net(testdata),y))

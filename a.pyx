import os
import torch.utils.data as Data
import torch
import torch.nn.functional as F
from torch.autograd import Variable

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

xArr, yArr = loadDataSet("cp.txt")
tensor = torch.FloatTensor(xArr)
x = Variable(tensor,requires_grad=True)
tensor = torch.FloatTensor(yArr)
y = Variable(tensor,requires_grad=True)


#torch_dataset = Data.TensorDataset(x, y)
#loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,)
net=torch.nn.Sequential(
    torch.nn.Linear(13,32),
    torch.nn.ReLU(),
    torch.nn.Linear(32,32),
    torch.nn.Sigmoid(),
    torch.nn.Linear(32,16),
    torch.nn.ReLU(),
    torch.nn.Linear(16,1)

)
print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=5e-4,betas=(0.9,0.99))
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

for epoch in range(20000):

    #print(epoch)
    prediction = net(x)  # input x and predict based on x
    loss = loss_func(prediction, y)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
    #print(loss)
torch.save(net.state_dict(), 'net_param.pkl')

testdata,y=loadDataSet("test.txt")
testdata=torch.FloatTensor(testdata)
print(net(testdata))
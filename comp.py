import os

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

# torch.manual_seed(1)    # reproducible

def loadDataSet(fileName):
    dataMat1 = []

    labelMat = []
    # fr = open(fileName)
    os.system("type " + fileName + " > result.txt")
    fr = open("result.txt")
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        dataMat1.append(list(map(float, curLine[:-1])))
        labelMat.append([float(curLine[-1])])
    return dataMat1, labelMat


xArr, yArr = loadDataSet("cp.txt")
tensor = torch.FloatTensor(xArr)
x = Variable(tensor)  # x data (tensor), shape=(100, 1)
tensor = torch.FloatTensor(yArr)
y = Variable(tensor)  # noisy y data (tensor), shape=(100, 1)


#hyper parameters
BATCH_SIZE=32
EPOCH=12

torch_dataset=Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


net = Net(n_feature=13, n_hidden=50, n_output=1)  # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss



for t in range(100000):
    prediction = net(x)  # input x and predict based on x

    loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

print(loss)
torch.save(net.state_dict(),'cp_net_param.pkl')

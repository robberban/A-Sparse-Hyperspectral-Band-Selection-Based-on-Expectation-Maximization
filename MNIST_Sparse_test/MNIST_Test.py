import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from m_spl import sp_cost
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

class BS_Layer_with_r(nn.Module):#No Attention, three losses
    def __init__(self, band):
        super(BS_Layer_with_r, self).__init__()
        self.weights = nn.Parameter(torch.zeros(1,band)+0.5)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()

    def forward(self, X):
        self.x_save = Variable(X,requires_grad=False)
        weights =  self.relu_2(1 - self.relu_1(1 - self.weights))
        out = weights * X
        return out, weights

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc2_1 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, 10)
        self.weights = nn.Parameter(torch.rand(1,128))
        self.BS_Layer = BS_Layer_with_r(128)


    def forward(self, x):
        x = x.view(-1, 784)
        x = nn.functional.relu(self.fc1(x))
        x = x, weights = self.BS_Layer(nn.functional.relu(self.fc2(x)))
        x = nn.functional.relu(self.fc2_1(x))
        x = self.fc3(x)
        return x, weights

def Loss_EM(prob, select_band=20):
    band_number = prob.shape[1]
    prob = prob.unsqueeze(2)
    prob = torch.cat((1-prob,prob),dim=2).permute(1,0,2).log()
    token = Variable(torch.ones(select_band)).cuda()
    sizes = Variable(torch.IntTensor(np.array([band_number]))).cuda()
    target_sizes = Variable(torch.IntTensor(np.array([select_band]))).cuda()
    cost = sp_cost(prob, token, sizes, target_sizes)
    return cost

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


model = Net().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()


for epoch in range(5):  
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader), 0):
        inputs, labels = data
        inputs, labels = Variable(inputs).float().cuda(), Variable(labels).long().cuda()
        optimizer.zero_grad()
        outputs, weights = model(inputs)
        loss = criterion(outputs, labels) + 1.0 * Loss_EM(weights)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # if i % 100 == 99: 
    print(weights.sum())   
    print('[%d] loss: %.3f' % (epoch+1, running_loss))
    running_loss = 0.0
print(weights.sum())
print(weights)

print('Finished Training')


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = Variable(images).float().cuda(), Variable(labels).long().cuda()
        outputs, weights = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
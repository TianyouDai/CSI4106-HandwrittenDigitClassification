import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import time


#setting parameters    
BATCH_SIZE = 64      
LR = 0.001        


#loading database and converting to variables
transform = transforms.ToTensor()
trainset = tv.datasets.MNIST(
    root='./data/',
    train=True,
    download=True,
    transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

testset = tv.datasets.MNIST(
    root='./data/',
    train=False,
    download=True,
    transform=transform)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=200,
    shuffle=False,
    )

#building model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            # Conv layer 1 output shape (6,28,28)
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            # Pooling layer 1 (max pooling) output shape (6,14,14)
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        self.conv2 = nn.Sequential(
            # Conv layer 2 output shape (16,14,14)
            nn.Conv2d(6, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Pooling layer 2 (max pooling) output shape (16,7,7)
            nn.MaxPool2d(2, 2)     
        )
        self.fc1 = nn.Sequential(
            # Fully connected layer 1 input shape (16*7*7)=(784)
            nn.Linear(16 * 7 * 7, 120),
            nn.ReLU()
        )

        # Fully connected layer 2 to shape (120) for 10 classes
        self.fc2 = nn.Linear(120, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1) 
        x = self.fc1(x)
        x = self.fc2(x)
        return x



#Compile model
net = CNN()
criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

# train

def train(EPOCH):
    
    for epoch in range(EPOCH):
        sum_loss = 0.0
        print("--------------epoch "+str(epoch+1)+"--------------")
        time_start=time.time()
        for i, data in enumerate(trainloader):
            inputs, labels = data        
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # every 100 batch, print loss
            sum_loss += loss.item()
            if i % 100 == 99:
                print('batch: %d loss: %.03f'
                      % (i + 1, sum_loss / 100))
                sum_loss = 0.0
        time_end=time.time()
        print('time cost: ',time_end-time_start,'s')
        # after each epoach,test accuracy
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('epoach: %d Accuracy：%d%%' % (epoch + 1, (100 * correct / total)))

# test demo

def test():
    correct = 0
    total = 0
    i=0
    count = random.randint(1,10)
    images, labels = next(iter(testloader))
    for data in testloader:
        i+=1
        if(i==count):
            images, labels = data
            break
    print("Test dataset label is: ")
    print(labels)
    output = net(images)
    _,predicted = torch.max(output,1)
    print("Test result is:")
    print(predicted)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    print('Accuracy: {}/{} ({:.0f}%)'.format(correct,total,(100 * correct / total)))
    X = tv.utils.make_grid(images)
    X = X.numpy().transpose(1,2,0)
    plt.imshow(X)
    plt.show()


    
    


    

import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes
dtype=torch.cuda.FloatTensor

class Net(nn.Module):


    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 200, kernel_size=7, padding=2)
        self.batch1 = nn.BatchNorm2d(200)
        self.conv2 = nn.Conv2d(200,250, kernel_size=4, padding=2)
        self.batch2 = nn.BatchNorm2d(250)
        self.conv3 = nn.Conv2d(250,350, kernel_size=4, padding=2)
        self.batch3 = nn.BatchNorm2d(350)
        self.fc1 = nn.Linear(6*6*350, 400)
        self.fc2 = nn.Linear(400, nclasses)

       # Spatial transformer localization-network
        self.localization1 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(3, 250, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(250, 250, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.localization2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(200, 150, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(150, 200, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.localization3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(250, 150, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(150, 200, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc1 = nn.Sequential(
            nn.Linear(250*6*6, 250),
            nn.ReLU(True),
            nn.Linear(250, 3 * 2)
        )

        self.fc_loc2 = nn.Sequential(
            nn.Linear(200*2*2, 300),
            nn.ReLU(True),
            nn.Linear(300, 3 * 2)
        )

        self.fc_loc3 = nn.Sequential(
            nn.Linear(200*1*1, 300),
            nn.ReLU(True),
            nn.Linear(300, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc1[2].weight.data.zero_()
        self.fc_loc1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0]).type(dtype))

        self.fc_loc2[2].weight.data.zero_()
        self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0]).type(dtype))

        self.fc_loc3[2].weight.data.zero_()
        self.fc_loc3[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0]).type(dtype))

     # Spatial transformer network forward function
    def stn(self, x, a, b, c):
        if c==1:
            xs = self.localization1(x)
            xs = xs.view(-1, 250*a*b)
            theta = self.fc_loc1(xs)
            theta = theta.view(-1, 2, 3)
        if c==2:
            xs = self.localization2(x)
            xs = xs.view(-1, 200*a*b)
            theta = self.fc_loc2(xs)
            theta = theta.view(-1, 2, 3)
        if c==3:
            xs = self.localization3(x)
            xs = xs.view(-1, 200*a*b)
            theta = self.fc_loc3(xs)
            theta = theta.view(-1, 2, 3)
    
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x=x.type(dtype)
        x = self.stn(x,6,6,1)
        x = F.leaky_relu(F.max_pool2d(self.conv1(x), 2))
        x = self.batch1(x)
        x = self.stn(x,2,2,2)
        x = F.leaky_relu(F.max_pool2d(self.conv2(x), 2))
        x = self.batch2(x)
        x = self.stn(x,1,1,3)
        x = F.leaky_relu(F.max_pool2d(self.conv3(x), 2))
        #x = self.batch3(x)
        x = x.view(-1, 6*6*350)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
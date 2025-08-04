import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        #layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #16

        #layer 2
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #8

        #layer 3
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.batchnorm5 = nn.BatchNorm2d(128)
        # self.relu = nn.ReLU()
        # self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.batchnorm6 = nn.BatchNorm2d(128)
        # self.relu = nn.ReLU()
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2) #4

        #FC1
        self.fc1 = nn.Linear(in_features=4096, out_features=512)
        self.batchnorm_fc1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(p=0.5)

        # self.fc2 = nn.Linear(in_features=1024, out_features=256)
        # self.batchnorm_fc2 = nn.BatchNorm1d(256)
        # self.relu = nn.ReLU()
        # self.dropout_fc2 = nn.Dropout(p=0.5)

        # self.fc3 = nn.Linear(in_features=256, out_features=64)
        # self.batchnorm_fc3 = nn.BatchNorm1d(64)
        # self.relu = nn.ReLU()
        # self.dropout_fc3 = nn.Dropout(p=0.5)

        self.fc4 = nn.Linear(in_features=512, out_features=2)


    def forward(self, x):
        #layer 1
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool1(out)

        #layer 2
        out = self.conv3(out)
        out = self.batchnorm3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.batchnorm4(out)
        out = self.relu(out)
        out = self.maxpool2(out)

        #layer 3
        # out = self.conv4(out)
        # out = self.batchnorm4(out)
        # out = self.relu(out)
        # out = self.conv5(out)
        # out = self.batchnorm5(out)
        # out = self.relu(out)
        # out = self.maxpool3(out)
        
        #Flatten()
        out = out.view(-1, 4096)

        #FC 1
        out = self.fc1(out)
        out = self.batchnorm_fc1(out)
        out = self.relu(out)
        out = self.dropout_fc1(out)

        #FC 2
        # out = self.fc2(out)
        # out = self.batchnorm_fc2(out)
        # out = self.relu(out)
        # out = self.dropout_fc2(out)

        # #FC 3
        # out = self.fc3(out)
        # out = self.batchnorm_fc3(out)
        # out = self.relu(out)
        # out = self.dropout_fc3(out)

        #Out
        out = self.fc4(out)

        return out

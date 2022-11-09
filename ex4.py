import torch
import sys
import numpy as np
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class ModelA(nn.Module):
    """Neural Network with two hidden layers, the first layer should
    have a size of 100 and the second layer should have a size of 50, both
    should be followed by ReLU activation function. SGD optimizer"""

    def __init__(self, image_size=784, first_size=100, second_size=50, output_size=10):
        super(ModelA, self).__init__()
        self.hidden1 = nn.Linear(image_size, first_size)
        self.hidden2 = nn.Linear(first_size, second_size)
        self.hidden3 = nn.Linear(second_size, output_size)
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return F.log_softmax(self.hidden3(x), dim=1)


class ModelB(nn.Module):
    """Neural Network with two hidden layers, the first layer should
    have a size of 100 and the second layer should have a size of 50, both
    should be followed by ReLU activation function. Adam optimizer"""

    def __init__(self, image_size=784, first_size=100, second_size=50, output_size=10):
        super(ModelB, self).__init__()
        self.hidden1 = nn.Linear(image_size, first_size)
        self.hidden2 = nn.Linear(first_size, second_size)
        self.hidden3 = nn.Linear(second_size, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return F.log_softmax(self.hidden3(x), dim=1)


class ModelC(nn.Module):
    """ModelB with dropout on the output of the hidden layer"""

    def __init__(self, dropout, image_size=784, first_size=100, second_size=50, output_size=10):
        super(ModelC, self).__init__()
        self.hidden1 = nn.Linear(image_size, first_size)
        self.drop1 = nn.Dropout(p=dropout)
        self.hidden2 = nn.Linear(first_size, second_size)
        self.drop2 = nn.Dropout(p=dropout)
        self.hidden3 = nn.Linear(second_size, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.hidden1(x))
        x = self.drop1(x)
        x = F.relu(self.hidden2(x))
        x = self.drop2(x)
        return F.log_softmax(self.hidden3(x), dim=1)


class ModelD(nn.Module):
    """add Batch Normalization layers to ModelB, after or before ? the activation function?"""

    def __init__(self, image_size=784, first_size=100, second_size=50, output_size=10):
        super(ModelD, self).__init__()
        self.hidden1 = nn.Linear(image_size, first_size)
        self.b1 = nn.BatchNorm1d(num_features=first_size)
        self.hidden2 = nn.Linear(first_size, second_size)
        self.b2 = nn.BatchNorm1d(num_features=second_size)
        self.hidden3 = nn.Linear(second_size, output_size)
        self.b3 = nn.BatchNorm1d(num_features=output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    # before
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.b1(self.hidden1(x)))
        x = F.relu(self.b2(self.hidden2(x)))
        return F.log_softmax(self.b3(self.hidden3(x)), dim=1)

    # after
    """def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.hidden1(x))
        x = self.b1(x)
        x = F.relu(self.hidden2(x))
        x = self.b2(x)
        x = F.log_softmax(self.hidden3(x), dim=1)
        x = self.b3(x)
        return x"""


class ModelE(nn.Module):
    """Neural Network with five hidden layers:[128,64,10,10,10] using ReLU"""

    def __init__(self):
        super(ModelE, self).__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.hidden3 = nn.Linear(64, 10)
        self.hidden4 = nn.Linear(10, 10)
        self.hidden5 = nn.Linear(10, 10)
        self.hidden6 = nn.Linear(10, 10)
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        return F.log_softmax(self.hidden6(x), dim=1)


class ModelF(nn.Module):
    """Neural Network with five hidden layers:[128,64,10,10,10] using Sigmoid."""

    def __init__(self):
        super(ModelF, self).__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.hidden3 = nn.Linear(64, 10)
        self.hidden4 = nn.Linear(10, 10)
        self.hidden5 = nn.Linear(10, 10)
        self.hidden6 = nn.Linear(10, 10)
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.sigmoid(self.hidden1(x))
        x = F.sigmoid(self.hidden2(x))
        x = F.sigmoid(self.hidden3(x))
        x = F.sigmoid(self.hidden4(x))
        x = F.sigmoid(self.hidden5(x))
        return F.log_softmax(self.hidden6(x), dim=1)


def shuffle(x, y):
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    return x[index], y[index]


def init1():
    x_train_sys, y_train_sys = sys.argv[1], sys.argv[2]
    X_train = np.loadtxt(x_train_sys) / 255
    Y_train = np.loadtxt(y_train_sys)
    # z-score
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train - mean) / std
    X_train, Y_train = shuffle(X_train, Y_train)
    div80 = int(0.8 * len(X_train))
    train_set, val_set = X_train[:div80, :], X_train[div80:, :]
    label_train, val_label = Y_train[:div80], Y_train[div80:]

    train_set = torch.from_numpy(train_set).float()
    label_train = torch.from_numpy(label_train).long()
    training = TensorDataset(train_set, label_train)
    train_loader = DataLoader(dataset=training, batch_size=64, shuffle=True)

    val_set = torch.from_numpy(val_set).float()
    val_label = torch.from_numpy(val_label).long()
    validation = TensorDataset(val_set, val_label)
    validation_loader = DataLoader(dataset=validation, batch_size=1, shuffle=False)

    return train_loader, validation_loader


def train(model, train_l):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, label) in enumerate(train_l):
        model.optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, label)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
        # get the index of the max log probability
        pred = output.max(1, keepdim=True)[1]
        # Sum all the correct classifications
        correct += pred.eq(label.view_as(pred)).sum().item()
    size = len(train_l.dataset)
    total_loss /= (size / 64)
    accuracy = (correct / size) * 100
    #return total_loss, accuracy


def validation_test(model, val_l):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in val_l:
            output = model(data)
            total_loss += F.nll_loss(output, label, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
        size = len(val_l.dataset)
        total_loss /= size
        accuracy = (correct / size) * 100
        #return total_loss, accuracy


"""def find_drop():
    train_l, val_l = init()
    pgood = 0
    acc = 0
    acc_test = 0
    for p in np.arange(0.1, 0.9, 0.1):
        model = ModelC(p)
        for epoch in range(1, 10 + 1):
            loss_train, acc_train = train(model, train_l)
            loss_test, acc_test = validation_test(model, val_l)
        if acc < acc_test:
            acc = acc_test
            pgood = p
    return pgood"""

def init2():
    x_train_sys, y_train_sys, x_test_sys = sys.argv[1], sys.argv[2], sys.argv[3]

    X_train = np.loadtxt(x_train_sys) / 255
    # z-score
    mean_train = np.mean(X_train)
    std_train = np.std(X_train)
    X_train = (X_train - mean_train) / std_train
    X_train = torch.from_numpy(X_train).float()

    Y_train = np.loadtxt(y_train_sys)
    Y_train = torch.from_numpy(Y_train).long()

    train_set = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)

    X_test = np.loadtxt(x_test_sys) / 255
    mean_test = np.mean(X_test)
    std_test = np.std(X_test)
    X_test = (X_test - mean_test) / std_test
    X_test = torch.from_numpy(X_test).float()
    test_set = TensorDataset(X_test)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    #test_y = open(test_y_sys, "w")

    return train_loader, test_loader




def main():
    """transforms_test = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((mean,), (std,))])
    test_loader = DataLoader(torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transforms_test), batch_size=64, shuffle=False)

    model = ModelD()
    Loss1 = []
    Accuracy1 = []
    Loss2 = []
    Accuracy2 = []
    Epoch = []
    for epoch in range(1, 10 + 1):
        loss_train, acc_train = train(model, train_l)
        Loss1.append(loss_train)
        Accuracy1.append(acc_train)
        loss_test, acc_test = validation_test(model, val_l)
        Loss2.append(loss_test)
        Accuracy2.append(acc_test)
        Epoch.append(epoch)

    loss_t, acc_t = validation_test(model, test_loader)
    print(acc_t)

    plt.plot(Epoch, Loss1, label="loss_train")
    plt.plot(Epoch, Loss2, label="loss_validation")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss ModelF')
    plt.legend()
    plt.show()"""

    prediction = []
    # train_l, test_l, y_test = init2()
    train_l, test_l = init2()

    model = ModelD()
    for epoch in range(1, 10 + 1):
        train(model, train_l)

    model.eval()
    with torch.no_grad():
        for x in test_l:
            output = model(x[0])
            pred = output.max(1, keepdim=True)[1]
            prediction.append(pred.item())

    with open("test_y", "w") as f:
        for yhat in prediction:
            f.write(str(yhat) + "\n")





if __name__ == '__main__':
    main()

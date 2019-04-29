import os, sys
import torch
import torch.nn as nn

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


class NNClassifier:

    def __init__(self, X, y, n_epochs=40, test_ratio=0.25):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=0)
        self.model = torch.nn.Sequential(torch.nn.Linear(self.X_train.shape[1], 100),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(100, 100),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(100, 100),
                                         torch.nn.Linear(100, 2))
        self.n_epochs = n_epochs

    # Training
    def train(epoch, net, trainloader):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print("Epoch: {} | Training Loss: {:0.3f} | Training Accuracy: {:0.3f}".format(epoch, train_loss, 100.*correct/total))

    def test(epoch, net, testloader):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                


    def run(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        trainset =
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=12)
        testset =
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=12)

        net = self.model
        net = net.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=e-4)

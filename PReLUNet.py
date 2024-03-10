import os
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models.model import PReLUNet, BasicBlock

# Setup environment variable and hyperparameters
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
batch_size = 32
num_epochs = 100

# Assuming num_classes is correct for your task
num_classes = 6

# Initialize storage for accuracy measurements
train_acc = np.zeros(num_epochs)

for k in range(27):  # Assuming 27 SNR levels for illustration
    snr = str(k)
    data_path = f'./data/case1withnoisevali/snr={snr}ttr=0.5.mat'
    data = sio.loadmat(data_path)

    train_data = torch.from_numpy(data['train_data']).type(torch.FloatTensor).unsqueeze(1)
    train_labels = torch.from_numpy(data['train_label'].squeeze()).type(torch.LongTensor) - 1

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model initialization
    net = PReLUNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)
    if torch.cuda.is_available():
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 250, 300], gamma=0.5)

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        correct = 0
        total = 0
        for samples, labels in train_loader:
            if torch.cuda.is_available():
                samples, labels = samples.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(samples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        # Calculate and store accuracy
        accuracy = 100 * correct / total
        train_acc[epoch] = accuracy
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/total}, Accuracy: {accuracy}%')

    # Save model and training accuracy after all epochs
    torch.save(net.state_dict(), f'./model_snr{snr}.pth')
    np.save(f'./train_acc_snr{snr}.npy', train_acc)

import os
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models.model import PReLUNet, BasicBlock

# Setup environment variable and hyperparameters
# The rest of the code remains the same...

# Setup environment variable and hyperparameters
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
batch_size = 32
num_epochs = 100

# Initialize storage for accuracy measurements and predictions
train_acc = np.zeros(num_epochs)
test_acc = np.zeros(num_epochs)
test_label_matrix = np.zeros((27, 1))  # Adjust dimensions according to your needs

for k in range(27):
    snr = str(k)
    # Load your data
    data_path = f'./data/case1withnoisevali/snr={snr}ttr=0.5.mat'
    data = sio.loadmat(data_path)

    # Process and load data as tensors
    train_data = torch.from_numpy(data['train_data']).type(torch.FloatTensor).unsqueeze(1)
    train_label = torch.from_numpy(data['train_label'].squeeze()).type(torch.LongTensor) - 1

    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize PReLUNet outside the loop
    net = PReLUNet(BasicBlock, [1, 1, 1, 1], num_classes=6)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 250, 300], gamma=0.5)

    # Training and evaluation loop
    # Training and evaluation loop
    for epoch in range(num_epochs):
        net.train()
        for samples, labels in train_loader:
            samples, labels = Variable(samples), Variable(labels)
            optimizer.zero_grad()
            outputs = net(samples)

            # Adjust the size of the output tensor to match the target size
            target_size = labels.size(0)  # Get the size of the target tensor
            outputs = outputs[0].repeat(target_size, 1)  # Repeat the output tensor to match the target size

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Add code for calculating and printing accuracy here...

        # Add code for calculating and printing accuracy here...

    # Save the results outside the epoch loop
    # Add code for saving the performance and predictions here...

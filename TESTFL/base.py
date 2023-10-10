import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torchvision import datasets, transforms
from get_data import *
from torch.utils.data import DataLoader



# Khởi tạo mô hình
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Reshape input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyModel()
data_dir = '../data/mnist/'
apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Chọn hàm mất mát và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

output = model(train_loader.data)

# Lặp qua dữ liệu huấn luyện và tính gradient
for batch_idx, (data, target) in enumerate(train_loader):
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print("----------------------------")

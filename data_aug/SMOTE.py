import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import pandas as pd
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm  # 引入进度条库
import os
import torch.nn.functional as F
from collections import defaultdict, Counter
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler


print("env is done")

# 定义数据集的保存路径
data_path = '../autodl-tmp/CIFAR10'  # 可以更改为你希望保存的目录

# 使用 torchvision 下载 CIFAR-10 数据集
train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())

batch_size = 256


test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=8)
new_indices = np.load("selected_indices.npy").tolist()
new_train_dataset = Subset(train_dataset, new_indices)
# new_train_loader = DataLoader(new_train_dataset,batch_size=batch_size,shuffle=True,num_workers=8)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = [
    "airplane",  # 0
    "automobile",  # 1
    "bird",  # 2
    "cat",  # 3
    "deer",  # 4
    "dog",  # 5
    "frog",  # 6
    "horse",  # 7
    "ship",  # 8
    "truck"  # 9
]

# 提取训练集的特征和标签，用于SMOTE过采样
def get_data_for_smote(dataset):
    data = []
    labels = []
    for img, label in dataset:
        data.append(img.numpy().flatten())  # 将图像展平为一维
        labels.append(label)
    return np.array(data), np.array(labels)

# 获取训练数据和标签
train_data, train_labels = get_data_for_smote(new_train_dataset)

# 应用SMOTE进行过采样
smote = SMOTE(sampling_strategy='auto', random_state=42)
train_data_resampled, train_labels_resampled = smote.fit_resample(train_data, train_labels)

# 将生成的合成样本转换为Tensor并重新包装成DataLoader
train_data_resampled_tensor = torch.tensor(train_data_resampled, dtype=torch.float32)
train_labels_resampled_tensor = torch.tensor(train_labels_resampled, dtype=torch.long)

# 将生成的合成样本包装成TensorDataset
from torch.utils.data import TensorDataset
resampled_train_dataset = TensorDataset(train_data_resampled_tensor.view(-1, 3, 32, 32), train_labels_resampled_tensor)
new_train_loader = DataLoader(resampled_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

def initial_model(class_names):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features,len(class_names))
    model = model.to(device)
    print("模型初始化完成")
    return model


model = initial_model(class_names)



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma = 0.1)

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 包装 train_loader
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Training")
    
    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累加损失和正确预测数量
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条后缀显示
        loop.set_postfix({
            "loss": running_loss / (batch_idx + 1),  # 平均损失
            "accuracy": 100. * correct / total      # 当前准确率
        })

    # 计算整个 epoch 的平均损失和准确率
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# 带进度条的 test 函数
def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 包装 test_loader
    loop = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 累加损失和正确预测数量
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条后缀显示
            loop.set_postfix({
                "loss": running_loss / (batch_idx + 1),  # 平均损失
                "accuracy": 100. * correct / total      # 当前准确率
            })

    # 计算整个测试集的平均损失和准确率
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def start_train(num_epochs, save_name, model, train_loader, test_loader, criterion, optimizer, scheduler):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    os.makedirs("log",exist_ok=True)
    os.makedirs("weight",exist_ok=True)
    
    csv_file = f'log/{save_name}.csv'
    
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_accuracy = test(model, test_loader, criterion)
        scheduler.step()

        # 记录每个 epoch 的结果
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # 打印每个 epoch 的训练和测试结果
        print(f'Epoch {epoch + 1}/{num_epochs} - '
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
            f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
        if train_accuracy > 99.999:
            break
    
    # 保存训练过程记录到 CSV 文件
    df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Train Accuracy': train_accuracies,
        'Test Loss': test_losses,
        'Test Accuracy': test_accuracies
    })
    df.to_csv(csv_file, index=False)
    torch.save(model.state_dict(), f"weight/{save_name}")
    print(f"训练过程已保存到 '{csv_file}' 文件中。模型权重文件已保存")
    
start_train(num_epochs=50, save_name="SMOTE", model=model, train_loader=new_train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
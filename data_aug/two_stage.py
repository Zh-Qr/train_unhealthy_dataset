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
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader


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
new_train_loader = DataLoader(new_train_dataset,batch_size=batch_size,shuffle=True,num_workers=8)



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

def get_class_count(class_names,dataset):
    class_count = np.zeros(len(class_names))
    for _,label in dataset:
        class_count[label] += 1
    for i in range(len(class_names)):
        print(f'{class_names[i]}：{class_count[i]}')

# get_class_count(class_names,train_dataset)

def initial_model(class_names):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features,len(class_names))
    model = model.to(device)
    print("模型初始化完成")
    return model


model = initial_model(class_names)


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

def train_stage(model, train_loader, test_loader, num_epochs, criterion, optimizer, scheduler, stage_name, log_data):
    print(f"{stage_name} 训练开始...")
    
    for epoch in range(num_epochs):
        # 使用 tqdm 包装 train_loader
        loop = tqdm(train_loader, desc=f"{stage_name} - Epoch {epoch + 1}/{num_epochs} - Training", leave=False)
        running_loss, correct, total = 0.0, 0, 0
        
        model.train()  # 设置为训练模式
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            loop.set_postfix({
                "loss": running_loss / (loop.n + 1),  # 平均损失
                "accuracy": 100. * correct / total   # 当前准确率
            })
        
        # 测试阶段
        loop = tqdm(test_loader, desc=f"{stage_name} - Epoch {epoch + 1}/{num_epochs} - Testing", leave=False)
        test_loss, test_correct, test_total = 0.0, 0, 0
        
        model.eval()  # 设置为评估模式
        with torch.no_grad():
            for inputs, targets in loop:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
                loop.set_postfix({
                    "loss": test_loss / (loop.n + 1),
                    "accuracy": 100. * test_correct / test_total
                })
        
        # 保存结果到 log_data
        log_data.append({
            "Stage": stage_name,
            "Epoch": epoch + 1,
            "Train Loss": running_loss / len(train_loader),
            "Train Accuracy": 100. * correct / total,
            "Test Loss": test_loss / len(test_loader),
            "Test Accuracy": 100. * test_correct / test_total
        })
        
        scheduler.step()
        
        print(f"{stage_name} - Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {100. * correct / total:.2f}%, "
              f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {100. * test_correct / test_total:.2f}%")
    
    print(f"{stage_name} 训练结束。")
    return model


def two_stage_training(model, train_loader, test_loader, num_epochs_stage_1, num_epochs_stage_2, class_names, device):
    # 初始化记录列表
    log_data = []

    # 第一阶段：标准训练
    stage_1_name = "Stage 1"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    model = train_stage(model, train_loader, test_loader, num_epochs_stage_1, criterion, optimizer, scheduler, stage_1_name, log_data)

    # 第二阶段：类别敏感训练
    stage_2_name = "Stage 2"
    # 计算类别权重
    class_count = np.zeros(len(class_names))
    for _, label in train_loader.dataset:
        class_count[label] += 1
    class_weights = 1.0 / class_count
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model = train_stage(model, train_loader, test_loader, num_epochs_stage_2, criterion, optimizer, scheduler, stage_2_name, log_data)

    # 保存结果到单一 CSV 文件
    df = pd.DataFrame(log_data)
    os.makedirs("log", exist_ok=True)
    df.to_csv("log/two_stage_training.csv", index=False)
    print("训练结果已保存到 'log/two_stage_training.csv' 文件中。")

    # 保存模型
    os.makedirs("weight", exist_ok=True)
    torch.save(model.state_dict(), "weight/two_stage_model.pth")
    print("模型训练完成并保存。")


# 开始两阶段训练
num_epochs_stage_1 = 30  # 第一阶段训练轮数
num_epochs_stage_2 = 20  # 第二阶段训练轮数

two_stage_training(
    model=model,
    train_loader=new_train_loader,
    test_loader=test_loader,
    num_epochs_stage_1=num_epochs_stage_1,
    num_epochs_stage_2=num_epochs_stage_2,
    class_names=class_names,
    device=device
)
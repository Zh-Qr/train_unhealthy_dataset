import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
import csv

# 带进度条的 test 函数
def test(model, test_loader, criterion,device):
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

def train_base(model, train_loader, criterion, optimizer, epoch,device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 包装 train_loader
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Training")

    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条后缀显示
        loop.set_postfix({
            "loss": running_loss / (batch_idx + 1),  # 平均损失
            "accuracy": 100. * correct / total      # 当前准确率
        })                                

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def train_ITLM(model, train_loader, criterion, optimizer, epoch, prune_ratio, device):
    prune_ratio = 0.1
    model.train()
    total_loss = 0.0
    running_loss = 0.0
    correct = 0
    total = 0
    all_losses = []
    all_data = []
    
    # 使用 tqdm 包装 train_loader
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Training")

    # 计算每个样本的损失
    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item()

        # 记录每个样本的损失值和数据
        all_losses.append(loss.detach().cpu().item())
        all_data.append((data, target, loss))
        
        # 更新进度条后缀显示
        loop.set_postfix({
            "loss": running_loss / (batch_idx + 1),  # 平均损失
            "accuracy": 100. * correct / total      # 当前准确率
        }) 
    
    # 排序并去掉损失较大的部分
    num_prune = int(prune_ratio * len(all_losses))
    sorted_indices = sorted(range(len(all_losses)), key=lambda i: all_losses[i], reverse=True)
    prune_indices = set(sorted_indices[:num_prune])

    # 保留损失较小的数据重新训练
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Training")
    for i, (data, target, loss) in loop:
        if i not in prune_indices:
            output = model(data)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        # 更新进度条后缀显示
        loop.set_postfix({
            "loss": running_loss / (batch_idx + 1),  # 平均损失
            "accuracy": 100. * correct / total      # 当前准确率
        }) 

    avg_loss = total_loss / (total - num_prune)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# inbalanced dataset
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Args:
            alpha (float): 平衡因子，用于控制正负样本的重要性。
            gamma (float): 调节因子，用于聚焦困难样本。
            reduction (str): 损失的聚合方式，'mean' 或 'sum' 或 'none'。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # 转换为概率
        pt = torch.exp(-ce_loss)
        # 计算 Focal Loss
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
def train_FL(model, train_loader, optimizer, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 包装 train_loader
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Training")

    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条后缀显示
        loop.set_postfix({
            "loss": running_loss / (batch_idx + 1),  # 平均损失
            "accuracy": 100. * correct / total      # 当前准确率
        })                                

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy
# coding=gbk
# -*- coding:uft-8 -*-
# train

import torch
import time
from tqdm import tqdm
from vcode_ddddocr_resnet.resnet18.codedataset import codeDataset
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
import torchvision.models as models

if __name__ == '__main__':
    train_dataset = codeDataset('../../dataset/train')
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=4)    # 为啥这里num_workers设置4没有报错
    '''
        模型里的num_classes设置了4*10
        每个验证码图片有4个字符(标签)，并且顺序固定；只要将卷积神经网络的最后一层稍加修改就能实现多标签分类
        验证码一共有4个数字，将4个数字转换成40位one_hot形式，输出层的[0-9]输出值对应第一个字符的onehot编码，[10-19]输出值对应第二个字符的onehot编码，[20-29]输出值对应第三个字符，[30-39]输出值对于第四个字符
    '''
    model = models.resnet18(num_classes=4 * 10).cuda(0)
    # 定义损失函数
    loss_fn = nn.MSELoss().cuda(0)
    # 优化器
    optim = Adam(model.parameters(), lr=0.001)
    model.train()   # 开启训练模式
    for epoch in range(4):
        now = time.time()
        print(f'训练轮数: {epoch + 1}')
        # tqdm(进度条配置)，train_dataloader=(images, labels)
        bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (images, labels) in bar:
            optim.zero_grad()       # 梯度置为0
            images = images.cuda(0)
            labels = labels.cuda(0)

            outputs = model(images).reshape(32, 4, 10)      # batch_size=32,那后面参数4,10对应的应该就是one-hot编码的预测值
            loss = loss_fn(outputs, labels)                 # 计算损失MSELoss
            loss.backward()     # 反向
            optim.step()        # 梯度优化
            if i % 100 == 0:
                print(f'训练次数: {i}, 损失率: {loss.item()}')
        torch.save(model, f'./save_model/model{epoch}.pth')

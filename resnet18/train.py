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
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=4)    # Ϊɶ����num_workers����4û�б���
    '''
        ģ�����num_classes������4*10
        ÿ����֤��ͼƬ��4���ַ�(��ǩ)������˳��̶���ֻҪ���������������һ���Լ��޸ľ���ʵ�ֶ��ǩ����
        ��֤��һ����4�����֣���4������ת����40λone_hot��ʽ��������[0-9]���ֵ��Ӧ��һ���ַ���onehot���룬[10-19]���ֵ��Ӧ�ڶ����ַ���onehot���룬[20-29]���ֵ��Ӧ�������ַ���[30-39]���ֵ���ڵ��ĸ��ַ�
    '''
    model = models.resnet18(num_classes=4 * 10).cuda(0)
    # ������ʧ����
    loss_fn = nn.MSELoss().cuda(0)
    # �Ż���
    optim = Adam(model.parameters(), lr=0.001)
    model.train()   # ����ѵ��ģʽ
    for epoch in range(4):
        now = time.time()
        print(f'ѵ������: {epoch + 1}')
        # tqdm(����������)��train_dataloader=(images, labels)
        bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (images, labels) in bar:
            optim.zero_grad()       # �ݶ���Ϊ0
            images = images.cuda(0)
            labels = labels.cuda(0)

            outputs = model(images).reshape(32, 4, 10)      # batch_size=32,�Ǻ������4,10��Ӧ��Ӧ�þ���one-hot�����Ԥ��ֵ
            loss = loss_fn(outputs, labels)                 # ������ʧMSELoss
            loss.backward()     # ����
            optim.step()        # �ݶ��Ż�
            if i % 100 == 0:
                print(f'ѵ������: {i}, ��ʧ��: {loss.item()}')
        torch.save(model, f'./save_model/model{epoch}.pth')

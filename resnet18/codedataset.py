# coding=gbk
# -*- coding:uft-8 -*-
# codedataset

import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms


# 制作验证码数据集
class codeDataset(Dataset):         # Dataset是父类
    def __init__(self, root_dir):   # root_dir才是构造函数的参数（__init__()方法又被称为构造器（constructor））
        super(codeDataset, self).__init__()
        # os.listdir(root_dir)就是把'../dataset/data'文件夹里的文件名弄成一个list，所以这行代码的作用就是弄出图片路径的list
        # image_path = ['../dataset/data/0_5177.jpg', '../dataset/data/1000_9403.jpg']
        self.image_path_list = [os.path.join(root_dir, image_name) for image_name in os.listdir(root_dir)]
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([100, 300]),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化不便于前端处理数据
        ])

    def __len__(self):
        return self.image_path_list.__len__()

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        # 转换成RGB
        image = Image.open(image_path).convert('RGB')
        # 对image进行transforms
        image = self.transforms(image)
        # 获取验证码图像的label，如图像的名字为1_4563.jpg，则label为4563
        code_label = image_path.split('/')[-1].split('_')[1].split('.')[0]
        # 将label类型转换成tensor（这里eval函数是执行参数，一个字符串表达式，并返回表达式的值），如将label：4563转成tensor
        code_label = torch.as_tensor([[eval(i)] for i in code_label], dtype=torch.int64)
        '''
            设置one_hot编码
            验证码一共有4个数字，将4个数字转换成40位one_hot形式
            输出层的[0-9]输出值对应第一个字符的onehot编码，[10-19]输出值对应第二个字符的onehot编码，[20-29]输出值对应第三个字符，[30-39]输出值对于第四个字符    
        '''
        one_hot = torch.zeros(4, 10).long()     # 为什么要转换成long类型？？？有点不理解，欢迎交流
        '''
            scatter_函数：简单说就是通过一个张量src来修改另一个张量，哪个元素需要修改、用src中的哪个元素来修改由dim和index决定
            scatter_(dim, index, src)
                dim：沿着哪个维度进行索引,dim为1也就是按照列来修改
                index：用来scatter的元素索引，code_label=tensor([[9],[4],[0],[3]])，所以在index=9、4、0、3处修改
                src：用来scatter的源元素，可以是一个标量或一个张量
        '''
        # index = code_label = tensor([[9],[4],[0],[3]])
        one_hot.scatter_(dim=1, index=code_label.long(), src=torch.ones(4, 10).long())  # scatter：分散
        # 得到one_hot = tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
        one_hot = one_hot.to(torch.float32)
        return image, one_hot


# if __name__ == '__main__':
#     train_data = codeDataset(root_dir='../../dataset/train')     # '../dataset/data'就是参数root_dir
#     print(train_data.__len__())
#     print(train_data.__getitem__(1))
#     print(train_data)

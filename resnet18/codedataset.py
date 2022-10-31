# coding=gbk
# -*- coding:uft-8 -*-
# codedataset

import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms


# ������֤�����ݼ�
class codeDataset(Dataset):         # Dataset�Ǹ���
    def __init__(self, root_dir):   # root_dir���ǹ��캯���Ĳ�����__init__()�����ֱ���Ϊ��������constructor����
        super(codeDataset, self).__init__()
        # os.listdir(root_dir)���ǰ�'../dataset/data'�ļ�������ļ���Ū��һ��list���������д�������þ���Ū��ͼƬ·����list
        # image_path = ['../dataset/data/0_5177.jpg', '../dataset/data/1000_9403.jpg']
        self.image_path_list = [os.path.join(root_dir, image_name) for image_name in os.listdir(root_dir)]
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([100, 300]),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ��һ��������ǰ�˴�������
        ])

    def __len__(self):
        return self.image_path_list.__len__()

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        # ת����RGB
        image = Image.open(image_path).convert('RGB')
        # ��image����transforms
        image = self.transforms(image)
        # ��ȡ��֤��ͼ���label����ͼ�������Ϊ1_4563.jpg����labelΪ4563
        code_label = image_path.split('/')[-1].split('_')[1].split('.')[0]
        # ��label����ת����tensor������eval������ִ�в�����һ���ַ������ʽ�������ر��ʽ��ֵ�����罫label��4563ת��tensor
        code_label = torch.as_tensor([[eval(i)] for i in code_label], dtype=torch.int64)
        '''
            ����one_hot����
            ��֤��һ����4�����֣���4������ת����40λone_hot��ʽ
            ������[0-9]���ֵ��Ӧ��һ���ַ���onehot���룬[10-19]���ֵ��Ӧ�ڶ����ַ���onehot���룬[20-29]���ֵ��Ӧ�������ַ���[30-39]���ֵ���ڵ��ĸ��ַ�    
        '''
        one_hot = torch.zeros(4, 10).long()     # ΪʲôҪת����long���ͣ������е㲻��⣬��ӭ����
        '''
            scatter_��������˵����ͨ��һ������src���޸���һ���������ĸ�Ԫ����Ҫ�޸ġ���src�е��ĸ�Ԫ�����޸���dim��index����
            scatter_(dim, index, src)
                dim�������ĸ�ά�Ƚ�������,dimΪ1Ҳ���ǰ��������޸�
                index������scatter��Ԫ��������code_label=tensor([[9],[4],[0],[3]])��������index=9��4��0��3���޸�
                src������scatter��ԴԪ�أ�������һ��������һ������
        '''
        # index = code_label = tensor([[9],[4],[0],[3]])
        one_hot.scatter_(dim=1, index=code_label.long(), src=torch.ones(4, 10).long())  # scatter����ɢ
        # �õ�one_hot = tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
        one_hot = one_hot.to(torch.float32)
        return image, one_hot


# if __name__ == '__main__':
#     train_data = codeDataset(root_dir='../../dataset/train')     # '../dataset/data'���ǲ���root_dir
#     print(train_data.__len__())
#     print(train_data.__getitem__(1))
#     print(train_data)

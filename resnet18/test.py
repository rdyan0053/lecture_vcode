# coding=gbk
# -*- coding:uft-8 -*-
# test
'''
    ע�⣺testset���Լ�Ҳ��Ҫ��ѵ�������ͼƬһ��������Ϊindex_label.jpg��ʽ
'''

from vcode_ddddocr_resnet.resnet18.codedataset import codeDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

captcha_array = list('0123456789')
captcha_size = 4


def text2Vec(text):
    vec = torch.zeros(captcha_size, len(captcha_array))
    for v in range(len(text)):
        vec[v, captcha_array.index(text[v])] = 1
    return vec


def vec2Text(vec):
    vec = torch.argmax(vec, dim=1)
    text = ''
    for v in vec:
        text += captcha_array[v]
    return text


if __name__ == '__main__':
    test_dataset = codeDataset('../../dataset/test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    model = torch.load('.save_model/model3.pth').cuda(0)

    total = len(test_dataset)
    count = 0
    # tqdm means "progress" in Arabic (taqadum)
    bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

    for i, (images, labels) in bar:
        images = images.cuda(0)
        labels = labels.cuda(0)
        labels = labels.view(-1, len(captcha_array))
        label_text = vec2Text(labels)
        # Ԥ����
        output = model(images)
        # ��Ԥ��������� text��ʽ
        output_text = output.view(-1, len(captcha_array))
        output_text = vec2Text(output_text)

        print(f'Ԥ��: {output_text} ����: {label_text}')
        if output_text == label_text:
            count += 1

    print('��ȷ��: {}%'.format(count / total * 100))

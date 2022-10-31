# coding=gbk
# -*- coding:uft-8 -*-
# test
'''
    注意：testset测试集也需要像训练集里的图片一样，设置为index_label.jpg格式
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
        # 预测结果
        output = model(images)
        # 将预测结果输出成 text格式
        output_text = output.view(-1, len(captcha_array))
        output_text = vec2Text(output_text)

        print(f'预测: {output_text} 期望: {label_text}')
        if output_text == label_text:
            count += 1

    print('正确率: {}%'.format(count / total * 100))

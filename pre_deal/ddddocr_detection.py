'''
    使用ddddocr进行标注数据
'''

import ddddocr
import csv


def detection(dataset_type, csv_path, image_nums):
    writer = csv.writer(open(csv_path, 'w', encoding='utf-8-sig', newline=''))
    writer.writerow(['id', 'data'])
    numLs = [str(i) for i in range(10)]

    # 创建ddddocr对象
    ocr = ddddocr.DdddOcr()

    for i in range(image_nums):
        img = open(f'../../dataset/{dataset_type}/{i}.jpg', 'rb').read()
        res = ocr.classification(img)
        if len(res) == 4 and res[0] in numLs and res[1] in numLs and res[2] in numLs and res[3] in numLs:
            print(i, res)
            writer.writerow([i, res])


if __name__ == '__main__':
    # 标注训练集，将检测出的label放到对应的csv文件
    # detection(dataset_type='pre_train', csv_path='../../dataset/pre_train.csv', image_nums=2826)
    # 标注测试集，将检测出的label放到对应的csv文件
    detection(dataset_type='pre_test', csv_path='../../dataset/pre_test.csv', image_nums=500)

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
            writer.writerow([i, str(res)])  # 防止写入数字前面缺少0


if __name__ == '__main__':
    # 标注训练集，将检测出的label放到对应的csv文件
    # detection(dataset_type='pre_train', csv_path='../../dataset/pre_train.csv', image_nums=2826)
    # 标注测试集，将检测出的label放到对应的csv文件
    detection(dataset_type='pre_test', csv_path='../../dataset/pre_test.csv', image_nums=500)

'''
    出现一个bug
        就是在写入csv的时候，假如数字是0023（从pycharm查看csv文件没有问题）
        但是从windows上的excel查看就会少了前面的数字00
        分析是读取文件出了问题，具体看make_dataset_by_csv文件
'''
'''
    将识别出的结果（csv文件），作为图片的文件名
    只能通过这种方式，也就是重写图片，因为有的图片识别错误，就跳过了，所以csv文件里不全，这里只识别了部分图片
    然后再重写
'''
import pandas as pd


def analy_csv(csv_path, src_path, dest_path):
    df = pd.read_csv(csv_path, dtype=str)   # 加上dtype指定类型，否则可能会出现将0032读取成32的情况
    dicLs = df.to_dict(orient='records', )  # dicLs是一个list，[{'id':0,'data':5177},{'id':1,'data':1676}...]
    i = 0
    for dic in dicLs:
        print(str(dic['data']))
        # ../../dataset/pre_train/0.jpg
        img_src_path = f'../../dataset/{src_path}/{dic["id"]}.jpg'
        img_dest_path = f'../../dataset/{dest_path}/{i}_{dic["data"]}.jpg'
        img = open(img_src_path, 'rb').read()  # 参数rb的含义是：r代表读取，b代表二进制
        open(img_dest_path, 'wb').write(img)
        print(i)
        i += 1


if __name__ == '__main__':
    # 将标注后的训练集里的label从csv文件读出，并作为图片的名字
    analy_csv('../../dataset/pre_train.csv', 'pre_train', 'train')
    # 将标注后的测试集里的label从csv文件读出，并作为图片的名字
    # analy_csv('../../dataset/pre_test.csv', 'pre_test', 'test')

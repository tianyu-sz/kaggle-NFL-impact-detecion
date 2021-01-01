# 撞击数据集转coco
import pandas as pd
import os
import platform

if platform.system() == "Windows":
    data_root_dir = r'D:/IdeaProjects/Kaggle/kaggle-NFL-impact-detecion/'
else:
    data_root_dir = r'/content/data/nfl-video-to-image/'  # 设定目录

df = pd.read_csv(data_root_dir + r'video_labels.csv')

df['image_id'] = df['image_name'].str[:-4]  # 去掉.jpg .png扩展名
# x y w h 在原始数据中已经计算完
df['x_center'] = df['x'] + df['w'] / 2
df['y_center'] = df['y'] + df['h'] / 2
df['classes'] = df['impact'] - 1  # 1 碰, 2 碰了 , 数值-1, 跟yolov5类别号对上,从零开始算

# 归一化
df['x'] = df['x'] / 1280
df['y'] = df['y'] / 720
df['w'] = df['w'] / 1280
df["h"] = df["h"] / 720
df['x_center'] = df['x_center'] / 1280
df['y_center'] = df['y_center'] / 720

from tqdm.auto import tqdm
import shutil as sh

df = df[['image_id', 'x', 'y', 'w', 'h', 'x_center', 'y_center', 'classes']]

# k折数定义
fold_count = 5  # 5 20 10
fold_numbers = [4]  # 0 18 9

index = list(set(df.image_id))

# 源图片目录
source = 'train_images'
if True:
    for fold in fold_numbers:
        val_index = index[len(index) * fold // fold_count:len(index) * (fold + 1) // fold_count]
        for name, mini in tqdm(df.groupby('image_id')):
            if name in val_index:
                path2save = 'val2017/'
            else:
                path2save = 'train2017/'
            if not os.path.exists('convertor/fold{}/labels/'.format(fold) + path2save):
                os.makedirs('convertor/fold{}/labels/'.format(fold) + path2save)
            with open('convertor/fold{}/labels/'.format(fold) + path2save + name + ".txt", 'w+') as f:
                row = mini[['classes', 'x_center', 'y_center', 'w', 'h']].astype(float).values
                row = row.astype(str)
                for j in range(len(row)):
                    text = ' '.join(row[j])
                    f.write(text)
                    f.write("\n")
            if not os.path.exists('convertor/fold{}/images/{}'.format(fold, path2save)):
                os.makedirs('convertor/fold{}/images/{}'.format(fold, path2save))
            sh.copy((data_root_dir + r"{}/{}.png").format(source, name), 'convertor/fold{}/images/{}/{}.png'.format(fold, path2save, name))

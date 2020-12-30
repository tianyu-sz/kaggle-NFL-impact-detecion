import numpy as np
import pandas as pd
import os

data_root_dir = r'D:/IdeaProjects/Kaggle/kaggle-NFL-impact-detecion/'
# data_root_dir = r'/content/data/nfl-impact-detection/'

df = pd.read_csv(data_root_dir + r'image_labels.csv')
df['image_id'] = df['image']
df['x'] = df['left']
df['y'] = df['top']
df['w'] = df["width"]
df["h"] = df['height']
df['x_center'] = df['x'] + df['w']/2
df['y_center'] = df['y'] + df['h']/2
df['classes'] = 0

from tqdm.auto import tqdm
import shutil as sh
df = df[['image_id','x', 'y', 'w', 'h','x_center','y_center','classes']]

# k折数定义
fold_count = 20     # 5 20
fold_numbers = [18]     # 0 18

index = list(set(df.image_id))
source = 'train'
if True:
    for fold in fold_numbers:
        val_index = index[len(index)*fold//fold_count:len(index)*(fold+1)//fold_count]
        for name,mini in tqdm(df.groupby('image_id')):
            if name in val_index:
                path2save = 'val2017/'
            else:
                path2save = 'train2017/'
            if not os.path.exists('convertor/fold{}/labels/'.format(fold)+path2save):
                os.makedirs('convertor/fold{}/labels/'.format(fold)+path2save)
            with open('convertor/fold{}/labels/'.format(fold)+path2save+name+".txt", 'w+') as f:
                row = mini[['classes','x_center','y_center','w','h']].astype(float).values
                row = row/1024
                row = row.astype(str)
                for j in range(len(row)):
                    text = ' '.join(row[j])
                    f.write(text)
                    f.write("\n")
            if not os.path.exists('convertor/fold{}/images/{}'.format(fold,path2save)):
                os.makedirs('convertor/fold{}/images/{}'.format(fold,path2save))
            sh.copy((data_root_dir + r"{}/{}.jpg").format(source,name),'convertor/fold{}/images/{}/{}.jpg'.format(fold,path2save,name))
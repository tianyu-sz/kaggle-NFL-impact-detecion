# kaggle-NFL-impact-detecion
NFL 1st and Future - Impact Detection

Log: 
* helmet_to_coco.py 数据集转coco处理正确, 图片是1280*720的.  看了几张增强后的图片, bbox坐标映射正确. 
* 未统一验证图片大小
* 比较多的图片是小目标. 针对性做优化处理
* kaggle上视频转图片时间较长,考虑保存图片生成后的结果.用于训练

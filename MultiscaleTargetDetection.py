import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l
import os
import pandas as pd
import torchvision
from d2l import torch as d2l



img = d2l.plt.imread('C:\\Users\\RJZhang\\Desktop\\d2l-zh\\pytorch\\img\\catdog.jpg')
h, w = img.shape[:2]

def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes, anchors[0] * bbox_scale)

# display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
# display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
# display_anchors(fmap_w=1, fmap_h=1, s=[0.8])


#@save
d2l.DATA_HUB['banana-detection'] = (d2l.DATA_URL + 'banana-detection.zip',
'5de26c8fce5ccdea9f91267273464dc968d20d72')

#@save
def read_data_bananas(is_train=True):
    """读取⾹蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
        else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                'bananas_val', 'images', f'{img_name}')))
        # 这⾥的target包含（类别，左上⻆x，左上⻆y，右下⻆x，右下⻆y），
        # 其中所有图像都具有相同的⾹蕉类（索引为0）
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256

#@save
class BananasDataset(torch.utils.data.Dataset):
    """⼀个⽤于加载⾹蕉检测数据集的⾃定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
            is_train else f' validation examples'))
    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])
    def __len__(self):
        return len(self.features)

#@save
def load_data_bananas(batch_size):
    """加载⾹蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
        batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
        batch_size)
    return train_iter, val_iter

batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))

# print(batch[0].shape, batch[1].shape)

imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])


plt.show()
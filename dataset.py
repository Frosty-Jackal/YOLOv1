import cv2
from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch

class MyDataset(Dataset):
    def __init__(self,dataset_dir,mode="train", train_val_ratio=0.9):
        self.dataset_dir = dataset_dir
        self.mode = mode
        label_csv = os.path.join(dataset_dir, mode+".csv")
        self.label = np.loadtxt(label_csv)
        self.num_all_data = 100
        all_ids = list(range(self.num_all_data))
        num_train = int(train_val_ratio*self.num_all_data)
        if self.mode == "train":
            self.use_ids = all_ids[:num_train]
    def __len__(self):
        return len(self.use_ids)
    def __getitem__(self, item):
        id = self.use_ids[item]
        label = torch.tensor(self.label[id, :])
        img = Image.open(self.dataset_dir+'/'+f'{id}.jpg')
        trans = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
        ])
        img = trans(img)
        
        return img, label

from util import labels2bbox,draw_bbox
if __name__ == "__main__":

    dataset_dir = "data"
    dataset = MyDataset(dataset_dir)
    print(len(dataset))
    for i in range(100):
        img, label = dataset[i]
        print(img.size())
        img = torch.unsqueeze(img, dim=0)
        preds = label
        preds = preds.view(7,7,30)
        bbox = labels2bbox(preds)
        draw_img = cv2.imread(f'data\\{i}.jpg')
        draw_bbox(draw_img, bbox)
import torch
import torch
import torch.nn as nn
import os
import torchvision.models as tvmodel
from train import epoch
import cv2
from PIL import Image
from dataset import MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from util import *
from train import dataset_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        resnet = tvmodel.resnet34(pretrained=True)  # 调用torchvision里的resnet34预训练模型
        resnet_out_channel = resnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数，方便连入后续卷积网络中
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去除resnet的最后两层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
        )
        self.Conn_layers = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 7 * 7 * (5*2+20)),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.resnet(inputs)
        x = self.Conv_layers(x)
        x = x.view(x.size()[0], -1)
        x = self.Conn_layers(x)
        self.pred = x.reshape(-1, (5 * 2 + 20), 7, 7) 
        return self.pred
    
    def calculate_loss(self, labels):
        self.pred = self.pred.double()
        labels = labels.double()
        num_gridx, num_gridy = 7, 7  
        noobj_confi_loss = 0.  
        coor_loss = 0.  
        obj_confi_loss = 0.  
        class_loss = 0.  
        n_batch = labels.size()[0] 
        for i in range(n_batch): 
            for n in range(num_gridx):  
                for m in range(num_gridy):  
                    if labels[i, 4, m, n] == 1: 
                        bbox1_pred_xyxy = ((self.pred[i, 0, m, n] + n) / num_gridx - self.pred[i, 2, m, n] / 2,
                                           (self.pred[i, 1, m, n] + m) / num_gridy - self.pred[i, 3, m, n] / 2,
                                           (self.pred[i, 0, m, n] + n) / num_gridx + self.pred[i, 2, m, n] / 2,
                                           (self.pred[i, 1, m, n] + m) / num_gridy + self.pred[i, 3, m, n] / 2)
                        bbox2_pred_xyxy = ((self.pred[i, 5, m, n] + n) / num_gridx - self.pred[i, 7, m, n] / 2,
                                           (self.pred[i, 6, m, n] + m) / num_gridy - self.pred[i, 8, m, n] / 2,
                                           (self.pred[i, 5, m, n] + n) / num_gridx + self.pred[i, 7, m, n] / 2,
                                           (self.pred[i, 6, m, n] + m) / num_gridy + self.pred[i, 8, m, n] / 2)
                        bbox_gt_xyxy = ((labels[i, 0, m, n] + n) / num_gridx - labels[i, 2, m, n] / 2,
                                        (labels[i, 1, m, n] + m) / num_gridy - labels[i, 3, m, n] / 2,
                                        (labels[i, 0, m, n] + n) / num_gridx + labels[i, 2, m, n] / 2,
                                        (labels[i, 1, m, n] + m) / num_gridy + labels[i, 3, m, n] / 2)
                        iou1 = calculate_iou(bbox1_pred_xyxy, bbox_gt_xyxy)
                        iou2 = calculate_iou(bbox2_pred_xyxy, bbox_gt_xyxy)
                        if iou1 >= iou2:
                            coor_loss = coor_loss + 5 * (torch.sum((self.pred[i, 0:2, m, n] - labels[i, 0:2, m, n]) ** 2) \
                                        + torch.sum((self.pred[i, 2:4, m, n].sqrt() - labels[i, 2:4, m, n].sqrt()) ** 2))
                            obj_confi_loss = obj_confi_loss + (self.pred[i, 4, m, n] - iou1) ** 2
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((self.pred[i, 9, m, n] - iou2) ** 2)
                        else:
                            coor_loss = coor_loss + 5 * (torch.sum((self.pred[i, 5:7, m, n] - labels[i, 5:7, m, n]) ** 2) \
                                        + torch.sum((self.pred[i, 7:9, m, n].sqrt() - labels[i, 7:9, m, n].sqrt()) ** 2))
                            obj_confi_loss = obj_confi_loss + (self.pred[i, 9, m, n] - iou2) ** 2
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((self.pred[i, 4, m, n] - iou1) ** 2)
                        class_loss = class_loss + torch.sum((self.pred[i, 10:, m, n] - labels[i, 10:, m, n]) ** 2)
                    else:  
                        noobj_confi_loss = noobj_confi_loss + 0.5 * torch.sum(self.pred[i, [4, 9], m, n] ** 2)
        loss = coor_loss + obj_confi_loss + noobj_confi_loss + class_loss
        return loss / n_batch

def calculate_TPFP(bbox, labels):
    stdbbox=labels2bbox(labels)
    TP=0
    FP=0

    for i in range(bbox.shape[0]):
        
        cls_namei = GL_CLASSES[int(bbox[i, 5])]
        if(cls_namei != 'car'):
            continue
        confidencei = bbox[i, 4]
        if confidencei < 0.6:
            continue
        TPpd=0
        FPpd=0
        for j in range(stdbbox.shape[0]):
            cls_namej = GL_CLASSES[int(stdbbox[j, 5])]
            if cls_namei != cls_namej:
                continue
            iou = calculate_iou(bbox[i], stdbbox[j])
            if iou >= 0.5:
                TPpd=1
                break
            
            if iou < 0.5 :
                FPpd=1
        
        if TPpd==1:
            TP+=1
        if FPpd==1 and TPpd==0:
            FP+=1
    return TP,FP

if __name__ == "__main__":
    #eval
    model = Mynet()
    model.load_state_dict(torch.load(f'yolov1mj_state_dict_{epoch}.pth', map_location=device))
    print(model)
    print("Start evaluating...")
    model.eval()
    dataset_dir_test='data'
    trans = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
        ])
    img_list = os.listdir(dataset_dir_test)

    train_dataset = MyDataset(dataset_dir, mode="train", train_val_ratio=0.9)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=1)

    # #img eval
    # for img_name in img_list:
    #     img_path = os.path.join(dataset_dir_test, img_name)
    #     img = Image.open(img_path).convert('RGB')
    #     img = trans(img)
    #     img = torch.unsqueeze(img, dim=0)
    #     print(img_name, img.shape)
    #     preds = torch.squeeze(model(img), dim=0).detach().cpu()
    #     preds = preds.permute(1,2,0)
    #     bbox = labels2bbox(preds)
    #     print("--------------------------------")
    #     print(bbox.shape[1])
    #     draw_img = cv2.imread(img_path)
    #     draw_bbox(draw_img, bbox)
    
    # ap eval
    TP=0
    FP=0
    for i,(imgs, labels) in enumerate(train_loader):
        labels = labels.view(1, 7, 7, -1)
        labels = torch.squeeze(labels, dim=0).detach().cpu()
        preds = torch.squeeze(model(imgs), dim=0).detach().cpu()
        preds = preds.permute(1,2,0)
        bbox = labels2bbox(preds)
        tmpTP,tmpFP = calculate_TPFP(bbox, labels)
        TP+=tmpTP
        FP+=tmpFP
        if i%10==0:
            print('imgnum={} , AP={}'.format(i,TP/(TP+FP+1e-4))   )
    print('finish')


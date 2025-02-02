import torch
import torch.nn as nn
import torchvision.models as tvmodel
from torch.utils.data import DataLoader
from util import *
from dataset import MyDataset
from configure import batch_size, epoch, print_freq,train_freq
dataset_dir='data'

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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




if __name__ == '__main__':
    train_dataset = MyDataset(dataset_dir, mode="train", train_val_ratio=1)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
    model = Mynet().to(device)
    for layer in model.children():
        layer.requires_grad = False
        break
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    #train
    model.train()
    print("Start training...")
    avg_loss = 0.0
    actrual_epoch=epoch
    for e in range(epoch):
        for i,(imgs, labels) in enumerate(train_loader):
            labels = labels.view(batch_size, 7, 7, -1)
            labels = labels.permute(0,3,1,2)
            
            labels=labels.to(device)
            imgs = imgs.to(device) 

            preds = model(imgs) 
            loss = model.calculate_loss(labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            avg_loss = (avg_loss+loss.item())
            if (i+1) % print_freq == 0: 
                print("Epoch %d/%d| training loss = %.3f, avg_loss = %.3f" %
                    (e, epoch, loss.item(), avg_loss/(i+1) ))
        avg_loss = 0.0
        if(e + 1 ) % train_freq == 0 :
            tmp =  input("Do you want to stop trainning? (Y/N): ")
            if(tmp=='Y') :
                actrual_epoch=e+1
                print('Training stopped.')
                break
    torch.save(model.state_dict(), f'yolov1mj_state_dict_{actrual_epoch}.pth')
    print(f'save the yolov1mj_state_dict_{actrual_epoch}.pth')
    
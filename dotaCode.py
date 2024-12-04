import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms  
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image


class DotaObjectDetection(nn.Module):
    def __init__(self, num_classes):
        # super(DotaObjectDetection, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  
        )

        self.cls_head = nn.Conv2d(512, self.num_classes, kernel_size=1)

        self.reg_head = nn.Conv2d(512, 4, kernel_size=1)  
    def forward(self, x):
        features = self.features(x)

        cls_logits = self.cls_head(features)  

        bbox_preds = self.reg_head(features)  

        batch_size = x.size(0)
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()  
        cls_logits = cls_logits.view(batch_size, -1, self.num_classes)  
        bbox_preds = bbox_preds.permute(0, 2, 3, 1).contiguous()  
        bbox_preds = bbox_preds.view(batch_size, -1, 4)  

        return cls_logits, bbox_preds


def compute_loss(cls_logits, bbox_preds, targets):
    batch_size = cls_logits.size(0)
    total_cls_loss = 0.0
    total_reg_loss = 0.0

    for i in range(batch_size):
        cls_logit = cls_logits[i]
        bbox_pred = bbox_preds[i]
        target = targets[i]
        labels = target['labels'] 
        boxes = target['boxes']     

        num_objects = labels.size(0)
        cls_logit = cls_logit[:num_objects]
        bbox_pred = bbox_pred[:num_objects]

        cls_loss = F.cross_entropy(cls_logit, labels)

        reg_loss = F.smooth_l1_loss(bbox_pred, boxes)

        total_cls_loss += cls_loss
        total_reg_loss += reg_loss

    total_loss = total_cls_loss + total_reg_loss
    return total_loss


class DatasetLoaders(Dataset):
    def __init__(self, image_paths, annotations, transform=None):
        self.image_paths = image_paths
        self.annotations = annotations  
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')

        if self.transform:
            img = self.transform(img)

        ann = self.annotations[idx]
        labels = torch.tensor(ann['labels'], dtype=torch.long)
        boxes = torch.tensor(ann['boxes'], dtype=torch.float32)

        target = {'labels': labels, 'boxes': boxes}

        return img, target

transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

from torch.utils.data import DataLoader

image_paths = "./images"
annotations = '.labels'

dataset = DatasetLoaders(image_paths, annotations, transform=transform)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: list(zip(*x)))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 3  
model = DotaObjectDetection(num_classes=num_classes)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        images = torch.stack(images)

        cls_logits, bbox_preds = model(images)

        loss = compute_loss(cls_logits, bbox_preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
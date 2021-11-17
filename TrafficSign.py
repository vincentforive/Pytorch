from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image

class TrafficSign(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# batch_size = 4
#
# trainset = TrafficSign(csv_file='train/_classes.csv', root_dir='train', transform=transforms)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)
#
# testset = TrafficSign(csv_file='test/_classes.csv', root_dir='train', transform=transforms)
#
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)
#
# classes = ('End', 'NoUturn', 'StopAllDay', 'NoEntry',
#           'NoLeft','Stop','NoRight','70','NoStud','GiveWay')
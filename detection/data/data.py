import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ClassificationDataset(Dataset):

    def __init__(self, path_to_data, path_to_labels, stage, input_size=224, transform=None):
        self.path_to_data = path_to_data
        self.transform = transform
        if stage == 'train' or stage == 'valid':
            self.labels = pd.read_csv(path_to_labels)
        else:
            self.labels = None

        # в таком же порядке сохраним пути до файлов
        if self.labels is not None:
            self.file_names = [path_to_data + file_nm for file_nm in self.labels['id']]
        else:
            self.file_names = [
                path_to_data + file_nm for file_nm in sorted(
                    os.listdir(self.path_to_data), key=lambda x: int(x[2:-4])
                )
            ]

        # необходимые трасформации
        self.input_size = input_size

        if stage == 'train':
            self._preprocess = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self._preprocess = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    def __len__(self):
        return len(self.file_names)

    def _load_file(self, path):
        with open(path, 'rb') as input_file:
            with Image.open(input_file) as img:
                return img.convert('RGB')

    def __getitem__(self, idx):
        image = self._load_file(self.file_names[idx])
        image = self._preprocess(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        if self.labels is not None:
            return image, self.labels.iloc[idx]['target_people']
        else:
            return image

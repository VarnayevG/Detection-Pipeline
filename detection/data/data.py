from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ClassificationDataset(Dataset):

    def __init__(self, path_to_data: Path, path_to_labels: Path, stage: str, input_size=224, transform=None):
        self.transform = transform
        if stage == 'train' or stage == 'val':
            self.labels = pd.read_csv(path_to_labels)
            self.file_names = [path_to_data / file_nm for file_nm in self.labels['id']]
        else:
            self.labels = None
            #TODO: parse as regex
            self.file_names = [
                path_to_data / file_nm for file_nm in sorted(path_to_data.glob('*'), key=lambda x: int(x.stem[2:]))
            ]

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

#TODO: add annotations for function outputs
    def _load_file(self, path: Path):
        with open(path, 'rb') as input_file:
            with Image.open(input_file) as img:
                return img.convert('RGB')

    def __getitem__(self, idx: int):
        image = self._load_file(self.file_names[idx])
        image = self._preprocess(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        if self.labels is not None:
            return image, self.labels.iloc[idx]['target_people']
        else:
            return image

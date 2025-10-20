from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD =[0.229,0.224,0.225]

def make_transforms(split):
    aug = []
    if split == "train":
        aug = [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(10),
        ]
    # do PIL-level augs first, then ToTensor, then Normalize
    return transforms.Compose([
        *aug,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

class ManifestDataset(Dataset):
    def __init__(self, csv_path, split):
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)
        self.paths  = df["path"].tolist()
        self.labels = df["label"].astype("category")
        self.y      = self.labels.cat.codes.values
        self.classes= list(self.labels.cat.categories)
        self.tfm    = make_transforms(split)

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        x = Image.open(self.paths[i]).convert("RGB")
        return self.tfm(x), int(self.y[i])

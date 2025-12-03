from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD =[0.229,0.224,0.225]

def make_albu_transforms(split):
    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),                       
            A.Rotate(limit=15, border_mode=1, p=0.7),     
            A.RandomResizedCrop(224, 224, scale=(0.9, 1.1), p=0.7),

            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),      
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.2),

            A.GaussNoise(var_limit=(5, 30), p=0.3),          
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),     

            A.ElasticTransform(alpha=10, sigma=10, alpha_affine=5, p=0.2),
            A.Affine(scale=(0.95, 1.05),
                     translate_percent=(0.02, 0.02),
                     rotate=0,
                     shear=0,
                     p=0.25),
            A.CoarseDropout(
                max_holes=2,
                max_height=40,
                max_width=40,
                min_holes=1,
                fill_value=0,
                p=0.3
            ),

            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    
    else:  # val/test
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

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

# class ManifestDataset(Dataset):
#     def __init__(self, csv_path, split):
#         df = pd.read_csv(csv_path)
#         df = df[df["split"] == split].reset_index(drop=True)
#         self.paths  = df["path"].tolist()
#         self.labels = df["label"].astype("category")
#         self.y      = self.labels.cat.codes.values
#         self.classes= list(self.labels.cat.categories)
#         self.tfm    = make_transforms(split)

#     def __len__(self): return len(self.paths)

#     def __getitem__(self, i):
#         x = Image.open(self.paths[i]).convert("RGB")
#         return self.tfm(x), int(self.y[i])

class ManifestDataset(Dataset):
    def __init__(self, csv_path, split):
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)

        self.paths   = df["path"].tolist()
        self.labels  = df["label"].astype("category")
        self.y       = self.labels.cat.codes.values
        self.classes = list(self.labels.cat.categories)
        
        self.tfm = make_albu_transforms(split)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
  
        augmented = self.tfm(image=img)
        img = augmented["image"]  

        return img, int(self.y[idx])

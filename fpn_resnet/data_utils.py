
import os 
import torch 
import random
import pickle
import augly.image as imgaug

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from vision_augs import get_transform
from torch.utils.data import Dataset, DataLoader


class RetrievalDataset(Dataset):
    
    def __init__(self, paths, transforms):
        super(RetrievalDataset, self).__init__()
        self.aug_transform = get_transform(transforms["aug"])
        self.std_transform = get_transform(transforms["std"])
        self.paths = paths 
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        orig = self.std_transform(img)
        aug_1 = self.aug_transform(img)
        aug_2 = self.aug_transform(img)
        return orig, aug_1, aug_2
    

class EvaluationDataloader:
    
    def __init__(self, paths, batch_size, transform):
        self.transform = transform
        self.batch_size = batch_size
        self.paths = paths 
        self.ptr = 0 
        
    def __len__(self):
        return len(self.paths) // self.batch_size
    
    def get(self):
        imgs, paths = [], []
        for _ in range(self.batch_size):
            try:
                path = self.paths[self.ptr]
                img = Image.open(path).convert("RGB")
                imgs.append(self.transform(img))
                path = path.replace(".png", ".jpg")
                paths.append(path)   
            except Exception as e:
                pass  
            
            self.ptr += 1
            if self.ptr >= len(self.paths):
                self.ptr = 0
                      
        imgs = torch.stack(imgs, dim=0)
        return {"img": imgs, "path": paths}
    

def collate_func(batch):
    img, aug1, aug2 = zip(*batch)
    img, aug1, aug2 = list(img), list(aug1), list(aug2)
    img, aug1, aug2 = torch.stack(img, 0), torch.stack(aug1, 0), torch.stack(aug2, 0)
    return {"img": img, "aug_1": aug1, "aug_2": aug2}

def get_mini_loaders(train_dir, val_dir, batch_size, transforms):
    train_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    val_paths = [os.path.join(val_dir, f) for f in os.listdir(val_dir)]
    train_dataset = RetrievalDataset(paths=train_paths, transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_func)
    val_loader = EvaluationDataloader(paths=val_paths, batch_size=batch_size, transform=get_transform(transforms["std"]))
    return train_loader, val_loader

def augment_val_set(dir, emoji_dir, num_augs=4):
    img_paths = [os.path.join(dir, f) for f in os.listdir(dir)]
    emoji_paths = [os.path.join(dir, f) for f in os.listdir(emoji_dir)]
    labels = {}
    count = 0 
    
    for path in tqdm(img_paths, total=len(img_paths), desc="Progress"):
        try:
            for _ in range(num_augs):
                emoji = random.choice(emoji_paths)
                transform = transforms.Compose([
                    imgaug.RandomBlur(min_radius=0.0, max_radius=2.0, p=0.3),
                    imgaug.RandomEmojiOverlay(opacity=0.5, p=0.2),
                    imgaug.ColorJitter(brightness_factor=0.4, contrast_factor=0.4, saturation_factor=0.4, p=0.8),
                    imgaug.Resize(width=256, height=256)
                ])
                img = Image.open(path).convert("RGB")
                aug_img = transform(img)
                aug_img.save(dir + f"{str(count)}.png")
                if path in labels.keys():
                    labels[path].append(count)
                else:
                    labels[path] = [count]
                count += 1
        except Exception as e:
            continue
        
    with open("data/val_labels.pk", "wb") as f:
        pickle.dump(labels, f)
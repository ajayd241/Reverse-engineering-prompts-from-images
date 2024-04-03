import os
import pandas as pd 
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sentence_transformers import SentenceTransformer
import pytorch_lightning as pl

from transformers import AutoProcessor
from config import CONFIG, BLIPConfig

class ViTDataset(Dataset):
    def __init__(self, image_dir, df, transform):
        self.image_dir = image_dir
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, row['image_name']))
        image = self.transform(image)
        prompt = row['prompt']
        return row['image_name'], image, prompt
    
class ViTTestDataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.transform(image)
        return image
    
class ViTCollator:
    def __init__(self):
        self.st_model = SentenceTransformer(
            '../all-MiniLM-L6-v2',
            device='cpu'
        )
    
    def __call__(self, batch):
        image_names, images, prompts = zip(*batch)
        images = torch.stack(images)
        prompt_embeddings = self.st_model.encode(
            prompts, 
            show_progress_bar=False, 
            convert_to_tensor=True
        )
        return image_names, images, prompt_embeddings
    
    
class ViTDataModule(pl.LightningDataModule):

    def __init__(self, image_dir, train_df, test_df, train_transform, test_transform, batch_size):
        super(ViTDataModule, self).__init__()
        self.image_dir = image_dir
        self.train_df = train_df 
        self.test_df = test_df
        self.train_transform = train_transform 
        self.test_transform = test_transform
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage=='fit' or stage is None:
            self.train_set = ViTDataset(self.image_dir, self.train_df, self.train_transform)
            self.collator = ViTCollator()
        if stage=='test' or stage is None:
            self.test_set = ViTDataset(self.image_dir, self.test_df, self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, batch_size=self.batch_size, pin_memory=True, num_workers=2, drop_last=True, collate_fn=self.collator) 

    def val_dataloader(self):
        return DataLoader(self.test_set, shuffle=False, batch_size=self.batch_size, pin_memory=True, num_workers=2, drop_last=True, collate_fn=self.collator) 
    
    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle=False, batch_size=self.batch_size, pin_memory=True, num_workers=2, drop_last=True, collate_fn=self.collator) 
    

class BLIPDataset(Dataset):

    def __init__(self, image_dir, df, processor):
        self.image_dir = image_dir 
        self.df = df
        self.processor= processor 

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_name = row['image_name']
        image = Image.open(os.path.join(self.image_dir, image_name))
        prompt = row['prompt']
        encoding = self.processor(images=image, text=prompt, 
                                  padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return image_name, encoding
    

class BLIPDataModule(pl.LightningDataModule):

    def __init__(self, image_dir, train_df, test_df, processor, batch_size=4):
        super(BLIPDataModule, self).__init__()
        self.image_dir = image_dir
        self.train_df = train_df
        self.test_df = test_df
        self.processor = processor
        self.batch_size = batch_size 

    def setup(self, stage=None):
        if stage=='fit' or stage is None:
            self.train_set = BLIPDataset(self.image_dir, self.train_df, self.processor)
        if stage=='test' or stage is None:
            self.test_set = BLIPDataset(self.image_dir, self.test_df, self.processor)

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, batch_size=self.batch_size, pin_memory=True, num_workers=2) 

    def val_dataloader(self):
        return DataLoader(self.test_set, shuffle=False, batch_size=self.batch_size, pin_memory=True, num_workers=2)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle=False, batch_size=self.batch_size, pin_memory=True, num_workers=2)


class CLIPInterrogatorDataset(Dataset):
    def __init__(self, image_dir, df):
        self.image_dir = image_dir
        self.df = df
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, row['image_name'])).convert("RGB")
        prompt = row['prompt']
        return row['image_name'], image, prompt


class CLIPInterrogatorDataModule(pl.LightningDataModule):

    def __init__(self, image_dir, train_df, test_df, batch_size=4):
        super(CLIPInterrogatorDataModule, self).__init__()
        self.image_dir = image_dir
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size 

    def setup(self, stage=None):
        if stage=='fit' or stage is None:
            self.train_set = CLIPInterrogatorDataset(self.image_dir, self.train_df)
        if stage=='test' or stage is None:
            self.test_set = CLIPInterrogatorDataset(self.image_dir, self.test_df)

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, batch_size=self.batch_size, pin_memory=True, num_workers=2) 

    def val_dataloader(self):
        return DataLoader(self.test_set, shuffle=False, batch_size=self.batch_size, pin_memory=True, num_workers=2)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle=False, batch_size=self.batch_size, pin_memory=True, num_workers=2)
    

class OFADataset(Dataset):

    def __init__(self, root, batch_size=32):
        self.root = root
        self.im_paths = os.listdir(self.root)
        self.batch_size = batch_size
        self.sz = len(self.im_paths)
        self.genlen = self.sz//self.batch_size + int(self.sz%self.batch_size > 0)
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 480
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ])
     
    def __len__(self):
        return self.genlen
        
    def __getitem__(self, index):
        if index >= self.genlen:
            raise IndexError("Out of bounds")
        
        l, r = index*self.batch_size, min(self.sz, (index+1)*self.batch_size)
        
        f_paths = [os.path.join(self.root, self.im_paths[i]) for i in range(l,r)]
        f_ids = [self.im_paths[i][:-4] for i in range(l,r)]
        
        ims = [Image.open(f_path) for f_path in f_paths]
        ims = [self.patch_resize_transform(im).cuda().unsqueeze(0) for im in ims]
        ims = torch.cat(ims)
        
        return ims, f_ids
    

if __name__ == "__main__":

    metadata_df = pd.read_parquet('../metadata.parquet')
    images = os.listdir('../images/')[:100]
    metadata_df = metadata_df[metadata_df['image_name'].isin(images)]
    train_df, test_df = train_test_split(metadata_df, test_size=0.2, random_state=42)

    # testing ViT Dataset
    transform = transforms.Compose([
        transforms.Resize((CONFIG.input_size, CONFIG.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    dm = ViTDataModule('../images/', train_df, test_df, train_transform=transform, test_transform=transform, batch_size=CONFIG.batch_size)
    dm.prepare_data()
    dm.setup()

    print(len(dm.test_set))

    for _, X, y in dm.train_dataloader():
        print(_)
        print(X.shape)
        print(y.shape)
        break

    # Testing BLIP Dataset
    processor =  AutoProcessor.from_pretrained(BLIPConfig.model_name)
    dm = BLIPDataModule('../images', train_df, test_df, processor)
    dm.prepare_data()
    dm.setup() 

    for _, encoding in dm.train_dataloader():
        print(_)
        print(len(encoding['input_ids'][0]))
        break


import os
import sys
sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import timm
import pytorch_lightning as pl

from src.dataset import DiffusionDataModule, DiffusionTestDataset
from src.model import VisionTransformer
from src.config import CONFIG

def main():
    args = get_args() 
    metadata_df = pd.read_parquet('../metadata.parquet')
    images = os.listdir(args.data_dir)
    metadata_df = metadata_df[metadata_df['image_name'].isin(images)]
    train_df, test_df = train_test_split(metadata_df, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((CONFIG.input_size, CONFIG.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    dm = DiffusionDataModule('../images/', train_df, test_df, train_transform=transform, test_transform=transform, batch_size=CONFIG.batch_size)
    dm.prepare_data()
    dm.setup()

    t_max = args.max_epochs * len(dm.train_dataloader())
    state_dict = torch.load(args.model_dir)
    model = VisionTransformer(t_max=t_max)
    model.load_state_dict(state_dict)
    model.eval()

    if args.test_data_dir:
        images = list(Path(args.test_data_dir).glob('*.png'))
        imgIds = [i.stem for i in images]
        EMBEDDING_LENGTH = 384
        imgId_eId = [
            '_'.join(map(str, i)) for i in zip(
                np.repeat(imgIds, EMBEDDING_LENGTH),
                np.tile(range(EMBEDDING_LENGTH), len(imgIds)))]
        dataset = DiffusionTestDataset(images, transform)
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=64,
            pin_memory=True,
            num_workers=2,
            drop_last=False
        )

        preds = []
        for names, X, _ in tqdm(dataloader, leave=False):
            with torch.no_grad():
                X_out = model(X)
                preds.append(X_out.cpu().numpy())
    else:
        dataloader = dm.test_dataloader()

        preds = []
        imgId_eId = []
        for names, X, _ in tqdm(dataloader, leave=False):
            imgId_eId.extend(names)
            with torch.no_grad():
                X_out = model(X)
                preds.append(X_out.cpu().numpy())
    
    prompt_embeddings = np.vstack(preds).flatten()
    torch.save(prompt_embeddings, 'vit_embeddings.pt')
    submission = pd.DataFrame(
        index=imgId_eId,
        data=prompt_embeddings,
        columns=['val']
    ).rename_axis('imgId_eId')
    submission.to_csv('submission.csv')


def get_args():
    parser = ArgumentParser() 
    parser.add_argument("--data_dir", default="../images/", type=str)
    parser.add_argument("--test_data_dir", default="../images/", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VisionTransformer.add_model_specific_args(parser)
    args = parser.parse_args() 
    return args 
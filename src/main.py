import os
import sys
sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser
import random
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms
import timm
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings('ignore')

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from transformers import AutoProcessor

from src.dataset import DiffusionDataModule, BLIPDataModule
from src.model import VisionTransformer, BLIP
from src.config import CONFIG, BLIPConfig


def main() -> None: 
    args = get_args() 

    metadata_df = pd.read_parquet('../metadata.parquet')
    images = os.listdir(args.data_dir)[:200000]
    print(len(images))
    metadata_df = metadata_df[metadata_df['image_name'].isin(images)]
    train_df, test_df = train_test_split(metadata_df, test_size=0.2, random_state=42)

    if args.model == "vit":

        transform = transforms.Compose([
            transforms.Resize((CONFIG.input_size, CONFIG.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
        dm = DiffusionDataModule('../images/', train_df, test_df, train_transform=transform, test_transform=transform, batch_size=CONFIG.batch_size)
        dm.prepare_data()
        dm.setup()

        t_max = args.max_epochs * len(dm.train_dataloader())
        model = VisionTransformer(t_max=t_max)

        wandb_logger = loggers.WandbLogger(name='vit-base-patch16-224-i2p', save_dir=".")
        early_stopping = EarlyStopping('val_loss_epoch', patience=5)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss_epoch",
            dirpath="../checkpoints/",
            filename="vit-{epoch:02d}-{val_loss:.2f}",
            save_top_k=5,

        )

        checkpoint_callback.FILE_EXTENSION = ".pth.tar"
        lr_monitor = LearningRateMonitor(logging_interval="step")

        trainer = pl.Trainer(gpus=args.gpus,
            max_epochs=args.max_epochs, 
            callbacks=[early_stopping, checkpoint_callback, lr_monitor],
            logger=wandb_logger
        )
                    
        trainer.fit(model, datamodule=dm)
        trainer.test(datamodule=dm)

        wandb_logger.experiment.finish()
    
    elif args.model == "blip":
        processor =  AutoProcessor.from_pretrained(BLIPConfig.model_name)
        dm = BLIPDataModule('../images', train_df, test_df, processor)
        dm.prepare_data()
        dm.setup() 

        model = BLIP()

        wandb_logger = loggers.WandbLogger(name='blip-image-captioning-base-i2p', save_dir=".")
        early_stopping = EarlyStopping('val_loss_epoch', patience=5)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss_epoch",
            dirpath="../checkpoints/",
            filename="blip-{epoch:02d}-{val_loss:.2f}",
            save_top_k=5,

        )

        checkpoint_callback.FILE_EXTENSION = ".pth.tar"
        lr_monitor = LearningRateMonitor(logging_interval="step")

        trainer = pl.Trainer(gpus=args.gpus,
            max_epochs=args.max_epochs, 
            callbacks=[early_stopping, checkpoint_callback, lr_monitor],
            logger=wandb_logger
        )
                    
        trainer.fit(model, datamodule=dm)
        trainer.test(datamodule=dm)

        wandb_logger.experiment.finish()


def get_args():
    parser = ArgumentParser() 
    parser.add_argument("--data_dir", default="../images/", type=str)
    parser.add_argument("--model", default="vit", type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VisionTransformer.add_model_specific_args(parser)
    parser = BLIP.add_model_specific_args(parser)
    args = parser.parse_args() 
    return args 


if __name__ == "__main__":
    pl.seed_everything(42)
    main()
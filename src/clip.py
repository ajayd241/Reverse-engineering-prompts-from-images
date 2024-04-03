import os
import sys
import inspect
from argparse import ArgumentParser
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import open_clip
from BLIP.models import blip
from clip_interrogator import clip_interrogator

sys.path.append('../sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer, models
from config import CLIPConfig

import pytorch_lightning as pl

from dataset import CLIPInterrogatorDataset, CLIPInterrogatorDataModule

def interrogate(image: Image) -> str:
    caption = ci.generate_caption(image)
    image_features = ci.image_to_features(image)
    
    medium = [ci.mediums.labels[i] for i in cos(image_features, mediums_features_array).topk(1).indices][0]
    movement = [ci.movements.labels[i] for i in cos(image_features, movements_features_array).topk(1).indices][0]
    flaves = ", ".join([ci.flavors.labels[i] for i in cos(image_features, flavors_features_array).topk(3).indices])

    if caption.startswith(medium):
        prompt = f"{caption}, {movement}, {flaves}"
    else:
        prompt = f"{caption}, {medium}, {movement}, {flaves}"

    return clip_interrogator._truncate_to_fit(prompt, ci.tokenize)

def add_text_limiters(text: str) -> str:
    return " ".join([
        word + "\n" if i % 15 == 0 else word 
        for i, word in enumerate(text.split(" "), start=1)
    ])

def plot_image(image: np.ndarray, original_prompt: str, generated_prompt: str) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.annotate(
        "Original prompt:\n" + add_text_limiters(original_prompt) + "\n\nGenerated prompt:\n" + add_text_limiters(generated_prompt), 
        xy=(1.05, 0.5), xycoords='axes fraction', ha='left', va='center', 
        fontsize=16, rotation=0, color="#104a6e"
    )

def get_args():
    parser = ArgumentParser() 
    parser.add_argument("--data_dir", default="../images/", type=str)
    args = parser.parse_args() 
    return args

if __name__ == "__main__":
    pl.seed_everything(42)
    args = get_args() 
    st_model = SentenceTransformer(CLIPConfig.sentence_model_path)
    model_config = clip_interrogator.Config(clip_model_name=CLIPConfig.ci_clip_model_name)
    model_config.cache_path = CLIPConfig.cache_path
    blip_path = inspect.getfile(blip)
    configs_path = os.path.join(os.path.dirname(os.path.dirname(blip_path)), 'configs')
    med_config = os.path.join(configs_path, 'med_config.json')
    blip_model = blip.blip_decoder(
        pretrained=CLIPConfig.blip_model_path,
        image_size=model_config.blip_image_eval_size, 
        vit=model_config.blip_model_type, 
        med_config=med_config
    )
    blip_model.eval()
    blip_model = blip_model.to(model_config.device)
    model_config.blip_model = blip_model

    clip_model = open_clip.create_model(CLIPConfig.clip_model_name, precision='fp16' if model_config.device == 'cuda' else 'fp32')
    open_clip.load_checkpoint(clip_model, CLIPConfig.clip_model_path)
    clip_model.to(model_config.device).eval()
    model_config.clip_model = clip_model

    clip_preprocess = open_clip.image_transform(
    clip_model.visual.image_size,
        is_train = False,
        mean = getattr(clip_model.visual, 'image_mean', None),
        std = getattr(clip_model.visual, 'image_std', None),
    )
    model_config.clip_preprocess = clip_preprocess

    ci = clip_interrogator.Interrogator(model_config)

    cos = torch.nn.CosineSimilarity(dim=1)

    mediums_features_array = torch.stack([torch.from_numpy(t) for t in ci.mediums.embeds]).to(ci.device)
    movements_features_array = torch.stack([torch.from_numpy(t) for t in ci.movements.embeds]).to(ci.device)
    flavors_features_array = torch.stack([torch.from_numpy(t) for t in ci.flavors.embeds]).to(ci.device)

    image_names = []
    original_prompts = []
    predicted_prompts = []

    metadata_df = pd.read_parquet('../metadata.parquet')
    images = os.listdir(args.data_dir)[:200000]
    print(len(images))
    metadata_df = metadata_df[metadata_df['image_name'].isin(images)]
    train_df, test_df = train_test_split(metadata_df, test_size=0.2, random_state=42)

    dm = CLIPInterrogatorDataModule('../images/', train_df, test_df, batch_size=CLIPConfig.batch_size)
    dm.prepare_data()
    dm.setup()

    images = list(Path(args.test_data_dir).glob('*.png'))
    imgIds = [i.stem for i in images]
    EMBEDDING_LENGTH = 384
    imgId_eId = [
        '_'.join(map(str, i)) for i in zip(
            np.repeat(imgIds, EMBEDDING_LENGTH),
            np.tile(range(EMBEDDING_LENGTH), len(imgIds)))]
    dataset = CLIPInterrogatorDataset(images, test_df)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=64,
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )

    for names, X, y in dataloader():
        generated = interrogate(X)
        original_prompts.append(y)
        predicted_prompts.append(generated)
        image_names.append(names[0])

    for image_name, original_prompt, predicted_prompt in zip(image_names, original_prompts, predicted_prompts):
        img = Image.open(args.data_dir + image_name).convert("RGB")
        plot_image(img, original_prompt, predicted_prompt)
        prompt_embeddings = st_model.encode(predicted_prompts).flatten()
    torch.save(prompt_embeddings, 'clip_embeddings.pt')
    submission = pd.DataFrame(
        index=imgId_eId,
        data=prompt_embeddings,
        columns=['val']
    ).rename_axis('imgId_eId')

    submission.to_csv('submission.csv')
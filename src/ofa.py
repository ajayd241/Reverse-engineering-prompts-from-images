import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator
from sentence_transformers import SentenceTransformer, models

from dataset import OFADataset

CKPT_DIR = "../OFA-large-caption/"
IMAGE_DIR = "../images"

BATCH_SIZE = 24

tokenizer = OFATokenizer.from_pretrained(CKPT_DIR)
model = OFAModel.from_pretrained(CKPT_DIR, use_cache=False).cuda()
txt = "what does the image describe?"
inputs = tokenizer([txt], return_tensors="pt").input_ids
st_model = SentenceTransformer('../all-MiniLM-L6-v2')

sub_ids = []
sub_embeds = []

dataset = OFADataset(IMAGE_DIR, BATCH_SIZE)

for batch in dataset:
    for j in range(len(batch[1])):
        sub_ids.extend([f"{batch[1][j]}_{i}" for i in range(384)])
    
    img_batch =batch[0]
    out = model.generate(inputs.repeat(len(img_batch), 1).cuda(), patch_images=img_batch, num_beams=5, no_repeat_ngram_size=2)
    out_captions = tokenizer.batch_decode(out, skip_special_tokens=True)
    out_captions = [cap + ", fine details, masterpiece" for cap in out_captions]
    
    embeddings = st_model.encode(out_captions).flatten()
    sub_embeds.extend(embeddings)

torch.save(embeddings, 'ofa_embeddings.pt')
sub = pd.DataFrame({"imgId_eId": sub_ids, "val": sub_embeds})
print(sub.shape)
sub.head()
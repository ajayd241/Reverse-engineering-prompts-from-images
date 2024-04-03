import os 
import torch 

# load all the embeddings 
vit_embedding = torch.load("vit_embedding.pt")
blip_embedding = torch.load("blip_embedding.pt")
clip_embedding = torch.load("clip_embedding.pt")
ofa_embedding = torch.load("ofa_embedding.pt")

# ensemble embedding 
ensemble_embedding = 0.3 * vit_embedding + 0.4 * blip_embedding + 0.2 * clip_embedding + 0.1 * ofa_embedding
torch.save(ensemble_embedding, "ensemble_embedding.pt")
class CONFIG:
    model_name = 'vit_base_patch16_224'
    input_size = 224
    batch_size = 64
    num_epochs = 3
    lr = 1e-4
    seed = 42

class BLIPConfig:
    model_name = "Salesforce/blip-image-captioning-base"
    train_batch_size = 4
    valid_batch_size: 8
    learning_rate =  1e-4
    scheduler =  'CosineAnnealingLR'
    min_lr = 1e-6
    T_max =  500
    weight_decay = 1e-6
    n_accumulate =  1

class CLIPConfig:
    device = "cuda"
    seed = 42
    embedding_length = 384
    sentence_model_path = "../all-MiniLM-L6-v2"
    blip_model_path = "../clip-interrogator-models-x/model_large_caption.pth"
    ci_clip_model_name = "ViT-H-14/laion2b_s32b_b79k"
    clip_model_name = "ViT-H-14"
    clip_model_path = "../clip-interrogator-models-x/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
    cache_path = "../clip-interrogator-models-x"
    batch_size = 1
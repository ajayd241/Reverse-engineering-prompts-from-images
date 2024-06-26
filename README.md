<div align="center">    
 
# Image to Prompts

</div>
 
## Description   
Predict prompt embeddings for images generated by Stable Diffusion. We use an ensemble of 4 models viz.,

1. Vision Transformer 
2. BLIP 
3. CLIP Interrogator (pretrained)
4. OFA (pretrained)

The prompts are converted into 384-dimension prompt embeddings using a pretrained Sentence Transformer.

## How to run   
First, install dependencies (a new python virtual environment is recommended).   
```bash
# clone project   
git clone https://github.com/the-neural-networker/image-2-prompts

# install project   
cd image-2-prompts
pip install -r requirements.txt
 ```   
 Next, navigate to `src` folder and run main.py with appropriate command line arguments, to train the vision transformer or the BLIP model.
 ```bash
# module folder
cd src

# run module
python main.py --gpus=1 --max_epochs=10
```

This `main.py` script requires a `data-dir` with the images in it, as well a `metadata.parquet` to read the prompts. The prompts and images can be downloaded from https://poloclub.github.io/diffusiondb/. The `sentence-transformers` library is also needed to convert prompts to their corresponding embeddings. The `sentence-transformers` library can be installed from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2.

To use pretrained CLIP Interrogator or OFA models, run either `clip.py` or `ofa.py`. To make the codes run, the pretrained models need to be downloaded and the paths need to given appropriately. 

To run `ofa.py` a modified version of the `transformers` library is needed, which can be installed by:

```bash
git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
pip install OFA/transformers/
git clone https://huggingface.co/OFA-Sys/OFA-large
```
# References

- https://www.kaggle.com/code/inversion/calculating-stable-diffusion-prompt-embeddings
- https://lightning.ai/docs/pytorch/latest/
- https://www.kaggle.com/code/shoheiazuma/stable-diffusion-vit-baseline-train
- https://huggingface.co/docs/transformers/model_doc/blip
- https://www.kaggle.com/code/mayukh18/ofa-transformer-lb-0-42644
- https://www.kaggle.com/code/inversion/stable-diffusion-sample-submission
- https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/clip#overview
- https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/vit#overview
- https://www.kaggle.com/code/leonidkulyk/lb-0-45836-blip-clip-clip-interrogator
- https://www.kaggle.com/code/debarshichanda/pytorch-blip-training

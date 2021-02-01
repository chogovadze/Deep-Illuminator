import yaml
import torchvision.utils as vutils
import torch
import argparse
import os
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from probe_relighting.streamlit.vae.models import *
from probe_relighting.streamlit.vae.experiment import VAEXperiment
import torch.backends.cudnn as cudnn
# from pytorch_lightning import Trainer
from torchvision.transforms.functional import to_pil_image
# from pytorch_lightning.callbacks import ModelCheckpoint
import streamlit as st
st.title('VAE Inspection')

def get_user_image(filename):
    return Image.open(filename)
gray_image = st.sidebar.file_uploader('Upload Gray', type=['jpg', 'png'])
chrome_image = st.sidebar.file_uploader('Upload Chrome', type=['jpg', 'png'])

 
filename = 'streamlit/vae/configs/bhvae.yaml'
with open(filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])


model_path = f'streamlit/vae/vae_checkpoint.ckpt'
state_dict = torch.load(model_path)['state_dict']
for key in list(state_dict.keys()):
    state_dict[key.replace('model.', '')] = state_dict.pop(key)
model.load_state_dict(state_dict)
model.cuda()


def standard_transform(x, y):
    resizing = transforms.Compose([transforms.Resize((x, y)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5))])
    return resizing


probe_tr = standard_transform(64, 64)



first_mu = st.sidebar.slider('First_mu', -1.0, 1.0, step=0.1)
second_mu = st.sidebar.slider('Second_mu', -1.0, 1.0, step=0.1)
third_mu = st.sidebar.slider('Third_mu', -1.0, 1.0, step=0.01)
fourth_mu = st.sidebar.slider('Fourth_mu', -1.0, 1.0, step=0.1)
# first_log = st.sidebar.slider('First_log', -1.0, 1.0, step=0.1)
# second_log = st.sidebar.slider('Second_log', -1.0, 1.0, step=0.1)
# third_log = st.sidebar.slider('Third_log', -1.0, 1.0, step=0.1)
# fourth_log = st.sidebar.slider('Fourth_log', -1.0, 1.0, step=0.1)
sample = torch.Tensor([first_mu, second_mu, third_mu, fourth_mu])
input_sample = sample.unsqueeze(0).cuda()
output = model.decode(input_sample)
chrome, gray = torch.split(output, 3, 1)


def open_probe(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = probe_tr(image)
    image = image.cuda()
    return image


def get_probes(gray, chrome):
    chrome = open_probe(chrome)
    gray = open_probe(gray)
    a = torch.cat([chrome, gray], dim=0)
    return a

def pilify(x): 
    x = vutils.make_grid(x, nrow=1, padding=0,normalize=True)
    return Image.fromarray(x.cpu().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
col1, col2 = st.beta_columns(2)
col1.image(pilify(gray), use_column_width=True)
col2.image(pilify(chrome), use_column_width=True)


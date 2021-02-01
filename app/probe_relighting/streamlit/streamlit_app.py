import torch
import base64
from pathlib import Path
from PIL import Image

import streamlit as st
from probe_relighting.utils.demotools import *
from probe_relighting.utils.preprocessing import *
from probe_relighting.streamlit.vae.models import *
from probe_relighting.streamlit.vae.experiment import VAEXperiment

import yaml
import torchvision.utils as vutils
import torch
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
#
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def probe_path(texture, style, angle, **kwargs):
    if style == 'Synthetic Probes':
        path = f'./data/{light.lower()}_{intensity}/{texture}_{str(angle).zfill(4)}.png'
    else:
        path = f'./data/mid_probes/dir_{angle}_{texture}256.jpg'

    return path
# Descriptions
st.title('Probe Relighting Demonstration')
st.markdown('---')
st.markdown('This is a simple application to visualize the effects of each \
             setting on the generated image.')
st.markdown('---')
st.sidebar.markdown(f'Device is: {device}')
st.sidebar.markdown('## Relighting Style')

# Probe Style Selection
style = st.sidebar.selectbox('Chose the relighting probes you would like to use:', 
                             ['Synthetic Probes', 'MID Averaged Probes', 'VAE Probes'])

if style == 'Synthetic Probes':
    light = st.sidebar.selectbox('Light Source', ['Spot', 'Point'])
    intensity = st.sidebar.select_slider('Intensity', ['1kW', '5kW', '10kW', '25kW'])
    # axis = st.sidebar.selectbox('Axis', [1, 2, 3, 4, 5])
    axis = 1
    angle = st.sidebar.slider('Angle', min_value=1, max_value=360, step=1)
    ang =  360 * (axis-1) + max(0, axis-2) * 45 + angle
    chrome_name = probe_path('chrome', 'Synthetic Probes', ang, light=light, intensity=intensity)
    gray_name = probe_path('gray', 'Synthetic Probes', ang, light=light, intensity=intensity)
    chrome = Image.open(chrome_name)
    gray = Image.open(gray_name)

elif style == 'MID Averaged Probes':
    step = 1
    direction = st.sidebar.slider('MID Direction', min_value=1, max_value=24, step=1)
    light = None
    intensity = None
    chrome_name = probe_path('chrome', 'MID', direction)
    gray_name = probe_path('gray', 'MID', direction)
    chrome = Image.open(chrome_name).resize((128, 128))
    gray = Image.open(gray_name).resize((128, 128))

elif style == 'VAE Probes':

    vae_filename = 'streamlit/vae/configs/bhvae.yaml'
    with open(vae_filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    vae_model = vae_models[config['model_params']['name']](**config['model_params'])
    vae_experiment = VAEXperiment(vae_model,
                              config['exp_params'])


    vae_model_path = f'streamlit/vae/vae_checkpoint.ckpt'
    vae_state_dict = torch.load(vae_model_path)['state_dict']
    for key in list(vae_state_dict.keys()):
        vae_state_dict[key.replace('model.', '')] = vae_state_dict.pop(key)
    vae_model.load_state_dict(vae_state_dict)
    vae_model.cuda()


    def standard_transform(x, y):
        resizing = transforms.Compose([transforms.Resize((x, y)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))])
        return resizing


    probe_tr = standard_transform(128, 128)



    first_mu = st.sidebar.slider('First_mu', -3.0, 3.0, step=0.1)
    second_mu = st.sidebar.slider('Second_mu', -3.0, 3.0, step=0.1)
    third_mu = st.sidebar.slider('Third_mu', -3.0, 3.0, step=0.01)
    fourth_mu = st.sidebar.slider('Fourth_mu', -3.0, 3.0, step=0.1)
    sample = torch.Tensor([first_mu, second_mu, third_mu, fourth_mu])
    input_sample = sample.unsqueeze(0).cuda()
    output = vae_model.decode(input_sample)
    chrome, gray = torch.split(output, 3, 1)


    def get_probes(gray, chrome):
        chrome = open_probe(chrome)
        gray = open_probe(gray)
        a = torch.cat([chrome, gray], dim=0)
        return a

    def pilify(x): 
        x = vutils.make_grid(x, nrow=1, padding=0,normalize=True)
        return Image.fromarray(x.cpu().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

    col1, col2 = st.beta_columns(2)
    gray = pilify(gray)
    chrome = pilify(chrome)
    gray_name = gray
    chrome_name = chrome


# Displaying Input Images
col1, col2 = st.beta_columns(2)
with col1:
    st.markdown('### Chrome Probe')
    st.image(chrome, use_column_width=True)
with col2:
    st.markdown('### Gray Probe')
    st.image(gray, use_column_width=True)
st.sidebar.markdown('## Scene')
eg_image = st.sidebar.selectbox('Example Images', ['Office', 'Outdoor', 'Mannequin'])
st.sidebar.markdown('Or upload your own')


# Generating Sample
@st.cache()
def get_user_image(filename):
    return Image.open(filename)
user_image = st.sidebar.file_uploader('Upload Image', type=['jpg', 'png'])

if user_image:
    user_image = get_user_image(user_image)
    image_name = 'user_image'
else:
    user_image = Image.open(f'./streamlit/example_images/{eg_image.lower()}.jpg') 
    image_name = eg_image 
@st.cache(allow_output_mutation=True)
def load_model():
    return get_model()


def input_sample(img, gray_name, chrome_name):
    if type(gray_name) is str:
        chrome = open_probe(chrome_name).unsqueeze(0)
        gray = open_probe(gray_name).unsqueeze(0)
    else:
        chrome = probe_tr(chrome_name).cuda().unsqueeze(0)
        gray = probe_tr(gray_name).cuda().unsqueeze(0)
    return {'original': img, 'probe_1': chrome, 'probe_2': gray}


model = load_model()
@st.cache(show_spinner=False)
def generate_frame(pil_img, chrome_name, gray_name):
    image = pil_img.convert('RGB')
    image = image_tr(image).unsqueeze(0)
    image = image.to(device)
    model.to(device)
    with torch.no_grad():
        model.eval()
        sample = input_sample(image, gray_name, chrome_name)
        output = model(sample)
        probes = torch.cat((sample['probe_1'], sample['probe_2']), 3)
        gif_output = torch.cat([probes, output['generated_img']], dim=2)
        output = output['generated_img']
        output = denorm(output.cpu().squeeze())
        output = to_pil_image(output)
        gif_output = denorm(gif_output.cpu().squeeze())
        gif_output = to_pil_image(gif_output)
 
    return output, gif_output

output, _ = generate_frame(user_image, chrome_name, gray_name)
st.markdown('### Generated Image')
st.image(output, use_column_width=True)

if style != 'VAE Probes':

    st.sidebar.markdown('## Create GIF')
    if style == 'Synthetic Probes':
        step = st.sidebar.number_input('Increment', min_value=1, value=1, step=1)

    generate_button = st.sidebar.button(label='Generate GIF')

    if generate_button:
        st.markdown('## Genertated GIF from user settings')
        gif_path = Path(f'./streamlit/temp/{style}_{image_name}_{step}_{intensity}.gif')
        if not gif_path.is_file():
            outputs = []
            style_dict = {'MID Averaged Probes': (1, 24), 'Synthetic Probes': (1, 360)}
            r_min = min(style_dict[style])
            r_max = max(style_dict[style])
            st.markdown("Generating GIF:")
            progress = st.progress(0)
            for idx in range(r_min, r_max, step):
                chr_name = probe_path('chrome', style, idx, light=light, intensity=intensity)
                gr_name = probe_path('gray', style, idx, light=light, intensity=intensity)
                _, gif_frame = generate_frame(user_image, chr_name, gr_name)
                outputs.append(gif_frame)
                progress.progress(int(100*idx/r_max))
            progress.progress(100)
            outputs[0].save(gif_path, save_all=True,
                            append_images=outputs[1:], loop=0, duration=200)

        file_ = open(gif_path, "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="Probe GIF" width=700>',
            unsafe_allow_html=True, 
        )

import yaml
import torch
from pathlib import Path
from probe_relighting.utils.preprocessing import denorm, open_image, open_probe
from probe_relighting.network import ProbeRelighting
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILE_PATH = Path(__file__)


def get_model():
    experiment_path = FILE_PATH.parent / '../network_config.yaml'
    with open(experiment_path) as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    model = ProbeRelighting
    model = model(opt['model_opt'])
    model_path = FILE_PATH.parent / '../checkpoint.ckpt'
    model.load_state_dict(torch.load(model_path, 
                                     map_location=device)['model_state_dict'])
    return model


def make_sample(img, style, idx):
    if style == 'synthetic':
        chrome_name = f'./data/point_1kW/chrome_{str(idx).zfill(4)}.png'
        gray_name = f'./data/point_1kW/gray_{str(idx).zfill(4)}.png'
    elif style == 'mid':
        chrome_name = f'./data/mid_probes/dir_{idx}_chrome256.jpg'
        gray_name = f'./data/mid_probes/dir_{idx}_gray256.jpg'
    chrome = open_probe(chrome_name).unsqueeze(0)
    gray = open_probe(gray_name).unsqueeze(0)
    return {'original': img, 'probe_1': chrome, 'probe_2': gray}


def generate_outputs(model, img, increments, style):
    idx = 0
    model.to(device)
    outputs = []
    if style == 'mid':
        minr = 1
        maxr = 25 
    else:
        minr = 1
        maxr = 361
    with torch.no_grad():
        model.eval()
        img_range = range(minr, maxr, increments)
        for idx in tqdm(img_range):
            sample = make_sample(img, style, idx)
            probes = torch.cat((sample['probe_1'], sample['probe_2']), 3)
            output = model(sample)
            output = torch.cat([probes, output['generated_img']], dim=2)
            output = denorm(output.cpu().squeeze())
            output = to_pil_image(output)
            outputs.append(output)
        return outputs

def get_images():
    org_path = Path('./originals/')
    img_paths = org_path.iterdir()
    img_paths = [x for x in img_paths if '.jpg' in x.name or '.png' in x.name]
    return list(img_paths)

import time
import torch
from pathlib import Path
from PIL import Image

from probe_relighting.utils.demotools import *
from probe_relighting.utils.options import args


# Loading model with weights
model = get_model()

# read images from original folder
img_list = get_images()

for img in tqdm(img_list):
    img_input = open_image(img).unsqueeze(0)
    # Getting generated images
    outputs = generate_outputs(model, img_input, increments=args.step, style=args.mode)

    # Saving GIF
    save_path = Path(f'./output/{img.stem}/{args.mode}_{args.intensity}kW_step{args.step}/')
    save_path.mkdir(parents=True, exist_ok=True)
    name = Path(save_path, f'{img.stem}.gif')
    outputs[0].save(name, save_all=True,
                    append_images=outputs[1:], loop=0, duration=200)
    for idx, ig in enumerate(outputs):
        name = Path(save_path, f'{img.stem}_{idx}.jpg')
        ig.save(name)

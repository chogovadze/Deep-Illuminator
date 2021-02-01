import torch
from torchvision import transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def standard_transform(x, y):
    resizing = transforms.Compose([transforms.Resize((x, y)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5))])
    return resizing


image_tr = standard_transform(256, 256)
probe_tr = standard_transform(128, 128)


def open_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image_tr(image)
    image = image.to(device)
    # image = np.transpose(np.array(image), (2, 0, 1))

    return image


def open_probe(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = probe_tr(image)
    image = image.to(device)
    return image


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

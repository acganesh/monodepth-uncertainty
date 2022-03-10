from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import networks
from utils import download_model_if_doesnt_exist

MODEL_ROOT = "/root/tmp/"
HEIGHT = 192
WIDTH = 640

def load_model(model_name):
    model_path = os.path.join(MODEL_ROOT, model_name)
    encoder_path = os.path.join(MODEL_ROOT, model_name, "encoder.pth")
    depth_decoder_path = os.path.join(MODEL_ROOT, model_name, "depth.pth")

    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4), dropout_rate=0.0)

    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    encoder.eval()
    depth_decoder.eval()
    print("Model loading complete!")
    return encoder, depth_decoder

def load_image(image_path):
    input_image = pil.open(image_path).convert('RGB')
    original_width, original_height = input_image.size

    #feed_height = loaded_dict_enc['height']
    #feed_width = loaded_dict_enc['width']
    feed_height = HEIGHT
    feed_width = WIDTH
    input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

    input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
    return input_image_pytorch, original_height, original_width

def predict(encoder, decoder, input_image_pytorch):
    with torch.no_grad():
        features = encoder(input_image_pytorch)
        outputs = decoder(features)

        disp = outputs[("disp", 0)]
    return disp

def visualize(disp, original_height, original_width, title, outpath):
    disp_resized = torch.nn.functional.interpolate(disp,
	(original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)

    plt.clf()
    plt.figure(figsize=(10, 10))

    plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)
    plt.title(title, fontsize=22)
    plt.axis('off');

    plt.savefig(outpath)


def construct_model_ids():
    # Only use 20epochs-a for this investigation.
    MODELS = ["20epochs-a"]
    result = []
    for m in MODELS:
        result.append(f"{m}/models/weights_19")
    return result

def t2n(disp):
    return disp.squeeze().cpu().numpy()

def resize(disp, original_height, original_width):
    disp_resized = torch.nn.functional.interpolate(disp,
                (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = t2n(disp_resized)
    return disp_resized_np

def get_aug_list():
    augs = []
    augs.append(("identity", transforms.RandomHorizontalFlip(p=0.0))) # identity transform
    augs.append(("horizontal_flip", transforms.RandomHorizontalFlip(p=1.0)))
    augs.append(("vertical_flip", transforms.RandomVerticalFlip(p=1.0)))
    return augs

def main():
    models = construct_model_ids()
    model = models[0]

    disps = []

    encoder, decoder = load_model(model)
    image_pytorch, orig_height, orig_width = load_image("assets/test_image.jpg")

    augs = get_aug_list()

    for aug_name, aug in augs:
        print(f"Applying {aug_name}...")
        aug_img = aug(image_pytorch)

        disp = predict(encoder, decoder, aug_img)
        aug_disp = aug(disp)

        disp_np = t2n(aug_disp)
        disps.append(disp_np)
     
    disps = torch.tensor(np.array(disps))
    mean_map = torch.reshape(torch.mean(disps, axis=0), (1, 1, HEIGHT, WIDTH))
    var_map = torch.reshape(torch.var(disps, axis=0), (1, 1, HEIGHT, WIDTH))

    visualize(mean_map, orig_height, orig_width, "Mean of depth estimation", "/tmp/mean.png")
    visualize(var_map, orig_height, orig_width, "Uncertainty of depth estimation (variance)", "/tmp/var.png")

if __name__ == '__main__':
    main()

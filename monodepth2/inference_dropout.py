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
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4), dropout_rate=0.1)

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
    # For now we use 20epochs-dropout-retry
    MODELS = ["20epochs-dropout-retry"]
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

def main():
    models = construct_model_ids()
    model = models[0]

    disps = []

    encoder, decoder = load_model(model)
    image_pytorch, orig_height, orig_width = load_image("assets/test_image.jpg")

    NUM_INFERENCES = 10
    for i in range(NUM_INFERENCES):
        disp = predict(encoder, decoder, image_pytorch)
        disp_np = t2n(disp)
        disp_resized_np = resize(disp, orig_height, orig_width)
        disps.append(disp_np)
     
    disps = torch.tensor(np.array(disps))
    mean_map = torch.reshape(torch.mean(disps, axis=0), (1, 1, HEIGHT, WIDTH))
    var_map = torch.reshape(torch.var(disps, axis=0), (1, 1, HEIGHT, WIDTH))

    visualize(mean_map, orig_height, orig_width, "Mean of depth estimation", "/tmp/mean.png")
    visualize(var_map, orig_height, orig_width, "Uncertainty of depth estimation (variance)", "/tmp/var.png")

if __name__ == '__main__':
    main()

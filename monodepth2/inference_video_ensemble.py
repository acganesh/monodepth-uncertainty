from __future__ import absolute_import, division, print_function

import io

import imageio
import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import datasets
from layers import disp_to_depth
import networks
from options import MonodepthOptions
from utils import download_model_if_doesnt_exist, readlines

MODEL_ROOT = "/root/tmp/"
splits_dir = os.path.join(os.path.dirname(__file__), "splits")
HEIGHT = 192
WIDTH = 640

MODELS = {}

def load_model(model_name, i):
    if i not in MODELS:
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

        encoder.cuda()
        depth_decoder.cuda()

        encoder.eval()
        depth_decoder.eval()

        MODELS[i] = (encoder, depth_decoder)
        print("Model loading complete!")
    else:
        encoder, depth_decoder = MODELS[i]
        print("Using cached model!")
        
    return encoder, depth_decoder

def predict(encoder, decoder, input_image_pytorch):
    with torch.no_grad():
        features = encoder(input_image_pytorch)
        outputs = decoder(features)

        disp = outputs[("disp", 0)]
    return disp

def visualize(disp, original_height, original_width, title, outpath, reverse=True):
    if reverse:
        cmap = 'magma_r'
    else:
        cmap = 'magma'
    disp_resized = torch.nn.functional.interpolate(disp,
	(original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)

    plt.clf()
    plt.figure(figsize=(10, 10))

    plt.imshow(disp_resized_np, cmap=cmap, vmax=vmax)
    plt.title(title, fontsize=22)
    plt.axis('off');

    plt.savefig(outpath)

def show_img(image, title, outpath):
    image = np.squeeze(image)
    image = np.transpose(image, [1, 2, 0])
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.savefig(outpath)


def construct_model_ids():
    MODELS = ["20epochs-a", "20epochs-b", "20epochs-c-retry"]
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

def main(opt):
    models = construct_model_ids()

    N = 70

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    #filenames = readlines(os.path.join(splits_dir, "odom", "test_files_10.txt"#)

    f0 = filenames[0]
    filenames_consecutive = []

    path0, frame0, side0 = f0.split(' ')
    for i in range(N):
        # Can replace 0 with "frame 0" to start at frame 0
        n = 0 + i
        framei = f'{n:10}'
        fc = f'{path0} {framei} {side0}'
        filenames_consecutive.append(fc)

    print("Obtained filenames")
    ds = datasets.KITTIRAWDataset(opt.data_path, filenames_consecutive[:N],
                                  HEIGHT, WIDTH, [0], 1, is_train=False)
    dataloader = DataLoader(ds, 1, shuffle=False, num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    print("Data loader initialized!")

    disps = []
    images = []
    all_means = []
    all_vars = []

    img_gif = []
    mean_gif = []
    var_gif = []

    for i, data in enumerate(dataloader):
        #image_pytorch, orig_height, orig_width = 
        image_pytorch = data[("color", 0, 0)].cuda()
        orig_height = image_pytorch.shape[-2]
        orig_width = image_pytorch.shape[-1]
        images.append(image_pytorch.cpu().numpy())

        cur_depths = []

        for j, model in enumerate(models):
            encoder, decoder = load_model(model, j)
            disp = predict(encoder, decoder, image_pytorch)
            pred_disp, pred_depth = disp_to_depth(disp, opt.min_depth, opt.max_depth)
            #depth_np = t2n(pred_depth)
           
            disp_resized_np = resize(pred_disp, orig_height, orig_width)
            #depth_resized_np = resize(pred_depth, orig_height, orig_width)
            cur_depths.append(disp_resized_np)
     
        cur_depths = torch.tensor(np.array(cur_depths))
        mean_map = torch.reshape(torch.mean(cur_depths, axis=0), (1, 1, HEIGHT, WIDTH))
        var_map = torch.reshape(torch.var(cur_depths, axis=0), (1, 1, HEIGHT, WIDTH))

        all_means.append(mean_map)
        all_vars.append(var_map)

        mean_path = f"/tmp/results/mean_{i}.png"
        var_path = f"/tmp/results/var_{i}.png"
        image_path = f"/tmp/results/image_{i}.png"

        mean_buf = io.BytesIO()
        visualize(mean_map, orig_height, orig_width, "Mean of depth estimation", mean_buf, reverse=False)
        mean_buf.seek(0)
        mean_gif.append(imageio.imread(mean_buf))

        var_buf = io.BytesIO()
        visualize(var_map, orig_height, orig_width, "Uncertainty of depth estimation (variance)", var_buf, reverse=False)
        var_buf.seek(0)
        var_gif.append(imageio.imread(var_buf))

        img_buf = io.BytesIO()
        show_img(image_pytorch.cpu().numpy(), f"Image {i}", img_buf)
        img_buf.seek(0)
        img_gif.append(imageio.imread(img_buf))

        print(f"Progress: {i}")

    imageio.mimsave('/tmp/results/images.gif', img_gif, format="GIF", loop=0)
    imageio.mimsave('/tmp/results/mean.gif', mean_gif, format="GIF", loop=0)
    imageio.mimsave('/tmp/results/vars.gif', var_gif, format="GIF", loop=0)



if __name__ == '__main__':
    options = MonodepthOptions()
    main(options.parse())

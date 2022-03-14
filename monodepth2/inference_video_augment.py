from __future__ import absolute_import, division, print_function

import io

import imageio
import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
from tqdm import tqdm

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
HEIGHT = 192 # same as orig_height
WIDTH = 640 # same as orig_width
DROPOUT_RATE = 0.0

MODELS = {}

def load_model(model_name):
    if model_name not in MODELS:
        model_path = os.path.join(MODEL_ROOT, model_name)
        encoder_path = os.path.join(MODEL_ROOT, model_name, "encoder.pth")
        depth_decoder_path = os.path.join(MODEL_ROOT, model_name, "depth.pth")

        encoder = networks.ResnetEncoder(18, False)
        depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4), dropout_rate=DROPOUT_RATE)

        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        depth_decoder.load_state_dict(loaded_dict)

        encoder.eval()
        depth_decoder.eval()

        encoder.cuda()
        depth_decoder.cuda()

        MODELS[model_name] = (encoder, depth_decoder)
        print("Model loading complete!")
    else:
        encoder, depth_decoder = MODELS[model_name]
        
    return encoder, depth_decoder

def predict(encoder, decoder, input_image_pytorch):
    with torch.no_grad():
        features = encoder(input_image_pytorch)
        outputs = decoder(features)

        disp = outputs[("disp", 0)]
    return disp

def visualize(ax, disp, original_height, original_width, title, outpath, reverse=True):
    if reverse:
        cmap = 'magma_r'
    else:
        cmap = 'magma'
    disp_resized = torch.nn.functional.interpolate(disp,
	(original_height, original_width), mode="bilinear", align_corners=False)

    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)

    ax.imshow(disp_resized_np, cmap=cmap, vmax=vmax)
    ax.axis('off');

def show_img(ax, image, title, outpath):
    image = np.squeeze(image)
    image = np.transpose(image, [1, 2, 0])
    ax.imshow(image)
    #ax.set_title(title)
    ax.axis('off')
    #plt.savefig(outpath)
    #plt.clf()


def construct_model_ids():
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

def get_scene(scenes, max_num_frames, eval_filenames, scene_id):
    scene_filenames = []
    if scene_id in scenes:
        N = min(5, max_num_frames[scene_id] - 1)
        i = scenes[scene_id]

        f0 = eval_filenames[i]

        path0, frame0, side0 = f0.split(' ')
        for i in range(N):
            framei = f'{i:10}'
            fc = f'{path0} {framei} {side0}'
            scene_filenames.append(fc)

    return N, scene_filenames

def main(opt):
    models = construct_model_ids()

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

    """
    List of keys in eval scene set:
    """
    SCENES = {'2011_09_26/2011_09_26_drive_0002_sync': 0, # the first one we looked at, has biker
    '2011_09_26/2011_09_26_drive_0009_sync': 25,
    '2011_09_26/2011_09_26_drive_0013_sync': 50,
    '2011_09_26/2011_09_26_drive_0020_sync': 75,
    '2011_09_26/2011_09_26_drive_0023_sync': 100,
    '2011_09_26/2011_09_26_drive_0027_sync': 125,
    '2011_09_26/2011_09_26_drive_0029_sync': 150,
    '2011_09_26/2011_09_26_drive_0036_sync': 175,
    '2011_09_26/2011_09_26_drive_0046_sync': 200,
    '2011_09_26/2011_09_26_drive_0048_sync': 225,
    '2011_09_26/2011_09_26_drive_0052_sync': 247,
    '2011_09_26/2011_09_26_drive_0056_sync': 272,
    '2011_09_26/2011_09_26_drive_0059_sync': 297,
    '2011_09_26/2011_09_26_drive_0064_sync': 322,
    '2011_09_26/2011_09_26_drive_0084_sync': 347,
    '2011_09_26/2011_09_26_drive_0086_sync': 372,
    '2011_09_26/2011_09_26_drive_0093_sync': 397,
    '2011_09_26/2011_09_26_drive_0096_sync': 422,
    '2011_09_26/2011_09_26_drive_0101_sync': 447,
    '2011_09_26/2011_09_26_drive_0106_sync': 472,
    '2011_09_26/2011_09_26_drive_0117_sync': 497,
    '2011_09_28/2011_09_28_drive_0002_sync': 522,
    '2011_09_29/2011_09_29_drive_0071_sync': 547,
    '2011_09_30/2011_09_30_drive_0016_sync': 572,
    '2011_09_30/2011_09_30_drive_0018_sync': 597,
    '2011_09_30/2011_09_30_drive_0027_sync': 622,
    '2011_10_03/2011_10_03_drive_0027_sync': 647,
    '2011_10_03/2011_10_03_drive_0047_sync': 672} # Has a bunch of cars on highway

    MAX_NUM_FRAMES = {}

    base_path = '/home/acg/ss-monodepth/monodepth2/kitti_data'
    def all_same(items):
        return all(x == items[0] for x in items)

    for scene in SCENES:
        scene_base_path = os.path.join(base_path, scene)
        nfiles_all = []
        for im in ['image_00', 'image_01', 'image_02', 'image_03']:
            full_path = os.path.join(scene_base_path, im, "data")
            files = os.listdir(full_path)
            numfiles = len(files)
            nfiles_all.append(numfiles)

        # Ensure that we have the same number of frames across all subdirs.
        assert(all_same(nfiles_all))
        MAX_NUM_FRAMES[scene] = nfiles_all[0]
   
    scene_list = list(SCENES.keys())
    for scene_ctr, scene_name in tqdm(enumerate(scene_list)):
        print("Processing Scene Name: ", scene_name)
        print(f"Scene Progress: {scene_ctr}/{len(scene_list)}")

        N, scene_filenames = get_scene(SCENES, MAX_NUM_FRAMES, filenames, scene_name)
        ds = datasets.KITTIRAWDataset(opt.data_path, scene_filenames[:N],
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

        encoder, decoder = load_model(models[0])

        for i, data in enumerate(dataloader):
            image_pytorch = data[("color", 0, 0)].cuda()
            orig_height = image_pytorch.shape[-2]
            orig_width = image_pytorch.shape[-1]
            images.append(image_pytorch.cpu().numpy())

            cur_depths = []

            augs = []
            augs.append(("identity", transforms.RandomHorizontalFlip(p=0.0))) # identity transform
            augs.append(("horizontal_flip", transforms.RandomHorizontalFlip(p=1.0)))
            augs.append(("color_jitter", transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)))
            augs.append(("color_jitter", transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)))
            augs.append(("blur", transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.1))))
            augs.append(("blur", transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.1))))

            for aug_name, aug in augs:
                aug_img = aug(image_pytorch)
                disp = predict(encoder, decoder, aug_img)
                pred_disp, pred_depth = disp_to_depth(disp, opt.min_depth, opt.max_depth)

                if aug_name == 'horizontal_flip':
                    # Need to flip the disp back in the case of horizontal flips.
                    pred_disp = aug(pred_disp)

                disp_resized_np = resize(pred_disp, orig_height, orig_width)
                cur_depths.append(disp_resized_np)
         
            cur_depths = torch.tensor(np.array(cur_depths))
            mean_map = torch.reshape(torch.mean(cur_depths, axis=0), (1, 1, HEIGHT, WIDTH))
            var_map = torch.reshape(torch.var(cur_depths, axis=0), (1, 1, HEIGHT, WIDTH))

            all_means.append(mean_map)
            all_vars.append(var_map)

            fig, (ax1, ax2, ax3) = plt.subplots(3)

            buf = io.BytesIO()
            show_img(ax3, image_pytorch.cpu().numpy(), f"Image {i}", None)
            visualize(ax1, mean_map, orig_height, orig_width, "Mean of depth estimation", None, reverse=False)
            visualize(ax2, var_map, orig_height, orig_width, "Uncertainty of depth estimation (variance)", None, reverse=False)


            plt.tight_layout()
            plt.savefig(buf, bbox_inches='tight')
            buf.seek(0)
            img_gif.append(imageio.imread(buf))
            plt.close(fig)

            print(f"Frame Progress: {i}")

        scene_name_sanitized = scene_name.replace('/', '-')
        basepath = f'/tmp/results-v2-augment'
        if not os.path.exists(basepath):
            os.makedirs(basepath)

        imageio.mimsave(f'{basepath}/{scene_name_sanitized}.gif', img_gif, format="GIF", loop=0)


if __name__ == '__main__':
    options = MonodepthOptions()
    main(options.parse())

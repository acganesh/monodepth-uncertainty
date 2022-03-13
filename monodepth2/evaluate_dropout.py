from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

MIN_DEPTH = 1e-3
MAX_DEPTH = 80
NUM_INFERENCES = 10

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def compute_rmse(gt, pred):
    """Computation of RMSE between predicted and ground truth depths
    """
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    return rmse

def evaluate_image(opt, gt_depth, pred_disps):
    gt_height, gt_width = gt_depth.shape[:2]
    if opt.eval_split == "eigen":
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)
    else:
        mask = gt_depth > 0
    gt_depth = gt_depth[mask]

    pred_depths = []

    for i in range(NUM_INFERENCES):
        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        pred_depth = pred_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        
        pred_depths.append(pred_depth)
    
    pred_depths = np.vstack(pred_depths)
    
    pred_depths_mean = np.mean(pred_depths, axis=0)
    pred_depths_var = np.var(pred_depths, axis=0)

    depth_error = np.abs(gt_depth - pred_depths_mean)
    return depth_error, pred_depths_var


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MODEL_PATH = "/root/tmp/20epochs-dropout-retry/models/weights_19"
    MODEL_PATH = os.path.expanduser(MODEL_PATH)

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    
    MODEL_PATH = "/root/tmp/20epochs-dropout-retry/models/weights_19"
    encoder_path = os.path.join(MODEL_PATH, "encoder.pth")
    decoder_path = os.path.join(MODEL_PATH, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                        encoder_dict['height'], encoder_dict['width'],
                                        [0], 4, is_train=False)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, 
                                            dropout_rate=opt.dropout_rate)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, allow_pickle=True, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")

    print("   Mono evaluation - using median scaling")
    

    pred_disps = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            pred_disps_batch = []

            input_color = data[("color", 0, 0)].cuda()

            for _ in range(NUM_INFERENCES):
                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                pred_disps_batch.append(pred_disp)

            pred_disps_batch = np.stack(pred_disps_batch, axis=1)
            pred_disps.append(pred_disps_batch)

    pred_disps = np.concatenate(pred_disps)
    mean_errors = []
    mean_vars = []
    
    for i in range(pred_disps.shape[0]):
        pred_disps_single_img = pred_disps[i]
        gt_depth = gt_depths[i]
        
        depth_error, pred_depths_var = evaluate_image(opt, gt_depth, pred_disps_single_img)

        mean_error = np.sqrt((depth_error ** 2).mean())
        mean_var = pred_depths_var.mean()

        mean_errors.append(mean_error)
        mean_vars.append(mean_var)

    plt.figure()
    plt.scatter(mean_errors, mean_vars)
    plt.title("Mean RMSE vs Mean Per-Pixel Variance with Dropout Model")
    plt.xlabel("Mean RMSE")
    plt.ylabel("Mean Per-Pixel Variance")
    plt.savefig("assets/per_image_dropout_plot.png")
    print("Mean RMSE", np.mean(mean_errors))
    print("Mean Variance (per pixel)", np.mean(mean_vars))
    
    plt.figure()

    plt.scatter(depth_error.flatten(), pred_depths_var.flatten())
    plt.title("Depth Errors vs Variance of Detph Values with Dropout Model")
    plt.xlabel("Depth Errors (Absolute Difference)")
    plt.ylabel("Variance of Estimated Detph Values")
    plt.savefig("assets/per_pixel_dropout_plot.png")

    print("\n-> Done!")

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())


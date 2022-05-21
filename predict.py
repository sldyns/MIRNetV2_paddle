import os

import glob

import cv2
import argparse
import utils
from networks.MIRNet_V2_model import MIRNet_v2
import paddle
import numpy as np
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description="MIRNet_val")
parser.add_argument("--model_ckpt", type=str, default="model_best.pdparams", help='path to model checkpoint')
parser.add_argument("--data_path", type=str, default="SIDD_patches/val_mini/", help='path to training data')
parser.add_argument("--save_path", type=str, default="results/", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('--save_images', action='store_true', help='Save images in result directory')

opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def main():

    if opt.save_images:
        os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = MIRNet_v2(n_feat=64)

    utils.load_checkpoint(model, opt.model_ckpt)

    model.eval()

    # load data info
    print('Loading data info ...\n')
    clean_files = sorted(os.listdir(os.path.join(opt.data_path, 'groundtruth')))
    noisy_files = sorted(os.listdir(os.path.join(opt.data_path, 'input')))

    clean_filenames = [os.path.join(opt.data_path, 'groundtruth', x) for x in clean_files if utils.is_png_file(x)]
    noisy_filenames = [os.path.join(opt.data_path, 'input', x) for x in noisy_files if utils.is_png_file(x)]

    # test
    psnr_test = 0
    ssim_test = 0
    for idx in range(len(clean_filenames)):
        # image
        clean = utils.load_img(clean_filenames[idx])
        noisy = utils.load_img(noisy_filenames[idx])

        clean = clean.transpose([2, 0, 1])
        noisy = noisy.transpose([2, 0, 1])

        clean = np.expand_dims(clean, 0)
        noisy = np.expand_dims(noisy, 0)

        clean = paddle.Tensor(clean)
        noisy = paddle.Tensor(noisy)

        with paddle.no_grad():  # this can save much memory
            restored = paddle.clip(model(noisy), 0., 1.)

        psnr = utils.batch_PSNR(restored, clean, 1.)
        ssim = utils.batch_SSIM(restored, clean)

        psnr_test += psnr
        ssim_test += ssim

        if opt.save_images:
            restored = restored.transpose([0, 2, 3, 1]).squeeze().numpy()

            denoised_img = img_as_ubyte(restored)
            utils.save_img(opt.save_path + clean_filenames[idx].split('/')[-1][:-4] + '.png', denoised_img)

    psnr_test /= len(clean_filenames)
    ssim_test /= len(clean_filenames)

    print("\nPSNR on test data {:.4f}, SSIM on test data {:.4f}, ".format(psnr_test, ssim_test))

if __name__ == "__main__":
    main()

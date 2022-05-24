import os
from config import Config

import argparse


import paddle

import paddle.optimizer as optim
from paddle.io import DataLoader
import random
import time
import numpy as np

import utils
from dataloaders.data_rgb import get_training_data, get_validation_data

from utils import MixUp

from networks.MIRNet_V2_model import MIRNet_v2
from networks.MIRNet_model import MIRNet

from losses import CharbonnierLoss
import paddle.distributed as dist

from visualdl import LogWriter


parser = argparse.ArgumentParser(description="MIRNet_TIPC_train")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--data_dir", type=str, default="SIDD_patches/train/", help="path of train dataset")
parser.add_argument("--val_dir", type=str, default="SIDD_patches/val/", help="path of val dataset")
parser.add_argument("--log_dir", type=str, default="output", help="path of save results")
parser.add_argument("--patch_size", type=int, default=128, help="Training patch size")
parser.add_argument("--model", type=str, default="MIRNet", help='model for train')

opt = parser.parse_args()

def main():
    dist.init_parallel_env()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
   
    print(nranks)

    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    paddle.seed(1234)

    start_epoch = 1

    result_dir = os.path.join(opt.log_dir, 'results')
    model_dir = os.path.join(opt.log_dir, 'models')
    log_dir = os.path.join(opt.log_dir, 'log')


    if local_rank == 0:
        utils.mkdir(result_dir)
        utils.mkdir(model_dir)

    ######### Model ###########
    if opt.model == "MIRNet":
        model = MIRNet()
    else:
        model = MIRNet_v2(n_feat=64)

    model.train()

    ######### Scheduler ###########

    scheduler = optim.lr.CosineAnnealingDecay(learning_rate=opt.lr, T_max=opt.epochs, eta_min=1e-6)

    optimizer = optim.Adam(parameters=model.parameters(), learning_rate=scheduler, weight_decay=1e-8)

    ######### Loss ###########
    criterion = CharbonnierLoss()

    ######### DataLoaders ###########
    img_options_train = {'patch_size': opt.patch_size}

    train_dataset = get_training_data(opt.data_dir, img_options_train)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False)

    val_dataset = get_validation_data(opt.val_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=False)

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer = paddle.distributed.fleet.distributed_optimizer(
            optimizer)  # The return is Fleet object
        ddp_model = paddle.distributed.fleet.distributed_model(model)

    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=opt.batchSize, shuffle=True, drop_last=False)

    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
        return_list=True
    )

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.epochs + 1))
    print('===> Loading datasets')

    with LogWriter(logdir=log_dir) as writer:
        step = 0
        best_psnr = 0
        best_epoch = 0
        best_iter = 0
        best_ssim = 0

        for epoch in range(start_epoch, opt.epochs + 1):
            epoch_start_time = time.time()
            epoch_loss = 0
            batch_loss = 0.

            eval_now = len(train_loader) // 4 - 1
            print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

            for i, data in enumerate(train_loader):
                target = data[0]
                input_ = data[1]

                # print(target.shape)
                # break

                if epoch > 5:
                    target, input_ = MixUp(target, input_)

                if nranks > 1:
                    restored = ddp_model(input_)
                else:
                    restored = model(input_)

                restored = paddle.clip(restored, 0, 1)

                loss = criterion(restored, target)

                batch_loss += loss.item() / 200.

                optimizer.clear_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # break

                if i % 200 == 0 and i > 0 and local_rank == 0:
                    # Log the scalar values
                    writer.add_scalar(tag='loss', value=batch_loss, step=step)

                    print("Epoch: {}\tBatch: {}/{}\tTime: {:.4f}\tLoss: {:.4f}".format(epoch, i, len(train_loader),
                                                                                       time.time() - epoch_start_time,
                                                                                       batch_loss))
                    batch_loss = 0.

                #### Evaluation ####
                if i % eval_now == 0 and i > 0 and local_rank == 0:
                    model.eval()
                    with paddle.no_grad():
                        psnr_val_rgb = []
                        ssim_val_rgb = []
                        for ii, data_val in enumerate(val_loader):
                            target = data_val[0]
                            input_ = data_val[1]

                            restored = model(input_)
                            restored = paddle.clip(restored, 0, 1)
                            psnr_val_rgb.append(utils.batch_PSNR(restored, target, 1.))
                            ssim_val_rgb.append(utils.batch_SSIM(restored, target))

                        psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)
                        ssim_val_rgb = sum(ssim_val_rgb) / len(ssim_val_rgb)

                        if psnr_val_rgb > best_psnr:
                            best_psnr = psnr_val_rgb
                            best_ssim = ssim_val_rgb
                            best_epoch = epoch
                            best_iter = i
                            paddle.save({'epoch': epoch,
                                         'state_dict': model.state_dict(),
                                         'optimizer': optimizer.state_dict()
                                         }, os.path.join(model_dir, "model_best.pdparams"))

                        print(
                            "[Ep %d it %d\t PSNR: %.4f\t SSIM: %.4f\t] ----  [best_Ep %d best_it %d Best_PSNR %.4f Best_SSIM %.4f] " % (
                                epoch, i, psnr_val_rgb, ssim_val_rgb, best_epoch, best_iter, best_psnr, best_ssim))

                    writer.add_scalar(tag='PSNR on validation data', value=psnr_val_rgb, step=step)

                    model.train()

            # update lr
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                lr_sche = optimizer.user_defined_optimizer._learning_rate
            else:
                lr_sche = optimizer._learning_rate
            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                lr_sche.step()

            print("------------------------------------------------------------------")
            print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch,
                                                                                      time.time() - epoch_start_time,
                                                                                      epoch_loss, scheduler.get_lr()))
            print("------------------------------------------------------------------")

            if local_rank == 0:
                paddle.save({'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }, os.path.join(model_dir, "model_latest.pdparams"))

                # paddle.save({'epoch': epoch,
                #              'state_dict': model.state_dict(),
                #              'optimizer': optimizer.state_dict()
                #              }, os.path.join(model_dir, f"model_epoch_{epoch}.pdparams"))


if __name__ == '__main__':
    main()

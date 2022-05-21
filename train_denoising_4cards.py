import os
from config import Config

opt = Config('training_4cards.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import paddle

import paddle.optimizer as optim
from paddle.io import DataLoader
import random
import time
import numpy as np

import utils
from dataloaders.data_rgb import get_training_loader, get_validation_data

from utils import MixUp

from networks.MIRNet_V2_model import MIRNet_v2
from losses import CharbonnierLoss
import paddle.distributed as dist

from visualdl import LogWriter


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
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)
    log_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'logs', session)

    if local_rank == 0:
        utils.mkdir(result_dir)
        utils.mkdir(model_dir)

    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    ######### Model ###########
    model = MIRNet_v2(n_feat=64)
    model.train()


    ######### Scheduler ###########
    new_lr = opt.OPTIM.LR_INITIAL

    warmup = True

    if warmup:
        warmup_epochs = 3
        scheduler_cosine = optim.lr.CosineAnnealingDecay(learning_rate=new_lr,
                                                         T_max=sum(opt.OPTIM.NUM_EPOCHS) - warmup_epochs, eta_min=1e-6)
        scheduler = optim.lr.LinearWarmup(scheduler_cosine, warmup_epochs, 0, new_lr)

    else:
        scheduler = optim.lr.CosineAnnealingDecay(learning_rate=new_lr, T_max=opt.OPTIM.NUM_EPOCHS, eta_min=1e-6)

    optimizer = optim.Adam(parameters=model.parameters(), learning_rate=scheduler, weight_decay=1e-8)

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer = paddle.distributed.fleet.distributed_optimizer(
            optimizer)  # The return is Fleet object
        ddp_model = paddle.distributed.fleet.distributed_model(model)


    ######### Resume ###########
    if opt.TRAINING.RESUME:
        # path_chk_rest = utils.get_last_path(model_dir, '_latest.pdparams')
        # utils.load_checkpoint(model, path_chk_rest)
        # start_epoch = utils.set_start_epoch(path_chk_rest) + 1
        # utils.load_optim(optimizer, path_chk_rest)

        ckpt = './pretrained_models/torch_init.pdparams'
        utils.load_checkpoint(model, ckpt)

        # start_epoch = utils.set_start_epoch(ckpt) + 1
        # utils.load_optim(optimizer, ckpt)

        # for i in range(1, start_epoch):
            # scheduler.step()
        # print(scheduler.get_lr())

        # new_lr = optimizer.get_lr()
        # print('------------------------------------------------------------------------------')
        # print("==> Resuming Training with learning rate:", new_lr)
        # print('------------------------------------------------------------------------------')


    ######### Loss ###########
    criterion = CharbonnierLoss()

    ######### DataLoaders ###########
    val_dataset = get_validation_data(val_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)

    Train_ps = opt.TRAINING.PATCH_SIZE
    Train_bs = opt.OPTIM.BATCH_SIZE
    Train_wk = opt.TRAINING.NUM_WORKERS

    pro_step = 0

    train_loader = get_training_loader(train_dir, Train_ps[pro_step], Train_bs[pro_step], Train_wk[pro_step])
    epoches = opt.OPTIM.NUM_EPOCHS

    up_epoch = epoches[pro_step]

    ######### Update Learning rate ###########
    if isinstance(optimizer, paddle.distributed.fleet.Fleet):
        lr_sche = optimizer.user_defined_optimizer._learning_rate
    else:
        lr_sche = optimizer._learning_rate
    if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
        lr_sche.step()


    new_lr = optimizer.get_lr()
    print('------------------------------------------------------------------------------')
    print("==> Start Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')


    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, sum(epoches) + 1))
    print('===> Loading datasets')

    with LogWriter(logdir=log_dir) as writer:
        step = 0
        best_psnr = 0
        best_epoch = 0
        best_iter = 0
        best_ssim = 0

        for epoch in range(start_epoch, sum(epoches) + 1):
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

            # update train_loader
            if epoch == up_epoch and epoch < sum(epoches):
                pro_step += 1
                up_epoch += epoches[pro_step]
                train_loader = get_training_loader(train_dir, Train_ps[pro_step], Train_bs[pro_step],
                                                   Train_wk[pro_step])

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

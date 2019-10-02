from comet_ml import Experiment

# import advex_uar
import numpy as np
import os
import sys
import torch

import torch.nn as nn
import torch.optim as optim

from dataset import load_dataset
from model import resnet20, resnet56
from options import Parser
from trainer import Trainer
from utils import report_epoch_status, Timer, get_attack


def main():
    opt = Parser().get()
    
    experiment = None
    if opt.comet:
        experiment = Experiment()
        experiment.set_name(opt.exp_name)
        experiment.log_parameters(opt.__dict__)
        experiment.add_tags(opt.add_tags)

    # dataset and data loader
    train_loader, val_loader, aval_loader = \
        load_dataset(opt.dataset, opt.batch_size,
                     opt.data_root, opt.noise,
                     opt.noise_std, 
                     opt.num_val_samples,
                     workers=4)

    # model
    if opt.arch == 'resnet20':
        model = resnet20()
    elif opt.arch == 'resnet56':
        model = resnet56()
    else:
        raise NotImplementedError

    # move model to device
    model.to(opt.device)
    if opt.gpu_ids:
        model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # attacker
    attacker = get_attack(opt.attack, opt.device, opt.num_steps,
                          opt.eps, opt.eps_iter, opt.scale_each)

    # optimizer
    if opt.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), opt.lr,
                               eps=1e-6, weight_decay=opt.wd)
    elif opt.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), opt.lr,
                              momentum=opt.momentum,
                              weight_decay=opt.wd)
    else:
        raise NotImplementedError

    # scheduler
    if opt.scheduler_steps:
        scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, opt.scheduler_steps,
                gamma=opt.scheduler_gamma)
    else:
        scheduler = None

    # timer
    timer = Timer(opt.num_epochs, 0)

    # trainer
    trainer = Trainer(opt, model, criterion, attacker, optimizer)
    
    # path to save model
    save_path = os.path.join('ckpt', opt.dataset, 'models', opt.exp_name + '.pth')

    # epoch iteration
    for epoch in range(1, opt.num_epochs+1):
        trainer.epoch = epoch
        if scheduler:
            scheduler.step(epoch - 1) # scheduler's epoch is 0-indexed.

        # training
        train_losses, train_acc1s, train_acc5s = \
                trainer.train(train_loader)

        # validation
        val_losses, val_acc1s, val_acc5s = \
                trainer.validate(val_loader)
        if opt.adv_val_freq != -1 and epoch % opt.adv_val_freq == 0:
            aval_losses, aval_acc1s, aval_acc5s = \
                trainer.adv_validate(aval_loader)
        else:
            aval_losses, aval_acc1s, aval_acc5s = \
                    dict(), dict(), dict()

        losses = dict(**train_losses, **val_losses, **aval_losses)
        acc1s = dict(**train_acc1s, **val_acc1s, **aval_acc1s)
        acc5s = dict(**train_acc5s, **val_acc5s, **aval_acc5s)
        report_epoch_status(losses, acc1s, acc5s, trainer.num_loss,
                            epoch, opt, timer, experiment)

        if epoch % opt.save_model_freq == 0:
            trainer.save_model(save_path)

    trainer.save_model(save_path)

if __name__ == '__main__':
    main()

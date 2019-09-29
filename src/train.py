from comet_ml import Experiment

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from advertorch.attacks import PGDAttack as LPPGDAttack
from advex-uar.attacks import PGDAttack as UARPGDAttack

import dataset import models from options import Parser
from utils import report_epoch_status, Timer
from trainer import Trainer
from models.initializer import init_he


def main():
    opt = Parser().get()
    
    experiment = None
    if opt.comet:
        experiment = Experiment()
        experiment.set_name(opt.exp_name)
        experiment.log_parameters(opt.__dict__)
        experiment.add_tags(opt.add_tags)

    # dataset and data loader
    if opt.uar:
        train_loader, val_loader, aval_loader = \
            dataset.uar.load_dataset(opt.dataset, opt.batch_size,
                                     opt.data_root, opt.noise,
                                     opt.noise_std, 
                                     opt.num_val_samples,
                                     workers=4)
    else:
        train_loader, val_loader, adv_val_loader, _, num_classes = \
            dataset.uar.load_dataset(opt.dataset, opt.batch_size,
                                     opt.data_root, opt.noise,
                                     opt.noise_std,
                                     opt.num_val_samples,
                                     workers=4)

    # model
    if opt.uar and opt.arch == 'resnet':
        model = models.uar.resnet56()
    if opt.arch == 'lenet':
        model = models.lp.LeNet(num_classes)
    elif opt.arch == 'resnet':
        model = models.lp.ResNetv2_20(num_classes)
    else:
        raise NotImplementedError

    # weight init
    if opt.weight_init == 'he':
        model.apply(init_he)

    # move model to device
    model.to(opt.device)
    if opt.gpu_ids:
        model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # advertorch attacker
    if opt.attack == 'pgd':
        attacker = LPPGDAttack(
            model, loss_fn = criterion, eps=opt.eps/255,
            nb_iter=opt.num_steps, eps_iter=opt.eps_iter/255,
            rand_init=True, clip_min=opt.clip_min, 
            clip_max=opt.clip_max, ord=np.inf, targeted=False)
    else:
        raise NotImplementedError

    # optimizer
    if opt.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), opt.lr,
                               eps=1e-6, weight_decay=opt.wd)
    elif opt.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), opt.lr,
                              weight_decay=opt.wd)
    else:
        raise NotImplementedError

    # scheduler
    if opt.scheduler_step:
        scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=opt.scheduler_step,
                gamma=opt.scheduler_gamma)
    else:
        scheduler = None

    # timer
    timer = Timer(opt.num_epochs, 0)

    # trainer
    trainer = Trainer(opt, model, criterion, attacker, optimizer)
    
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
                trainer.adv_validate(adv_val_loader)
        else:
            aval_losses, aval_acc1s, aval_acc5s = \
                    dict(), dict(), dict()

        losses = dict(**train_losses, **val_losses, **aval_losses)
        acc1s = dict(**train_acc1s, **val_acc1s, **aval_acc1s)
        acc5s = dict(**train_acc5s, **val_acc5s, **aval_acc5s)
        report_epoch_status(losses, acc1s, acc5s, trainer.num_loss,
                            epoch, opt, timer, experiment)

    save_path = os.path.join('ckpt', opt.dataset, 'models', opt.exp_name + '.pth')
    trainer.save_model(save_path)

if __name__ == '__main__':
    main()

import os
import sys
import numpy as np
import torch
import torch.nn as nn

from advertorch.attacks import PGDAttack
from options import Parser
from utils import convert_model_from_parallel
from dataset import load_dataset
from trainer import Trainer
from models import LeNet, ResNetv2_20


def main():
    opt = Parser(train=False).get()
    
    # dataset and data loader
    _, val_loader, adv_val_loader, _, num_classes = \
            load_dataset(opt.dataset, opt.batch_size, opt.data_root,
                         False, 0.0, opt.num_val_samples,
                         workers=4)

    # model
    if opt.arch == 'lenet':
        model = LeNet(num_classes)
    elif opt.arch == 'resnet':
        model = ResNetv2_20(num_classes)
    else:
        raise NotImplementedError

    # move model to device
    model.to(opt.device)

    # load trained weight
    try:
        model.load_state_dict(torch.load(opt.weight_path))
    except:
        model_weight = convert_model_from_parallel(opt.weight_path)
        model.load_state_dict(model_weight)
    
    # criterion
    criterion = nn.CrossEntropyLoss()

    # advertorch attacker
    if opt.attack == 'pgd':
        attacker = PGDAttack(
            model, loss_fn = criterion, eps=opt.eps/255,
            nb_iter=opt.num_steps, eps_iter=opt.eps_iter/255,
            rand_init=True, clip_min=opt.clip_min, 
            clip_max=opt.clip_max, ord=np.inf, targeted=False
            )
    else:
        raise NotImplementedError

    # trainer
    trainer = Trainer(opt, model, criterion, attacker)
    trainer.print_freq = -1

    # validation
    val_losses, val_acc1s, val_acc5s = \
        trainer.validate(val_loader)
    aval_losses, aval_acc1s, aval_acc5s = \
        trainer.adv_validate(adv_val_loader)

    print('[model] {}'.format(opt.weight_path))
    print('[standard]\n'
          'loss: {:.4f} | acc1: {:.2f}% | acc5: {:.2f}%'
          '\n[adversarial]\n'
          'loss: {:.4f} | acc1: {:.2f}% | acc5: {:.2f}%'.format(
              val_losses['val'].avg, val_acc1s['val'].avg,
              val_acc5s['val'].avg, aval_losses['aval'].avg,
              aval_acc1s['aval'].avg, aval_acc5s['aval'].avg
              ))


if __name__ == '__main__':
    main()

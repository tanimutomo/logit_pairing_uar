import datetime
import numpy as np
import os
import sys
import torch
import torch.nn as nn

from dataset import load_dataset
from model import resnet20, resnet56
from options import Parser
from trainer import Trainer
from utils import (report_epoch_status, Timer, convert_model_from_parallel,
                   get_attack)


def main():
    opt = Parser(train=False).get()

    # dataset and data loader
    train_loader, val_loader, aval_loader = \
        load_dataset(opt.dataset, opt.batch_size,
                     opt.data_root, False, 0.0, 
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

    # load trained weight
    try:
        model.load_state_dict(torch.load(opt.weight_path))
    except:
        model_weight = convert_model_from_parallel(opt.weight_path)
        model.load_state_dict(model_weight)
    
    # criterion
    criterion = nn.CrossEntropyLoss()

    # attacker
    attacker = get_attack(opt.attack, opt.device, opt.num_steps,
                          opt.eps, opt.eps_iter, opt.scale_each)

    # trainer
    trainer = Trainer(opt, model, criterion, attacker)
    trainer.print_freq = -1

    # validation
    val_losses, val_acc1s, val_acc5s = \
        trainer.validate(val_loader)
    aval_losses, aval_acc1s, aval_acc5s = \
        trainer.adv_validate(aval_loader)

    finished_time = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    report = '[time] {}\n' \
             '[model] {}\n' \
             '[attack] method: {} | eps: {:.2f} | num_steps: {}\n' \
             '[standard]\n' \
             '  loss: {:.4f} | acc1: {:.2f}% | acc5: {:.2f}%\n' \
             '[adversarial]\n' \
             '  loss: {:.4f} | acc1: {:.2f}% | acc5: {:.2f}%\n\n'.format(
                     finished_time, opt.weight_path,
                     opt.attack, opt.eps, opt.num_steps, 
                     val_losses['val'].avg, val_acc1s['val'].avg,
                     val_acc5s['val'].avg, aval_losses['aval'].avg,
                     aval_acc1s['aval'].avg, aval_acc5s['aval'].avg)

    print(report)
    exp_name = opt.weight_path.split('/')[-1][:-4]
    report_path = os.path.join('ckpt', opt.dataset, 'reports', exp_name + '.txt')
    with open(report_path, mode='a') as f:
        f.write(report)


if __name__ == '__main__':
    main()

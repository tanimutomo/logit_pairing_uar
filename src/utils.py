import argparse
import sys
import time
import datetime
import torch
from collections import OrderedDict


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1) # top-k index: size (B, k)
        pred = pred.t() # size (k, B)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        acc = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            acc.append(correct_k * 100.0 / batch_size)
        return acc


def convert_model_from_parallel(weight_path):
    weight = torch.load(weight_path)
    new_weight = OrderedDict()
    for name, w in weight.items():
        new_weight[name[7:]] = w
    return new_weight


def report_epoch_status(losses, acc1s, acc5s, num_loss,
                        epoch, opt, timer, experiment):
    opt.total = True
    opt.val = True
    opt.aval = opt.adv_val_freq != -1 and epoch % opt.adv_val_freq == 0
    log = '\r\033[{}A\033[J'.format(num_loss+2) \
          + 'epoch [{:d}/{:d}]'.format(epoch, opt.num_epochs)

    # loss
    log += '\n[loss] '
    for name in ['ct', 'at', 'alp', 'clp', 'lsq',
                 'total', 'val', 'aval']:
        if getattr(opt, name):
            log += '{} {:.4f} / '.format(name, losses[name].avg)
            if experiment:
                experiment.log_metric(name + '-loss',
                                      losses[name].avg,
                                      step=epoch)

    # acc1 log
    log += '\n[acc1] '
    for name in ['ct', 'at', 'val', 'aval']:
        if getattr(opt, name):
            log += '{} {:.2f}% / '.format(name, acc1s[name].avg)
            if experiment:
                experiment.log_metric(name + '-acc1',
                                      acc1s[name].avg,
                                      step=epoch)

    # acc5 log
    log += '\n[acc5] '
    for name in ['ct', 'at', 'val', 'aval']:
        if getattr(opt, name):
            log += '{} {:.2f}% / '.format(name, acc5s[name].avg)

    timer.step()

    # time log
    log += '\n[time] elapsed: {} / '.format(timer.get_elapsed_time()) \
                 + 'estimated: {}\n'.format(timer.get_estimated_time())

    sys.stdout.write(log)
    sys.stdout.flush()


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.cnt = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt


class Timer():
	def __init__(self, num_steps, start_step=0):
		self.num_steps = num_steps
		self.start_step = start_step
		self.current_step = start_step
		self.start_time = time.time()
		self.elapsed_time = time.time()

	def step(self):
		self.current_step += 1

	def set_current_step(self, step):
		self.current_step = step

	def get_elapsed_time(self):
		self.elapsed_time = time.time() - self.start_time
		return str(datetime.timedelta(seconds=int(self.elapsed_time)))

	def get_estimated_time(self):
		self.elapsed_time = time.time() - self.start_time
		remaining_step = self.num_steps - self.current_step

		if self.current_step == self.start_step:
			return str(datatime.timedelta(seconds=int(0)))
		estimated_time = self.elapsed_time * remaining_step / (self.current_step - self.start_step)
		return str(datetime.timedelta(seconds=int(estimated_time)))


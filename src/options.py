import argparse
import random
import torch

class Parser():
    default_options = dict(
        mnist=dict(batch_size=200,
                   num_epochs=500,
                   arch='lenet',
                   eps=76.5,
                   eps_iter=2.55,
                   num_steps=40,
                   num_restarts=1,
                   clip_min=0.0,
                   clip_max=1.0,
                   noise_std=0.5),

        cifar10=dict(batch_size=128,
                     num_epochs=100,
                     arch='resnet',
                     eps=16.0,
                     eps_iter=2.0,
                     num_steps=10,
                     num_restarts=1,
                     clip_min=0.0,
                     clip_max=1.0,
                     noise_std=0.06)
        )

    def __init__(self, train=True):
        self.train = train

        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        # device
        parser.add_argument('--gpu_ids', type=int, nargs='*', default=[], help='id of gpu')

        # data
        parser.add_argument('--data_root', type=str, default='~/data', help='root of dataset')
        parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10'],
                            help='dataset name')
        parser.add_argument('--batch_size', type=int, help='batch size')
        parser.add_argument('--num_val_samples', type=int, default=1000, help='number of samples for validation')

        # architecture
        parser.add_argument('--arch', type=str, choices=['lenet', 'resnet'], help='architecture name')

        # attack
        parser.add_argument('--attack', type=str, default='pgd', help='name of adversarial attack method')
        parser.add_argument('--eps', type=float, help='epsilon for lp-norm attack')
        parser.add_argument('--eps_iter', type=float, help='epsilon for each attack step')
        parser.add_argument('--num_steps', type=int, help='number of steps for attack')
        parser.add_argument('--num_restarts', type=int, help='number of restats for attack')
        parser.add_argument('--clip_min', type=float, help='minimum value for cliping AEs')
        parser.add_argument('--clip_max', type=float, help='miximum value for cliping AEs')

        if train:
            # loss
            parser.add_argument('--ct', type=float, default=0.0, help='coef for clean example training')
            parser.add_argument('--at', type=float, default=0.0, help='coef of adversarial example training')
            parser.add_argument('--alp', type=float, default=0.0, help='coef of adversarial logit pairing')
            parser.add_argument('--clp', type=float, default=0.0, help='coef of clean logit pairing')
            parser.add_argument('--lsq', type=float, default=0.0, help='coef of logit squeezing')
            parser.add_argument('--lsq_grad', action='store_true', help='use gradual coef for logit squeezing')
            parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

            # noise
            parser.add_argument('--noise', action='store_true', help='gaussian noise')
            parser.add_argument('--noise_std', type=float, help='std of gaussian')

            # optimization
            parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'], help='name of optimization method')
            parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
            parser.add_argument('--scheduler_step', type=int, default=0, help='step for lr-scheduler')
            parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='gamma for lr-scheduler')
            parser.add_argument('--num_epochs', type=int, help='number of epochs')

            # weight initialization
            parser.add_argument('--weight_init', type=str, default='he', help='method of weight initialization')

            # others
            parser.add_argument('--comet', action='store_true', help='use comet for training log')
            parser.add_argument('--print_freq', type=int, default=10, help='frequency of printing logs')
            parser.add_argument('--adv_val_freq', type=int, default=1, help='frequency of adversarial validation')
            parser.add_argument('--add_names', type=str, nargs='*', default=[], help='additional experiment name')
            parser.add_argument('--add_tags', type=str, nargs='*', default=[], help='additinal tags for comet')

            # debug
            parser.add_argument('--report_itr_loss', type=str, nargs='*', default=[], 
                                choices=['ct', 'at', 'alp', 'clp', 'lsq'],
                                help='report loss each iteration in the training phase')

        else:
            # path to trained weight
            parser.add_argument('--weight_path', type=str, required=True, help='path to trained weight')

        self.opt = parser.parse_args()

    def get(self):
        # Set None options in default_options
        for name, value in self.default_options[self.opt.dataset].items():
            if hasattr(self.opt, name) and not getattr(self.opt, name):
                setattr(self.opt, name, value)


        # Add tags and set experiment name
        if self.train:
            base_tags = ['{}e'.format(self.opt.num_epochs),
                         self.opt.dataset]
            for name in ['ct', 'at', 'alp', 'clp', 'lsq']:
                if getattr(self.opt, name):
                    base_tags.append(name)
                    self.opt.add_names.append(name)
            self.opt.add_tags.extend(base_tags)

            self.opt.add_names.append(str(random.randint(100, 999)))
            self.opt.exp_name = '_'.join(self.opt.add_names)

        # set device
        if torch.cuda.is_available():
            if not self.opt.gpu_ids:
                self.opt.device = torch.device('cuda:0')
                self.opt.gpu_ids = [i for i in range(torch.cuda.device_count())]
            elif len(self.opt.gpu_ids) > 1:
                self.opt.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0]))
            else:
                self.opt.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0]))
                self.opt.gpu_ids = []
            self.opt.cuda = True
        else:
            self.opt.device = torch.device('cpu')
            self.opt.gpu_ids = []
            self.opt.cuda = False

        return self.opt

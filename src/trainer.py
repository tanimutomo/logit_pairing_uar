import sys
import torch
import torch.nn.functional as F
from advertorch.context import ctx_noparamgrad_and_eval

from utils import AverageMeter, accuracy


class Trainer():
    """Trainer class
    Args:
        
    """
    def __init__(self, opt, model, criterion, attacker, optimizer=None):
        self.model = model
        self.criterion = criterion
        self.attacker = attacker
        self.optimizer = optimizer
        self.epoch = 0
        for name, value in opt.__dict__.items():
            setattr(self, name, value)

    def set_train_meters(self):
        # set number of losses
        self.num_loss = 0

        # set loss meters
        self.loss_meters = dict(total=AverageMeter())
        for name in ['ct', 'at', 'alp', 'clp', 'lsq']:
            if getattr(self, name):
                self.loss_meters[name] = AverageMeter()
                # count number of losses for stdout
                self.num_loss += 1

        # set acc meters
        self.acc1_meters = dict()
        self.acc5_meters = dict()
        if self.ct:
            self.acc1_meters['ct'] = AverageMeter()
            self.acc5_meters['ct'] = AverageMeter()
        if self.ct:
            self.acc1_meters['at'] = AverageMeter()
            self.acc5_meters['at'] = AverageMeter()

        self.report_itr_loss.append('total')

    def set_val_meters(self, val_type):
        self.loss_meters = {val_type: AverageMeter()}
        self.acc1_meters = {val_type: AverageMeter()}
        self.acc5_meters = {val_type: AverageMeter()}
        self.num_loss = 1 

    def update_log_meters(self, name, size, loss,
                          acc1=None, acc5=None):
        # loss
        self.loss_meters[name].update(loss, size)
        self.log += '[{}] loss {:.4f}, '.format(name, loss)
        # if name in self.report_itr_loss:
        #     self.experiment.log_metric('{}-loss'.format(name),
        #                                loss)

        # acc1
        if acc1 is not None:
            self.acc1_meters[name].update(acc1, size)
            self.log += 'acc1 {:.2f}%, '.format(acc1)

        # acc5
        if acc5 is not None:
            self.acc5_meters[name].update(acc5, size)
            self.log += 'acc5 {:.2f}%, '.format(acc5)

        self.log += '\n'

    def train(self, loader):
        # initialize all loss values
        ct_loss, at_loss, alp_loss, clp_loss, lsq_loss = \
            0.0, 0.0, 0.0, 0.0, 0.0
        # initialize log and meters
        self.set_train_meters()
        self.model.train()

        print("\n" * (self.num_loss + 1))
        for itr, (x, t) in enumerate(loader):
            # log for printing a training status
            self.log = '\r\033[{}A\033[J'.format(self.num_loss+2) \
                       + '[train mode] ' \
                       + 'itr [{:d}/{:d}]\n'.format(itr, len(loader))

            x = x.to(self.device, non_blocking=self.cuda)
            t = t.to(self.device, non_blocking=self.cuda)
            y = self.model(x)

            # create adversarial examples
            if self.at or self.alp:
                with ctx_noparamgrad_and_eval(self.model):
                    perturbed_x = self.attacker.perturb(x, t)
                perturbed_y = self.model(perturbed_x)

            # clean examples training
            if self.ct:
                ct_loss = self.criterion(y, t)
                ct_acc1, ct_acc5 = accuracy(y, t, topk=(1,5))
                self.update_log_meters('ct', x.size(0), ct_loss.item(),
                                       ct_acc1.item(), ct_acc5.item())

            # adversarial examples training
            if self.at:
                at_loss = self.criterion(perturbed_y, t)
                at_acc1, at_acc5 = accuracy(perturbed_y, t, topk=(1,5))
                self.update_log_meters('at', x.size(0), at_loss.item(),
                                       at_acc1.item(), at_acc5.item())

            # adversarial logit pairing
            if self.alp:
                alp_loss = F.mse_loss(y, perturbed_y)
                self.update_log_meters('alp', x.size(0), alp_loss.item())

            # clean logit pairing
            if self.clp:
                clp_loss = F.mse_loss(y[:y.shape[0] // 2], y[y.shape[0] // 2:])
                self.update_log_meters('clp', x.size(0), clp_loss.item())

            # clean logit squeezing
            if self.lsq:
                lsq_loss = torch.norm(y, p=2, dim=1).mean()
                self.update_log_meters('lsq', x.size(0), lsq_loss.item())

            # sum all losses
            loss = (self.ct * ct_loss) + (self.at * at_loss) \
                 + (self.alp * alp_loss) + (self.clp * clp_loss)
            if self.lsq_grad:
                loss += (min(0.1 * self.epoch, self.lsq) * lsq_loss)
            else:
                loss += self.lsq * lsq_loss
            self.update_log_meters('total', x.size(0), loss.item())

            # update model weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # report training status
            if itr % self.print_freq == 0:
                sys.stdout.write(self.log)

        print('\r\033[{}A\033[J'.format(self.num_loss+2))
        return self.loss_meters, self.acc1_meters, self.acc5_meters

    def validate(self, loader):
        self.set_val_meters('val')
        self.model.eval()

        print("\n" * (self.num_loss + 1))
        with torch.no_grad():
            for itr, (x, t) in enumerate(loader):
                # log for printing a training status
                self.log = '\r\033[{}A\033[J'.format(self.num_loss+1) \
                           + '[val mode] ' \
                           + 'itr [{:d}/{:d}]\n'.format(itr, len(loader))

                x = x.to(self.device, non_blocking=self.cuda)
                t = t.to(self.device, non_blocking=self.cuda)

                # calcurate clean loss and accuracy
                y = self.model(x)
                val_loss = self.criterion(y, t)
                val_acc1, val_acc5 = accuracy(y, t, topk=(1,5))
                self.update_log_meters('val', x.size(0), val_loss.item(),
                                       val_acc1.item(), val_acc5.item())

                if itr % self.print_freq == 0:
                    sys.stdout.write(self.log)

        print('\r\033[{}A\033[J'.format(self.num_loss+1))
        return self.loss_meters, self.acc1_meters, self.acc5_meters

    def adv_validate(self, loader):
        self.set_val_meters('aval')
        self.model.eval()

        print("\n" * (self.num_loss + 1))
        for itr, (x, t) in enumerate(loader):
            # log for printing a training status
            self.log = '\r\033[{}A\033[J'.format(self.num_loss+1) \
                       + '[adv val mode] ' \
                       + 'itr [{:d}/{:d}]\n'.format(itr, len(loader))

            x = x.to(self.device, non_blocking=self.cuda)
            t = t.to(self.device, non_blocking=self.cuda)

            # attack
            perturbed_x = self.attacker.perturb(x, t)

            # calcurate adversarial loss and accuracy
            perturbed_y = self.model(perturbed_x)
            aval_loss = self.criterion(perturbed_y, t)
            aval_acc1, aval_acc5 = accuracy(perturbed_y, t, topk=(1, 5))
            self.update_log_meters('aval', x.size(0), aval_loss.item(),
                                   aval_acc1.item(), aval_acc5.item())

            if itr % self.print_freq == 0:
                sys.stdout.write(self.log)

        print('\r\033[{}A\033[J'.format(self.num_loss+1))
        return self.loss_meters, self.acc1_meters, self.acc5_meters

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


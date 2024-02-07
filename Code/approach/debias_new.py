import torch
from argparse import ArgumentParser
from torch_geometric.data import DataLoader
from data_process.Exemplar_dataset import ExemplarsSet
import torch.nn.functional as F
from copy import deepcopy
from models.sparsemax import Sparsemax
import numpy as np
from sklearn.metrics import f1_score
import time


class CIL_debias_new:
    """Class implementing the finetuning baseline"""

    def __init__(self, model, init_model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=10,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=0.1, fix_bn=False,
                 eval_on_train=False,
                 logger=None, exemplars=None, all_outputs=False, T=2, loss_normalize='sparsemax'):
        self.model = model
        self.init_model = init_model
        self.model_c = self.model.model.model_c
        self.model_b = self.model.model.model_b
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars = exemplars
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = lr * wu_lr_factor
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.optimizer_c = None
        self.optimizer_b = None
        self.model_old = None
        self.all_out = all_outputs
        self.T = T
        self.loss_normalize = loss_normalize
        self.bias_scaler = 1
        print('bias scaler', self.bias_scaler)

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsSet

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all  outputs to be modified (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars.exemplars_dataset) == 0 and len(self.model.heads_c) > 1 and not self.all_out:
            # if there are no exemplars, previous heads are not modified
            params_c = list(self.model_c.parameters()) + list(self.model.heads_c[-1].parameters())
            params_b = list(self.model_b.parameters()) + list(self.model.heads_b[-1].parameters())
        else:
            params_c = list(self.model_c.parameters()) + list(self.model.heads_c.parameters())
            params_b = list(self.model_b.parameters()) + list(self.model.heads_b.parameters())

        optimizer_c = torch.optim.Adam(params_c, self.lr, weight_decay=self.wd)
        optimizer_b = torch.optim.Adam(params_b, self.lr, weight_decay=self.wd)
        return optimizer_c, optimizer_b

    def train(self, t, trn_loader, val_loader):
        """Main train structure"""
        self.pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, val_loader)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""

        # Warm-up phase
        if self.warmup_epochs and t > 0:
            self.optimizer = torch.optim.Adam(self.model.heads_c[-1].parameters(), lr=self.warmup_lr)

            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmupclock0 = time.time()
                self.model.heads_c[-1].train()
                for idx, data in enumerate(trn_loader):
                    outputs, _ = self.model(data.to(self.device))
                    loss = self.warmup_loss(outputs[t], data.y - self.model.task_offset[t])
                    self.optimizer.zero_grad()
                    loss.backward()
                    #  torch.nn.utils.clip_grad_norm_(self.model.heads[-1].parameters(), self.clipgrad)
                    self.optimizer.step()
                warmupclock1 = time.time()
                with torch.no_grad():
                    total_loss, total_acc_taw = 0, 0
                    self.model.eval()
                    for idx, data in enumerate(trn_loader):
                        outputs, _ = self.model(data.to(self.device))
                        loss = self.warmup_loss(outputs[t], data.y - self.model.task_offset[t])
                        pred = torch.zeros_like(data.y)
                        for m in range(len(pred)):
                            cus = self.model.task_cls.cumsum(0)
                            this_task = (cus.to(self.device) <= data.y[m]).sum()
                            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
                        hits_taw = (pred == data.y).float()
                        total_loss += loss.item() * len(data.y)
                        total_acc_taw += hits_taw.sum().item()
                total_num = len(trn_loader.dataset)
                trn_loss, trn_acc = total_loss / total_num, total_acc_taw / total_num
                warmupclock2 = time.time()
                print('| Warm-up Epoch {:3d}, time={:5.1f}s/{:.6f}s | Train: loss={:.6f}, TAw acc={:.6f}% |'.format(
                    e + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, trn_loss, 100 * trn_acc))
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=trn_loss, group="warmup")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * trn_acc, group="warmup")

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        lr = self.lr
        best_loss = np.inf
        best_acc = 0
        patience = self.lr_patience

        self.optimizer_c, self.optimizer_b = self._get_optimizer()

        # add exemplars to train_loader
        if len(self.exemplars.exemplars_dataset) > 0 and t > 0:
            trn_loader = DataLoader(trn_loader.dataset + self.exemplars.exemplars_dataset,
                                    batch_size=trn_loader.batch_size,
                                    shuffle=True)

        for e in range(self.nepochs):
            # Train

            clock0 = time.time()
            best_model = self.model.get_copy()
            self.train_epoch(t, trn_loader)
            clock1 = time.time()
            train_loss, train_acc, train_tag, train_macro_taw, train_macro_tag, train_acc_b, _, _ = self.eval(t,
                                                                                                              trn_loader)

            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | '
                  'Train: loss={:.3f}, TAw acc={:5.1f}%, TAg acc={:5.1f}%, '
                  'TAw macro={:5.1f}%, TAg macro={:5.1f}%, TAw_bb acc={:5.1f}% |'.format(
                e + 1, clock1 - clock0, clock2 - clock1,
                train_loss, 100 * train_acc, 100 * train_tag,
                100 * train_macro_taw, 100 * train_macro_tag, 100 * train_acc_b), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="train acc", value=100 * train_acc, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="train tag", value=100 * train_tag, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="train taw macro", value=100 * train_macro_taw,
                                   group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="train tag macro", value=100 * train_macro_tag,
                                   group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="train acc b", value=100 * train_acc_b, group="train")

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, valid_tag, valid_macro_taw, valid_macro_tag, valid_acc_b, _, _ = self.eval(t, val_loader)


            clock4 = time.time()
            print(
                ' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}%, TAg acc={:5.1f}%, TAw macro={:5.1f}%, TAg macro={:5.1f}%, TAw_bb acc={:5.1f}%|'.format(
                    clock4 - clock3, valid_loss, 100 * valid_acc, 100 * valid_tag, 100 * valid_macro_taw,
                    100 * valid_macro_tag, 100 * valid_acc_b), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="valid taw acc", value=100 * valid_acc, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="valid tag acc", value=100 * valid_tag, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="valid taw macro", value=100 * valid_macro_taw,
                                   group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="valid tag macro", value=100 * valid_macro_tag,
                                   group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="valid acc b", value=100 * valid_acc_b, group="valid")

            if torch.isnan(torch.tensor(train_loss)).sum():
                break
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                print(' *', end='')
            
            if valid_acc > best_acc:
                best_acc = valid_acc
                print('+', end='')
                

            patience -= 1
            if patience <= 0:
                # if it runs out of patience, reduce the learning rate
                lr /= self.lr_factor
                print(' lr={:.1e}'.format(lr), end='')
                patience = self.lr_patience
                if lr < self.lr_min:
                    break
                self.optimizer_c.param_groups[0]['lr'] = lr
                self.optimizer_b.param_groups[0]['lr'] = lr

            best_model = self.model.get_copy()
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()

        self.model.set_state_dict(best_model)
        self.model_old = deepcopy(self.model)

        model_b_par = self.init_model.state_dict()
        model_b_keys = ['model.model_b.masker.conv1.weight',
                        'model.model_b.masker.conv1.bias',
                        'model.model_b.masker.node_score_layer.weight',
                        'model.model_b.masker.node_score_layer.bias',
                        'model.model_b.masker.edge_score_layer.weight',
                        'model.model_b.masker.edge_score_layer.bias',
                        'model.model_b.conv1.weight',
                        'model.model_b.conv1.bias',
                        'model.model_b.conv2.weight',
                        'model.model_b.conv2.bias',
                        'model.model_b.fc1.weight',
                        'model.model_b.fc1.bias',
                        'model.model_b.fc2.weight',
                        'model.model_b.fc2.bias',
                        'heads_b.0.weight',
                        'heads_b.0.bias',
                        'heads_b.1.weight',
                        'heads_b.1.bias',
                        'heads_b.2.weight',
                        'heads_b.2.bias',
                        'heads_b.3.weight',
                        'heads_b.3.bias',
                        'heads_b.4.weight',
                        'heads_b.4.bias',
                        'heads_b.5.weight',
                        'heads_b.5.bias',
                        'heads_b.6.weight',
                        'heads_b.6.bias',
                        'heads_b.7.weight',
                        'heads_b.7.bias',
                        'heads_b.8.weight',
                        'heads_b.8.bias',
                        'heads_b.9.weight',
                        'heads_b.9.bias',
                        'heads_b.10.weight',
                        'heads_b.10.bias']

        state_dict = {k: v for k, v in model_b_par.items() if k in model_b_keys}

        model_dict = self.model.state_dict()
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars.collect_exemplars(self.model, trn_loader, self.exemplars.max_num_exemplars_per_class)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model_c.train()
        self.model_b.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for idx, data in enumerate(trn_loader):
            data = data.to(self.device)
            targets_old_c = None
            targets_old_b = None
            if t > 0:
                targets_old_c, targets_old_b = self.model_old(data)
            # Forward current model
            x_c, x_b = self.model(data)
            loss = self.criterion(t, x_c, x_b, data.y, targets_old_c, targets_old_b)

            if torch.isnan(loss.detach()).sum():
                break
            else:
                # Backward
                self.optimizer_c.zero_grad()
                self.optimizer_b.zero_grad()
                loss.backward()
                #            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.optimizer_c.step()
                self.optimizer_b.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            total_acc_taw_b, total_acc_tag_b = 0, 0
            self.model.eval()

            pred_debias_tag = torch.tensor([0])
            pred_debias_taw = torch.tensor([0])
            outputs = []
            yy = torch.tensor([0])
            for idx, data in enumerate(val_loader):
                data = data.to(self.device)
                targets_old_c = None
                targets_old_b = None
                if t > 0:
                    targets_old_c, targets_old_b = self.model_old(data)
                # Forward current model
                x_c, x_b = self.model(data)
                loss = self.criterion(t, x_c, x_b, data.y, targets_old_c, targets_old_b)
                hits_taw, hits_tag, pred_taw, pred_tag = self.calculate_metrics(t, x_c, data.y)
                hits_taw_b, hits_tag_b, pred_taw_b, pred_tag_b = self.calculate_metrics(t, x_b, data.y)

                pred_debias_tag = torch.cat([pred_debias_tag, pred_tag.cpu()])
                pred_debias_taw = torch.cat([pred_debias_taw, pred_taw.cpu()])
                yy = torch.cat([yy, data.y.cpu()])

                outputs.append(x_c)
                # Log
                total_loss += loss.item() * len(data.y)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_acc_taw_b += hits_taw_b.sum().item()
                total_acc_tag_b += hits_tag_b.sum().item()
                total_num += len(data.y)

            pred_debias_tag = pred_debias_tag[1::]
            pred_debias_taw = pred_debias_taw[1::]
            yy = yy[1::]

            macro_debias_tag = f1_score(pred_debias_tag, yy, average='macro')
            macro_debias_taw = f1_score(pred_debias_taw, yy, average='macro')

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, macro_debias_taw, macro_debias_tag, total_acc_tag_b / total_num, outputs, yy

    def calculate_metrics(self, t, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""

        pred_taw = torch.zeros_like(targets.to(self.device))
        pred_tag = torch.zeros_like(targets.to(self.device))
        # print(len(pred))
        # Task-Aware Multi-Head
        for m in range(len(pred_taw)):
            cumtaskcls = self.model.task_cls.cumsum(0)
            # print('targets[m]', targets[m])
            this_task = (cumtaskcls.to(self.device) <= targets[m]).sum()
            pred_taw[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred_taw == targets.to(self.device)).float()

        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred_tag = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred_tag = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred_tag == targets.to(self.device)).float()

        return hits_taw, hits_tag, pred_taw, pred_tag

    def debias_logit(self, t, x_c, x_b, targets):
        output_c = torch.cat(x_c, dim=1)
        output_b = torch.cat(x_b, dim=1)
        #        print('output_c', output_c)
        output_b = output_b / self.bias_scaler

        sparse_attention = Sparsemax()
        loss_future = F.nll_loss(F.log_softmax(output_b, dim=1), targets, reduction='none')
        loss_now = F.nll_loss(F.log_softmax(output_c, dim=1), targets, reduction='none')

        if self.loss_normalize == 'softmax':
            weight_future = torch.nn.functional.softmax(loss_future.detach(), dim=-1)
            weight_now = torch.nn.functional.softmax(loss_now.detach(), dim=-1)
        elif self.loss_normalize == 'sparsemax':
            weight_future = sparse_attention(loss_future.detach())
            weight_now = sparse_attention(loss_now.detach())

        weight = weight_future * weight_now
        weight = torch.nn.functional.softmax(weight, dim=-1)

        ind = torch.isnan(loss_now)
        weight[ind] = 0
        output_c = output_c / (torch.exp(weight)).view(-1, 1)

        return output_c, output_b

    def disl_cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, x_c, x_b, targets, targets_old_c, targets_old_b):
        """Returns the loss value"""

        output_c, output_b = self.debias_logit(t, x_c, x_b, targets)

        loss_c = F.nll_loss(F.log_softmax(output_c, dim=1), targets)
        loss_b = F.nll_loss(F.log_softmax(output_b, dim=1), targets)

        return loss_c + loss_b





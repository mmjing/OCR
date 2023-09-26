import torch
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.data.transforms import build_transform
from dassl.data import DataManager
import torch.nn as nn
import torch.nn as nn

@TRAINER_REGISTRY.register()
class Demix(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.DEMIX.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        self.dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes
        self.lab2cname = self.dm.lab2cname


    def Entropy(self, input_):
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy 

    # decrease from warm_up_factor to 0
    def get_annealing_down_params(self,warm_up_factor,current_epoch, end_epoch):
        gamma=10
        power=0.75
        decay = (1 + gamma * current_epoch / end_epoch) ** (-power)
        param_factor = warm_up_factor * decay
        return param_factor


    def Demix_Loss(self, feat1, feat2, netC, lam):
        epsilon = 1e-3
        nume = feat1 - feat2 * lam
        denomi = 1 - lam

        if denomi < epsilon:
            denomi += epsilon
        if 1 - denomi < epsilon:
            denomi -= epsilon 

        feat_desc = nume/denomi
        output_desc = netC(feat_desc)
        softmax_desc = nn.Softmax(dim=1)(output_desc)
        entropy_desc = torch.mean(self.Entropy(softmax_desc))
        return entropy_desc      

    def forward_backward(self, batch):
        input, input_aug, label = self.parse_batch_train(batch)        
        lam = self.cfg.TRAINER.DEMIX.LAM - self.get_annealing_down_params(self.cfg.TRAINER.DEMIX.LAM,self.epoch, self.max_epoch)

        output, feat_orig = self.model(input,return_feature=True)
        _,feat_aug = self.model(input_aug,return_feature=True)

        demix_loss = self.Demix_Loss(feat_aug,feat_orig,self.model.classifier,lam)
        demix_loss *= self.cfg.TRAINER.DEMIX.PAR_DESC

        loss = F.cross_entropy(output, label)
        loss -= demix_loss

        self.model_backward_and_update(loss)

        loss_summary = {
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        input_aug = batch['img2']
        label = batch['label']
        input = input.to(self.device)
        input_aug = input_aug.to(self.device)
        label = label.to(self.device)
        return input, input_aug, label

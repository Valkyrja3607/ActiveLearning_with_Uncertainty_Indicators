import torch
import torch.nn as nn
import numpy as np


class AdversarySampler:
    def __init__(self, budget, args):
        self.budget = budget
        self.args = args

    def OUI(self, v):
        v = nn.functional.softmax(v, dim=1)
        v_max, idx = torch.max(v, 1)
        bc = v.size()[0]
        c = v.size()[1]
        v_ = (1 - v_max) / (c - 1)  # [128]
        v__ = [[] for i in range(bc)]
        v__ = [[j.item()] * (c - 1) + [v_max[i]] for i, j in enumerate(v_)]
        v_ = torch.tensor(v__)  # [128,10]
        var_v = torch.var(v, 1)
        min_var = torch.var(v_, 1)
        if self.args.cuda:
            var_v = var_v.cuda()
            min_var = min_var.cuda()
        indicator = 1 - min_var / var_v * v_max  # [128]
        return indicator.detach()

    def sample(self, task_model, data, cuda):
        all_preds = []
        all_indices = []

        for images, _, indices in data:
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                preds = task_model(images)
                preds = self.OUI(preds)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        # all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices

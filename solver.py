import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score

import sampler
import copy


class Solver:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader
        self.ce_loss = nn.CrossEntropyLoss()

        self.sampler = sampler.AdversarySampler(self.args.budget, self.args)


    def train(self, querry_dataloader, val_dataloader, task_model, unlabeled_dataloader):
        lr_change = self.args.train_epochs // 4

        optim_task_model = optim.SGD(task_model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)

        task_model.train()

        if self.args.cuda:
            task_model = task_model.cuda()

        best_acc = 0
        for epoch in range(self.args.train_epochs):
            if epoch != 0 and epoch % lr_change ==0:
                for param in optim_task_model.param_groups:
                    param["lr"] = param["lr"] / 10

            for labeled_imgs, labels, _ in querry_dataloader:
                if self.args.cuda:
                    labeled_imgs = labeled_imgs.cuda()
                    labels = labels.cuda()
                preds = task_model(labeled_imgs)
                task_loss = self.ce_loss(preds, labels)
                optim_task_model.zero_grad()
                task_loss.backward()
                optim_task_model.step()

            print("Current training epoch: {}".format(epoch))
            print("Current task model loss: {:.4f}".format(task_loss.item()))
            acc = self.validate(task_model, val_dataloader)
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(task_model)
            print("acc: {}".format(acc))
            print("best acc: ", best_acc)

        if self.args.cuda:
            best_model = best_model.cuda()

        final_accuracy = self.test(best_model)
        return final_accuracy, task_model

    def sample_for_labeling(self, task_model, unlabeled_dataloader):
        querry_indices = self.sampler.sample(
            task_model,
            unlabeled_dataloader,
            self.args.cuda,
        )

        return querry_indices

    def validate(self, task_model, loader):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels, _ in loader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100

    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100



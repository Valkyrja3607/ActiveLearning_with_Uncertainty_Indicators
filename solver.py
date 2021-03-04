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
    
    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

    def train(self, querry_dataloader, val_dataloader, task_model, unlabeled_dataloader):
        self.args.train_iterations = (self.args.num_images * self.args.train_epochs) // self.args.batch_size
        lr_change = self.args.train_iterations // 4

        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        optim_task_model = optim.SGD(task_model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)

        task_model.train()

        if self.args.cuda:
            task_model = task_model.cuda()

        best_acc = 0
        for iter_count in range(self.args.train_iterations):
            if iter_count!=0 and iter_count % lr_change == 0:
                for param in optim_task_model.param_groups:
                    param["lr"] = param["lr"] / 10

            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            # task_model step
            preds = task_model(labeled_imgs)
            task_loss = self.ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()

            if iter_count % 1000 == 0:
                print("Current training iteration: {}".format(iter_count))
                print("Current task model loss: {:.4f}".format(task_loss.item()))

            if iter_count % 1000 == 0:
                acc = self.validate(task_model, val_dataloader)
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(task_model)

                print("current step: {} acc: {}".format(iter_count, acc))
                print("best acc: ", best_acc)

        if self.args.cuda:
            best_model = best_model.cuda()

        final_accuracy = self.test(best_model)
        return final_accuracy, task_model
        #return final_accuracy, best_model

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



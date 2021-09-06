import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import recall_score

def forward_pass(self, dataloader):
    # # train
    # if self.args.mode == 'train':
    #     self.model.train()
    #     dataloader = self.dataloader_train
    # else:
    #     self.model.eval()
    #     dataloader = self.dataloader_test

    if self.args.mode == 'train':
        self.model.train()
    else:
        self.model.eval()

    loss_epoch = 0
    correct_epoch = 0
    num_samples = 0
    accuracy_list, sensitivity_list, specificity_list = [], [], []

    for index_batch, (features, labels_real) in enumerate(dataloader):
        features, labels_real = features.to(self.args.device), labels_real.to(self.args.device)
        self.optimizer.zero_grad()
        labels_pred = self.model(features)

        # # convert predictions between 0 and 1 (one_by_three)
        # sigmoid = nn.Sigmoid()
        # for index_class in range(labels_real.shape[1]):
        #     labels_pred[:, index_class] = sigmoid(labels_pred[:, index_class])
        #
        # loss_batch = self.optimize_parameters(labels_real, labels_pred)

        loss_batch = self.optimize_parameters(labels_real, labels_pred)

        # self.criterion_cross_entropy(torch.randn(len(labels_real), 5, requires_grad=True).cuda(), labels_real)
        # self.criterion_cross_entropy(labels_pred, torch.empty(len(labels_real), dtype=torch.long).random_(5).cuda())

        # # convert predictions to 0 and 1
        # threshold = 0.5
        # labels_pred = torch.where(labels_pred > threshold, 1, 0)
        #
        # correct_batch = torch.sum(torch.sum((labels_pred == labels_real), axis=1) == 3).item()

        loss_epoch += loss_batch
        labels_pred = torch.argmax(labels_pred, dim=1)
        correct_batch = torch.sum(labels_pred == labels_real).item()
        length_batch = len(labels_real)


        correct_epoch += correct_batch
        num_samples += length_batch

        accuracy_batch = (correct_batch / length_batch)
        accuracy_list.append(accuracy_batch)

        # if self.args.feature == 'alcoholism' and self.args.mode == 'val':
        #     sensitivity_batch = recall_score(labels_real.cpu(), labels_pred.cpu(), pos_label=1) * 100.
        #     specificity_batch = recall_score(labels_real.cpu(), labels_pred.cpu(), pos_label=0) * 100.
        #     sensitivity_list.append(sensitivity_batch)
        #     specificity_list.append(specificity_batch)

    num_samples_epoch = len(accuracy_list)
    accuracy_epoch = 100 * (np.sum(accuracy_list) / num_samples_epoch)
    loss_epoch /= num_samples_epoch

    print(
        f'{self.args.mode}\tEpoch:\t[{self.args.index_epoch + 1}/{self.args.num_epochs}]\t{self.args.mode.upper()} Loss:\t{loss_epoch:.3f}\tAccuracy:\t{accuracy_epoch:.3f} %\tCorrect:\t[{correct_epoch}/{num_samples}]',
        end='')
    print()
    # if self.args.feature == 'alcoholism' and self.args.mode == 'val':
    #     sensitivity_epoch = (np.sum(sensitivity_list) / num_samples_epoch)
    #     specificity_epoch = (np.sum(specificity_list) / num_samples_epoch)
    #     print(f'\tSensitivity:\t{sensitivity_epoch:.3f} %\tSpecificity:\t{specificity_epoch:.3f} %')
    # else:
    #     print()
    if self.args.mode == 'val':
        self.save_model(accuracy_epoch, loss_epoch)
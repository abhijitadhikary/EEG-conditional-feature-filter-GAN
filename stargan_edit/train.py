from stargan_edit.train_options import TrainOptions
from stargan_edit.dataloader import get_dataloader
from stargan_edit.stargan_model import get_model
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from stargan_edit.utils import mkdirs, convert
import torch
import os

def train(self):
    self.acc_fake_B_best = -1e9
    parser = TrainOptions()
    opt = parser.parse()
    self.opt = opt
    dataloader_train = get_dataloader(opt, 'train')
    dataloader_val = get_dataloader(opt, 'val')
    self.model = get_model(opt)
    self.model.load_model(opt)
    create_tensorboard(self)

    self.index_step = 0
    for index_epoch in range(opt.epoch_start, opt.num_epochs):
        self.index_epoch = index_epoch
        # epoch_runner(self, opt, dataloader_train, 'train')

        # evaluate model
        epoch_runner(self, opt, dataloader_val, 'val')

def epoch_runner(self, opt, dataloader, run_mode):
    self.num_batches = len(dataloader)
    self.run_mode = run_mode
    self.total = 0
    # train
    for index_batch, batch in enumerate(dataloader):
        self.index_batch = index_batch
        # put models to train or eval mode
        self.model.prepare_models(run_mode)

        # extract features and labels from dataloader
        self.model.set_input(batch)
        self.batch_size = len(self.model.real_A)
        self.total += self.batch_size

        # perform forward pass and optimize parameters
        self.model.optimize_parameters(run_mode)

        # calculate logs
        update_loss_batch(self)
        print_logs_batch(self)
        update_tensorboard(self)
        save_image_grid(self)
        self.index_step += 1

    update_logs_epoch(self)
    print_logs_epoch(self)
    reset_epoch_parameters(self)

def reset_epoch_parameters(self):
    self.total = 0

    for key in self.loss_epoch:
        self.loss_epoch[key] = 0

    for key in self.correct_epoch:
        self.correct_epoch[key] = 0

def print_logs(logs, type, total=0):
    if type == 'loss':
        for key, value in logs.items():
            print(f'{key}: {value:.4f}', end='\t\t')
    elif type == 'acc' or type == 'confidence':
        for key, value in logs.items():
            print(f'{key}: {value:.2f} %', end='\t\t')
    elif type == 'correct':
        for key, value in logs.items():
            total_temp = total
            if 'total' in key:
                total_temp = total * 4
            print(f'{key}: [{value} / {total_temp}]', end='\t\t')
    print()

def print_logs_epoch(self):
    print(f'\n\n{20*"*"} End of epoch {self.index_epoch+1} {20*"*"}')
    print(f'{self.run_mode.upper()}\tEpoch: [{self.index_epoch+1} / {self.opt.num_epochs}]')
    print_logs(self.loss_epoch, 'loss')
    print_logs(self.correct_epoch, 'correct', self.total)
    print_logs(self.acc_epoch, 'acc')
    print_logs(self.confidence_epoch, 'confidence')

def print_logs_batch(self):
    print(f'\n\n{self.run_mode.upper()} ------> Epoch: [{self.index_epoch} / {self.opt.num_epochs}]'
          f'\t\tBatch: [{self.index_batch} / {self.num_batches}]'
          f'\t\tStep: {self.index_step}')
    print_logs(self.loss_batch, 'loss')
    print_logs(self.correct_batch, 'correct', self.batch_size)
    print_logs(self.acc_batch, 'acc')
    print_logs(self.confidence_batch, 'confidence')

def update_logs_epoch(self):
    # loss
    for key in self.loss_epoch:
        total = self.total
        if 'total' in key:
            total = total * 4
        self.loss_epoch[key] /= total

    # confidence
    for key in self.confidence_epoch:
        total = self.total
        self.confidence_epoch[key] /= total

    # acc
    self.acc_epoch = {}
    self.acc_epoch['acc/real_A'] = 0
    self.acc_epoch['acc/real_B'] = 0
    self.acc_epoch['acc/fake_B'] = 0
    self.acc_epoch['acc/rec_A'] = 0
    self.acc_epoch['acc/total'] = 0

    for key_c, key_a in zip(self.correct_epoch, self.acc_epoch):
        total = self.total
        if 'total' in key_a:
            total = total * 4
        self.acc_epoch[key_a] = (self.correct_epoch[key_c] / total) * 100

def update_loss_batch(self):
    # loss
    self.loss_batch = self.model.loss.copy()

    if self.index_step == 0:
        self.loss_epoch = self.loss_batch.copy()
    else:
        for key in self.loss_batch:
            self.loss_epoch[key] += self.loss_batch[key]

    for key in self.loss_batch:
        batch_size = self.batch_size
        if 'total' in key:
            batch_size = batch_size * 4
        self.loss_batch[key] /= batch_size

    # correct
    self.correct_batch = self.model.correct.copy()
    if self.index_step == 0:
        self.correct_epoch = self.correct_batch.copy()
    else:
        for key in self.correct_batch:
            self.correct_epoch[key] += self.correct_batch[key]

    # confidence
    self.confidence_batch = self.model.confidence.copy()
    if self.index_step == 0:
        self.confidence_epoch = self.confidence_batch.copy()
        for key in self.confidence_epoch:
            self.confidence_epoch[key] *= self.batch_size
    else:
        for key in self.confidence_batch:
            self.confidence_epoch[key] = self.confidence_epoch[key] + (self.confidence_batch[key] * self.batch_size)

    # accuracy
    self.acc_batch = {}
    self.acc_batch['acc/real_A'] = 0
    self.acc_batch['acc/real_B'] = 0
    self.acc_batch['acc/fake_B'] = 0
    self.acc_batch['acc/rec_A'] = 0
    self.acc_batch['acc/total'] = 0

    for key_c, key_a in zip(self.correct_batch, self.acc_batch):
        batch_size = self.batch_size
        if 'total' in key_a:
            batch_size = batch_size * 4
        self.acc_batch[key_a] = (self.correct_batch[key_c] / batch_size) * 100

def create_tensorboard(self):
    writer_path = os.path.join(f'{self.opt.path_tensorboard}', f'{self.opt.name_experiment}')
    self.writer = SummaryWriter(writer_path)

def update_tensorboard(self):
    losses = self.loss_batch
    correct = self.correct_batch
    acc = self.acc_batch
    confidence = self.confidence_batch

    if self.run_mode == 'train':
        index = self.index_step
    else:
        index = self.index_epoch

    self.writer.add_scalars(f'Loss/{self.run_mode}', losses, index)
    self.writer.add_scalars(f'Correct/{self.run_mode}', correct, index)
    self.writer.add_scalars(f'Accuracy/{self.run_mode}', acc, index)
    self.writer.add_scalars(f'Confidence/{self.run_mode}', confidence, index)

def save_image_grid(self):
    opt = self.opt
    if self.index_batch == 0:
        mkdirs(opt.path_save_image)
        num_display = opt.num_imsave
        img_grid_real_A = make_grid(convert(self.model.real_A[:num_display], 0, 1))
        img_grid_fake_A = make_grid(convert(self.model.fake_B[:num_display], 0, 1))
        img_grid_rec_A = make_grid(convert(self.model.rec_A[:num_display], 0, 1))
        img_grid_real_B = make_grid(convert(self.model.real_B[:num_display], 0, 1))

        # combine the grids
        img_grid_combined = torch.cat((
            img_grid_real_A, img_grid_fake_A, img_grid_rec_A,  img_grid_real_B
        ), dim=1)
        output_path_full = os.path.join(f'{opt.path_save_image}', f'{self.index_epoch}_{self.index_batch}_{self.run_mode}.jpg')
        save_image(img_grid_combined, output_path_full)

from ccg.train_options import TrainOptions
from ccg.dataloader import get_dataloader
from ccg.ccg_model import get_model
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from ccg.utils import mkdirs, convert
import torch
import os

def train(self):
    parser = TrainOptions()
    opt = parser.parse()
    self.opt = opt
    dataloader_train = get_dataloader(opt, 'train')
    dataloader_val = get_dataloader(opt, 'val')
    self.model = get_model(opt)
    self.model.load_model(opt)
    create_tensorboard(self)

    for index_epoch in range(opt.epoch_start, opt.num_epochs):
        self.index_epoch = index_epoch

        self.index_step = 0
        epoch_runner(self, opt, dataloader_train, 'train')
        # epoch_runner(self, opt, dataloader_val, 'val')





def epoch_runner(self, opt, dataloader, run_mode):

    self.loss_D_A_epoch = 0
    self.loss_D_B_epoch = 0
    self.loss_G_AtoB_epoch = 0
    self.loss_G_BtoA_epoch = 0
    self.loss_cyc_A_epoch = 0
    self.loss_cyc_B_epoch = 0
    self.loss_idt_A_epoch = 0
    self.loss_idt_B_epoch = 0
    self.loss_cls_A_epoch = 0
    self.loss_cls_B_epoch = 0
    self.correct_cls_A_epoch = 0
    self.correct_cls_B_epoch = 0

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
        self.model.optimize_parameters()

        # calculate logs
        update_loss_batch(self)
        print_loss_batch(self)
        update_tensorboard(self)
        save_image_grid(self)
        self.index_step += 1

    update_loss_epoch(self)
    print_loss_epoch(self)

def print_loss_epoch(self):
    print(f'\n{20*"*"} End of epoch {self.index_epoch+1} {20*"*"}')
    print(f'{self.run_mode.upper()}\tEpoch: [{self.index_epoch} / {self.opt.num_epochs}]'
          f'\n\tLosses:'
          f'\t\tD_A: {self.loss_D_A_epoch:.4f}\tD_B: {self.loss_D_B_epoch:.4f}'
          f'\t\tG_AtoB: {self.loss_G_AtoB_epoch:.4f}\tG_BtoA: {self.loss_G_BtoA_epoch:.4f}'
          f'\t\tcyc_A: {self.loss_cyc_A_epoch:.4f}\tcyc_B: {self.loss_cyc_B_epoch:.4f}'
          f'\t\tidt_A: {self.loss_idt_A_epoch:.4f}\tidt_B: {self.loss_idt_B_epoch:.4f}'
          f'\t\tcls_A: {self.loss_cls_A_epoch:.4f}\tcls_B: {self.loss_cls_B_epoch:.4f}\t'
          f'\n\tAccuracy:'
          f'\tcls_A: {self.acc_cls_A_epoch:.2f} %\t[{self.correct_cls_A_epoch} / {self.total}]'
          f'\t\tcls_B: {self.acc_cls_B_epoch:.2f} %\t[{self.correct_cls_B_epoch} / {self.total}]')

def print_loss_batch(self):
    print(f'{self.run_mode.upper()}\tEpoch: [{self.index_epoch} / {self.opt.num_epochs}]'
          f'\tBatch: [{self.index_batch} / {self.num_batches}'
          f'\tStep: {self.index_step}'
          f'\n\tLosses:'
          f'\t\tD_A: {self.loss_D_A_batch:.4f}\tD_B: {self.loss_D_B_batch:.4f}'
          f'\t\tG_AtoB: {self.loss_G_AtoB_batch:.4f}\tG_BtoA: {self.loss_G_BtoA_batch:.4f}'
          f'\t\tcyc_A: {self.loss_cyc_A_batch:.4f}\tcyc_B: {self.loss_cyc_B_batch:.4f}'
          f'\t\tidt_A: {self.loss_idt_A_batch:.4f}\tcyc_B: {self.loss_idt_B_batch:.4f}'
          f'\t\tcls_A: {self.loss_cls_A_batch:.4f}\tcls_B: {self.loss_cls_B_batch:.4f}\t'
          f'\n\tAccuracy:'
          f'\tcls_A: {self.acc_cls_A_batch:.2f} %\t[{self.correct_cls_A_batch} / {self.batch_size}]'
          f'\t\tcls_B: {self.acc_cls_A_batch:.2f} %\t[{self.correct_cls_B_batch} / {self.batch_size}]')

def update_loss_epoch(self):
    total = self.total
    self.loss_D_A_epoch /= total
    self.loss_D_B_epoch /= total
    self.loss_G_AtoB_epoch /= total
    self.loss_G_BtoA_epoch /= total
    self.loss_cls_A_epoch /= total
    self.loss_cls_B_epoch /= total
    self.loss_D_A_epoch /= total
    self.loss_cyc_A_epoch /= total
    self.loss_cyc_B_epoch /= total
    self.loss_idt_A_epoch /= total
    self.loss_idt_B_epoch /= total

    self.acc_cls_A_epoch = (self.correct_cls_A_epoch / total) * 100
    self.acc_cls_B_epoch = (self.correct_cls_B_epoch / total) * 100

def update_loss_batch(self):
    batch_size = self.batch_size
    self.loss_D_A_batch = self.model.loss_D_A.item()
    self.loss_D_A_epoch += self.loss_D_A_batch
    self.loss_D_A_batch /= batch_size

    self.loss_D_B_batch = self.model.loss_D_B.item()
    self.loss_D_B_epoch += self.loss_D_B_batch
    self.loss_D_B_batch /= batch_size

    self.loss_G_AtoB_batch = self.model.loss_G_AtoB.item()
    self.loss_G_AtoB_epoch += self.loss_G_AtoB_batch
    self.loss_G_AtoB_batch /= batch_size

    self.loss_G_BtoA_batch = self.model.loss_G_BtoA.item()
    self.loss_G_BtoA_epoch += self.loss_G_BtoA_batch
    self.loss_G_BtoA_batch /= batch_size

    self.loss_cyc_A_batch = self.model.loss_cyc_A
    self.loss_cyc_A_epoch += self.loss_cyc_A_batch
    self.loss_cyc_A_batch /= batch_size

    self.loss_cyc_B_batch = self.model.loss_cyc_B
    self.loss_cyc_B_epoch += self.loss_cyc_B_batch
    self.loss_cyc_B_batch /= batch_size

    self.loss_idt_A_batch = self.model.loss_idt_A
    self.loss_idt_A_epoch += self.loss_idt_A_batch
    self.loss_idt_A_batch /= batch_size

    self.loss_idt_B_batch = self.model.loss_idt_B
    self.loss_idt_B_epoch += self.loss_idt_B_batch
    self.loss_idt_B_batch /= batch_size

    self.loss_cls_A_batch = self.model.loss_cls_A.item()
    self.loss_cls_A_epoch += self.loss_cls_A_batch
    self.loss_cls_A_batch /= batch_size

    self.loss_cls_B_batch = self.model.loss_cls_B.item()
    self.loss_cls_B_epoch += self.loss_cls_B_batch
    self.loss_cls_B_batch /= batch_size

    self.correct_cls_A_batch = self.model.correct_cls_A
    self.correct_cls_A_epoch += self.correct_cls_A_batch
    self.acc_cls_A_batch = (self.correct_cls_A_batch / batch_size) * 100

    self.correct_cls_B_batch = self.model.correct_cls_B
    self.correct_cls_B_epoch += self.correct_cls_B_batch
    self.acc_cls_B_batch = (self.correct_cls_B_batch / batch_size) * 100

def create_tensorboard(self):
    writer_path = os.path.join(f'{self.opt.path_tensorboard}', f'{self.opt.name_experiment}')
    self.writer = SummaryWriter(writer_path)

def update_tensorboard(self):
    losses = {
        'D_A': self.loss_D_A_batch,
        'D_B': self.loss_D_B_batch,
        'G_AtoB': self.loss_G_AtoB_batch,
        'G_BtoA': self.loss_G_BtoA_batch,
        'cycle_A': self.loss_cyc_A_batch,
        'cycle_B': self.loss_cyc_B_batch,
        'idt_A': self.loss_idt_A_batch,
        'idt_B': self.loss_idt_B_batch,
        'cls_A': self.loss_cls_A_batch,
        'cls_B': self.loss_cls_B_batch
    }
    accuracies = {
        'A': self.acc_cls_A_batch,
        'B': self.acc_cls_A_batch
    }
    self.writer.add_scalars('Loss', losses, self.index_step)
    self.writer.add_scalars('Accuracy', accuracies, self.index_step)

def save_image_grid(self):
    opt = self.opt
    if self.index_batch == 0:
        mkdirs(opt.path_save_image)
        num_display = opt.num_imsave
        img_grid_real_A = make_grid(convert(self.model.real_A[:num_display], 0, 1))
        img_grid_fake_A = make_grid(convert(self.model.fake_A[:num_display], 0, 1))
        img_grid_rec_A = make_grid(convert(self.model.rec_A[:num_display], 0, 1))
        img_grid_idt_A = make_grid(convert(self.model.idt_A[:num_display], 0, 1))

        img_grid_real_B = make_grid(convert(self.model.real_B[:num_display], 0, 1))
        img_grid_fake_B = make_grid(convert(self.model.fake_B[:num_display], 0, 1))
        img_grid_rec_B = make_grid(convert(self.model.rec_B[:num_display], 0, 1))
        img_grid_idt_B = make_grid(convert(self.model.idt_B[:num_display], 0, 1))

        # combine the grids
        img_grid_combined = torch.cat((
            img_grid_real_A, img_grid_fake_B, img_grid_rec_A, img_grid_idt_A,
            img_grid_real_B, img_grid_fake_A, img_grid_rec_B, img_grid_idt_B,
        ), dim=1)
        output_path_full = os.path.join(f'{opt.path_save_image}', f'{self.index_epoch}_{self.index_batch}_{self.run_mode}.jpg')
        save_image(img_grid_combined, output_path_full)

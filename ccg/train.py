from ccg.train_options import TrainOptions
from ccg.dataloader import get_dataloader
from ccg.ccg_model import get_model

def train(self):
    parser = TrainOptions()
    opt = parser.parse()
    self.opt = opt
    dataloader_train = get_dataloader(opt, 'train')
    dataloader_val = get_dataloader(opt, 'val')
    self.model = get_model(opt)

    self.model.load_model(opt)

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

        # calculate losses
        update_loss_batch(self)
        print_loss_batch(self)
        self.index_step += 1

    update_loss_epoch(self)
    print_loss_epoch(self)



def print_loss_epoch(self):
    print(f'\n{20*"*"} End of epoch {self.index_epoch+1} {20*"*"}')
    print(f'{self.run_mode.upper()}\tEpoch: [{self.index_epoch} / {self.opt.num_epochs}]'
          f'\tBatch: [{self.index_batch} / {self.num_batches}'
          f'\tStep: {self.index_step}'
          f'\n\tLosses:'
          f'\t\tD_A: {self.loss_D_A_batch:.4f}\tD_B: {self.loss_D_B_batch:.4f}'
          f'\t\tG_AtoB: {self.loss_G_AtoB_batch:.4f}\tG_BtoA: {self.loss_G_BtoA_batch:.4f}'
          f'\t\tcls_A: {self.loss_cls_A_batch:.4f}\tcls_B: {self.loss_cls_B_batch:.4f}\t'
          f'\n\tAccuracy:\tcls_A: {self.correct_cls_A_batch:.4f} %\tcls_B: {self.correct_cls_B_batch:.4f} %')

def print_loss_batch(self):
    print(f'{self.run_mode.upper()}\tEpoch: [{self.index_epoch} / {self.opt.num_epochs}]'
          f'\tBatch: [{self.index_batch} / {self.num_batches}'
          f'\tStep: {self.index_step}'
          f'\n\t\tLosses:'
          f'\t\tD_A: {self.loss_D_A_batch:.4f}\tD_B: {self.loss_D_B_batch:.4f}'
          f'\t\tG_AtoB: {self.loss_G_AtoB_batch:.4f}\tG_BtoA: {self.loss_G_BtoA_batch:.4f}'
          f'\t\tcls_A: {self.loss_cls_A_batch:.4f}\tcls_B: {self.loss_cls_B_batch:.4f}\t'
          f'\t\tAccuracy:\tcls_A: {self.correct_cls_A_batch:.4f} %\tcls_B: {self.correct_cls_B_batch:.4f} %')

def update_loss_epoch(self):
    total = self.total
    self.loss_D_A_epoch /= total
    self.loss_D_B_epoch /= total
    self.loss_G_AtoB_epoch /= total
    self.loss_G_BtoA_epoch /= total
    self.loss_cls_A_epoch /= total
    self.loss_cls_B_epoch /= total
    self.loss_D_A_epoch /= total
    self.correct_cls_A_epoch /= total
    self.correct_cls_B_epoch /= total
    self.correct_cls_A_epoch *= 100
    self.correct_cls_B_epoch *= 100

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

    self.loss_cls_A_batch = self.model.loss_cls_A.item()
    self.loss_cls_A_epoch += self.loss_cls_A_batch
    self.loss_cls_A_batch /= batch_size

    self.loss_cls_B_batch = self.model.loss_cls_B.item()
    self.loss_cls_B_epoch += self.loss_cls_B_batch
    self.loss_cls_B_batch /= batch_size

    self.correct_cls_A_batch = self.model.correct_cls_A
    self.correct_cls_A_epoch += self.correct_cls_A_batch
    self.correct_cls_A_batch /= batch_size
    self.correct_cls_A_batch *= 100

    self.correct_cls_B_batch = self.model.correct_cls_B
    self.correct_cls_B_epoch += self.correct_cls_B_batch
    self.correct_cls_B_batch /= batch_size
    self.correct_cls_B_batch *= 100



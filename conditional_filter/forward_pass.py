import torch
import os
import torchvision


def forward_pass(self, dataloader, mode='train'):
    num_batches = len(dataloader)
    loss_D_cls_epoch = 0
    loss_D_adv_epoch = 0
    loss_D_total_epoch = 0
    loss_G_epoch = 0
    D_cls_conf_real_epoch = 0
    D_adv_conf_real_epoch = 0
    D_cls_conf_fake_epoch = 0
    D_adv_conf_fake_epoch = 0

    # train
    for index_batch, batch in enumerate(dataloader):
        if mode == 'train':
            self.model_D_cls.train()
            self.model_D_adv.train()
            self.model_G.train()
        else:
            self.model_D_cls.eval()
            self.model_D_adv.eval()
            self.model_G.eval()

        # extract batch from dataloader
        image, image_c_real, image_c_fake, condition_array_real, condition_array_fake, identity, stimulus, alcoholism, \
        targets_real_cls, targets_real_adv, targets_fake_cls, targets_fake_adv = batch

        # to device
        image, image_c_real, image_c_fake, condition_array_real, condition_array_fake, identity, stimulus, \
            alcoholism, targets_real_cls, targets_real_adv, targets_fake_cls, targets_fake_adv = \
            image.to(self.args.device), image_c_real.to(self.args.device), image_c_fake.to(self.args.device), \
            condition_array_real.to(self.args.device), condition_array_fake.to(self.args.device), \
            identity.to(self.args.device), stimulus.to(self.args.device), alcoholism.to(self.args.device), \
            targets_real_cls.to(self.args.device), targets_real_adv.to(self.args.device), \
            targets_fake_cls.to(self.args.device), targets_fake_adv.to(self.args.device)

        batch_size, num_channels, height, width = image.shape
        num_channels_cat = image_c_fake.shape[1]

        # generate image_fake image
        image_fake_temp = self.model_G.forward(image_c_fake)
        image_fake_rec = torch.ones((batch_size, num_channels_cat, height, width)).to(self.args.device)
        image_fake_rec[:, :3] = image_fake_temp
        image_fake_rec[:, 3:] = image_c_fake[:,
                                3:]  # this needs to be checked, should be the condition instead of the real image
        # image_fake_rec[:, 3:] = image # this needs to be checked, should be the condition instead of the real image

        # train discriminator - real
        self.model_D_cls.zero_grad()
        self.model_D_adv.zero_grad()
        out_D_cls_real = self.model_D_cls(image_c_real).squeeze(3)
        loss_D_cls_real = self.criterion_D_cls(out_D_cls_real.squeeze(2), targets_real_cls.squeeze(2))

        out_D_adv_real = self.model_D_adv(image_c_real).reshape(-1, 1)
        loss_D_adv_real = self.criterion_D_adv(out_D_adv_real, targets_real_adv)

        # train discriminator - fake
        out_D_cls_fake = self.model_D_cls(image_fake_rec.detach()).squeeze(3)
        loss_D_cls_fake = self.criterion_D_cls(out_D_cls_fake, condition_array_fake)
        out_D_adv_fake = self.model_D_adv(image_fake_rec.detach()).reshape(-1, 1)
        loss_D_adv_fake = self.criterion_D_adv(out_D_adv_fake, targets_fake_adv)

        loss_D_cls = (loss_D_cls_real + loss_D_cls_fake) * self.args.loss_D_cls_factor
        loss_D_adv = (loss_D_adv_real + loss_D_adv_fake) * self.args.loss_D_adv_factor

        # measure the confidence of the discriminator
        D_cls_conf_real = torch.mean(out_D_cls_real)
        D_adv_conf_real = torch.mean(out_D_adv_real)
        D_cls_conf_fake = torch.mean(out_D_cls_fake)
        D_adv_conf_fake = torch.mean(out_D_adv_fake)

        # add batch confidences to the epoch
        D_cls_conf_real_epoch += D_cls_conf_real
        D_adv_conf_real_epoch += D_adv_conf_real
        D_cls_conf_fake_epoch += D_cls_conf_fake
        D_adv_conf_fake_epoch += D_adv_conf_fake

        # final D loss
        loss_D = (loss_D_cls + loss_D_adv) * self.args.loss_D_total_factor

        if mode == 'train':
            loss_D.backward()
            self.optimizer_D_cls.step()
            self.optimizer_D_adv.step()

        # train generator
        out_D_cls_fake = self.model_D_cls(image_fake_rec).squeeze(3)
        out_D_adv_fake = self.model_D_adv(image_fake_rec).reshape(-1, 1)
        self.model_G.zero_grad()
        loss_G_cls = self.criterion_D_cls(out_D_cls_fake, condition_array_fake)  # need to consider D adv
        loss_G_adv = self.criterion_D_adv(out_D_adv_fake,
                                          targets_real_adv)  # create a new/separate tensor for targets_real_adv
        loss_G_gan = loss_G_cls + loss_G_adv
        loss_G_L1 = self.criterion_G(image_fake_temp, image)

        loss_G = (loss_G_gan * self.args.loss_G_gan_factor) + (loss_G_L1 * self.args.loss_G_l1_factor)

        if mode == 'train':
            loss_G.backward()
            self.optimizer_G.step()

        loss_D_cls_value = loss_D_cls.detach().cpu().item()
        loss_D_adv_value = loss_D_adv.detach().cpu().item()
        loss_D_total_value = loss_D.detach().cpu().item()
        loss_G_value = loss_G.detach().cpu().item()

        # loss_D_value, loss_G_value = forward_pass(args, batch, mode='train')
        loss_D_cls_epoch += loss_D_cls_value
        loss_D_adv_epoch += loss_D_adv_value
        loss_D_total_epoch += loss_D_total_value
        loss_G_epoch += loss_G_value

        # create image grids for visualization
        if index_batch == 0:
            num_display = 8
            img_grid_real = torchvision.utils.make_grid(image[:num_display], normalize=True, range=(0, 1))
            img_grid_fake = torchvision.utils.make_grid(image_fake_temp[:num_display], normalize=True, range=(0, 1))

            # combine the grids
            img_grid_combined = torch.hstack((img_grid_real, img_grid_fake))
            output_path = os.path.join('.', 'conditional_filter', 'output', self.args.model_name,
                                       self.args.split_variant, f'{self.args.index_epoch}_{index_batch}_{mode}.jpg')
            torchvision.utils.save_image(img_grid_combined, output_path)

    loss_D_cls_epoch /= num_batches
    loss_D_adv_epoch /= num_batches
    loss_D_total_epoch /= num_batches
    loss_G_epoch /= num_batches
    D_cls_conf_real_epoch /= num_batches
    D_adv_conf_real_epoch /= num_batches
    D_cls_conf_fake_epoch /= num_batches
    D_adv_conf_fake_epoch /= num_batches

    print_loss(loss_D_cls_epoch, loss_D_adv_epoch, loss_D_total_epoch, loss_G_epoch,
               D_cls_conf_real_epoch, D_adv_conf_real_epoch, D_cls_conf_fake_epoch, D_adv_conf_fake_epoch,
               self.args.index_epoch, self.args.num_epochs, mode)

    return loss_D_cls_epoch, loss_D_adv_epoch, loss_D_total_epoch, loss_G_epoch, \
           D_cls_conf_real_epoch, D_adv_conf_real_epoch, D_cls_conf_fake_epoch, D_adv_conf_fake_epoch


def print_loss(loss_D_cls, loss_D_adv, loss_D_total, loss_G,
               D_cls_conf_real, D_adv_conf_real, D_cls_conf_fake, D_adv_conf_fake,
               index_epoch=-1, num_epochs=-1, mode='train'):
    epoch_num = ''
    # space_pre = '\t\t\t\t'
    space_pre = ''
    space_post = '\t\t'
    if mode == 'train':
        epoch_num = f'epoch: [{index_epoch + 1}/{num_epochs}]\t'
        space_pre = '\n'
        space_post = '\t'
    elif mode == 'test':
        space_post = '\t'
    print(f'{epoch_num}{space_pre}{mode}{space_post}'
          f'Loss: -->\t'
          f'D_cls: {loss_D_cls:.4f}\t'
          f'D_adv: {loss_D_adv:.4f}\t\t'
          f'D_total: {loss_D_total:.4f}\t\t'
          f'G: {loss_G:.4f}\t\t'
          f'Confidence: -->\t\t'
          f'D_cls_real: {D_cls_conf_real: .4f}\t\t'
          f'D_cls_fake: {D_cls_conf_fake: .4f}\t\t'
          f'D_adv_real: {D_adv_conf_real: .4f}\t\t'
          f'D_adv_fake: {D_adv_conf_fake: .4f}'
          )

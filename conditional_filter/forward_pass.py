import torch
import os
import torchvision

def forward_pass(args, dataloader, mode='train'):
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
            args.model_D_cls.train()
            args.model_D_adv.train()
            args.model_G.train()
        else:
            args.model_D_cls.eval()
            args.model_D_adv.eval()
            args.model_G.eval()

        image, image_c_real, image_c_fake, condition_array_real, condition_array_fake, identity, stimulus, alcoholism, \
        targets_real_cls, targets_real_adv, targets_fake_cls, targets_fake_adv = batch
        image, image_c_real, image_c_fake, condition_array_real, condition_array_fake, identity, stimulus, \
        alcoholism, targets_real_cls, targets_real_adv, targets_fake_cls, targets_fake_adv = \
            image.to(args.device), image_c_real.to(args.device), image_c_fake.to(args.device), \
            condition_array_real.to(args.device), condition_array_fake.to(args.device), \
            identity.to(args.device), stimulus.to(args.device), alcoholism.to(args.device), \
            targets_real_cls.to(args.device), targets_real_adv.to(args.device), \
            targets_fake_cls.to(args.device), targets_fake_adv.to(args.device)

        batch_size, num_channels, height, width = image.shape
        num_channels_cat = image_c_fake.shape[1]

        # generate image_fake image
        image_fake_temp = args.model_G.forward(image_c_fake)
        image_fake_rec = torch.ones((batch_size, num_channels_cat, height, width)).to(args.device)
        image_fake_rec[:, :3] = image_fake_temp
        image_fake_rec[:, 3:] = image_c_fake[:,
                                3:]  # this needs to be checked, should be the condition instead of the real image
        # image_fake_rec[:, 3:] = image # this needs to be checked, should be the condition instead of the real image

        # train discriminator - real
        args.model_D_cls.zero_grad()
        args.model_D_adv.zero_grad()
        out_D_cls_real = args.model_D_cls(image_c_real).squeeze(3)
        loss_D_cls_real = args.criterion_D_cls(out_D_cls_real, targets_real_cls)

        out_D_adv_real = args.model_D_adv(image_c_real).reshape(-1, 1)
        loss_D_adv_real = args.criterion_D_adv(out_D_adv_real, targets_real_adv)

        # train discriminator - fake
        out_D_cls_fake = args.model_D_cls(image_fake_rec.detach()).squeeze(3)
        loss_D_cls_fake = args.criterion_D_cls(out_D_cls_fake, condition_array_fake)
        out_D_adv_fake = args.model_D_adv(image_fake_rec.detach()).reshape(-1, 1)
        loss_D_adv_fake = args.criterion_D_adv(out_D_adv_fake, targets_fake_adv)

        loss_D_cls = (loss_D_cls_real + loss_D_cls_fake) * args.loss_D_cls_factor
        loss_D_adv = (loss_D_adv_real + loss_D_adv_fake) * args.loss_D_adv_factor

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
        loss_D = (loss_D_cls + loss_D_adv) * args.loss_D_total_factor

        if mode == 'train':
            loss_D.backward()
            args.optimizer_D_cls.step()
            args.optimizer_D_adv.step()

        # train generator
        out_D_cls_fake = args.model_D_cls(image_fake_rec).squeeze(3)
        out_D_adv_fake = args.model_D_adv(image_fake_rec).reshape(-1, 1)
        args.model_G.zero_grad()
        loss_G_cls = args.criterion_D_cls(out_D_cls_fake, condition_array_fake)  # need to consider D adv
        loss_G_adv = args.criterion_D_adv(out_D_adv_fake,
                                          targets_real_adv)  # create a new/separate tensor for targets_real_adv
        loss_G_gan = loss_G_cls + loss_G_adv
        loss_G_L1 = args.criterion_G(image_fake_temp, image)

        loss_G = (loss_G_gan * args.loss_G_gan_factor) + (loss_G_L1 * args.loss_G_l1_factor)

        if mode == 'train':
            loss_G.backward()
            args.optimizer_G.step()

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
            output_path = os.path.join('.', 'output', args.model_name, args.split_variant, f'{args.index_epoch}_{index_batch}_{mode}.jpg')
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
               args.index_epoch, args.num_epochs, mode)

    return loss_D_cls_epoch, loss_D_adv_epoch, loss_D_total_epoch, loss_G_epoch, \
           D_cls_conf_real_epoch, D_adv_conf_real_epoch, D_cls_conf_fake_epoch, D_adv_conf_fake_epoch
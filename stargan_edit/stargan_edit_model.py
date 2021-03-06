import torch
import torch.nn as nn
import itertools
import os
from stargan_edit.model import Discriminator, Generator
from stargan_edit.networks import define_G, define_C, define_D
from stargan_edit.generator import Generator
from stargan_edit.discriminator import Discriminator
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

def get_model(opt):
    model = StarGANModelEdit(opt)
    print(f'model [{model.get_model_name()}] was created')
    return model

class StarGANModelEdit:
    def get_model_name(self):
        return self.model_name

    def __init__(self, opt):
        self.model_name = 'StarGAN EDIT Model'
        self.num_classes = opt.num_classes

        self.loss_names = ['net_D', 'net_G']

        if opt.mode_run == 'train':
            self.model_names = ['net_G', 'net_D']
        else:  # during test time, only load Gs
            self.model_names = ['net_G']

        self.define_generators(opt)
        self.define_discriminators(opt)
        self.define_classsifier(opt)
        self.define_additional_classsifier(opt)
        self.define_criterions(opt)
        self.define_optimizers(opt)
        self.opt = opt
        self.loss = {}


    def define_generators(self, opt):

        # my unet
        self.net_G = Generator(in_channels=opt.num_classes+opt.num_channels,
                               out_channels=opt.num_channels,
                               ngf=opt.ngf,
                               activation='sigmoid').cuda()

        self.net_G = torch.nn.DataParallel(self.net_G)

        # stargan's resnet
        # self.net_G = Generator(conv_dim=opt.ngf,
        #                        c_dim=opt.num_channels+opt.num_classes,
        #                        repeat_num=opt.num_resnet_blocks).cuda()

        # resnet
        # self.net_G = define_G(input_nc=opt.num_channels + opt.num_classes,
        #                       output_nc=opt.num_channels,
        #                       ngf=opt.ngf,
        #                       which_model_netG='resnet_9blocks',
        #                       ).cuda()


        # unet dimension mismatch
        # self.net_G = define_G(input_nc=opt.num_channels+opt.num_classes,
        #                       output_nc=opt.num_channels,
        #                       ngf=opt.ngf,
        #                       which_model_netG='unet_256',
        #                       ).cuda()

    def define_discriminators(self, opt):
        if opt.mode_run == 'train':
            self.net_D = Discriminator(in_channels=opt.num_classes+opt.num_channels,
                                       ngf=opt.ngf,
                                       activation='sigmoid')

            self.net_D = torch.nn.DataParallel(self.net_D)

            # self.net_D = define_D(input_nc=opt.num_channels,
            #                       ndf=opt.ngf,
            #                       which_model_netD='basic').cuda()
            # self.net_D = Discriminator(image_size=opt.image_size,
            #                            conv_dim=opt.ndf,
            #                            c_dim=opt.num_classes,
            #                            repeat_num=opt.num_resnet_blocks).cuda()

            # TODO implement image pool
            # self.pool_fake_A = ImagePool(opt.size_image_pool)
            # self.pool_fake_B = ImagePool(opt.size_image_pool)

    def define_classsifier(self, opt):
        self.net_cls = ResNet34(opt.num_classes)
        self.net_cls = torch.nn.DataParallel(self.net_cls)

        load_path = os.path.join('stargan_edit', 'additional_classifier_checkpoint.pth')
        checkpoint = torch.load(load_path)
        self.net_cls.load_state_dict(checkpoint['net_ad_cls_state_dict'])
        self.net_cls.load_state_dict(checkpoint['net_ad_cls_state_dict'])

        # self.net_cls = define_C(classifier_name='ResNet18',
        #                         num_classes=opt.num_classes).cuda()

        # self.net_cls = Discriminator(image_size=opt.image_size,
        #                                 conv_dim=opt.ndf,
        #                                 c_dim=opt.num_classes,
        #                                 repeat_num=opt.num_resnet_blocks).cuda()

    def define_additional_classsifier(self, opt):
        self.net_ad_cls = ResNet34(opt.num_classes)
        self.optimizer_ad_cls = torch.optim.Adam(self.net_ad_cls.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.net_ad_cls = torch.nn.DataParallel(self.net_ad_cls)

        load_path = os.path.join('stargan_edit', 'additional_classifier_checkpoint.pth')
        checkpoint = torch.load(load_path)
        self.net_ad_cls.load_state_dict(checkpoint['net_ad_cls_state_dict'])
        self.optimizer_ad_cls.load_state_dict(checkpoint['optimizer_ad_cls_state_dict'])

        # self.net_ad_cls = define_C(classifier_name='ResNet18',
        #                            num_classes=opt.num_classes).cuda()

        # self.net_ad_cls = Discriminator(image_size=opt.image_size,
        #                                 conv_dim=opt.ndf,
        #                                 c_dim=opt.num_classes,
        #                                 repeat_num=opt.num_resnet_blocks).cuda()

    def define_criterions(self, opt):
        self.criterion_D = self.get_criterion_D(opt)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_id = nn.L1Loss()

    def define_optimizers(self, opt):
        self.optimizer_G = torch.optim.SGD(self.net_G.parameters(),
                                           lr=opt.lr,
                                           momentum=opt.momentum,
                                           weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.SGD(self.net_D.parameters(),
                                           lr=opt.lr,
                                           momentum=opt.momentum,
                                           weight_decay=opt.weight_decay)
        self.optimizer_cls = torch.optim.SGD(self.net_cls.parameters(),
                                             lr=opt.lr,
                                             momentum=opt.momentum,
                                             weight_decay=opt.weight_decay)

        # self.optimizer_G = torch.optim.Adam(itertools.chain(self.net_G.parameters()),
        #                                     lr=opt.lr,
        #                                     betas=(opt.beta1, 0.999))
        # self.optimizer_D = torch.optim.Adam(itertools.chain(self.net_D.parameters()),
        #                                     lr=opt.lr,
        #                                     betas=(opt.beta1, 0.999))
        # self.optimizer_cls = torch.optim.Adam(itertools.chain(self.net_cls.parameters()),
        #                                       lr=opt.lr,
        #                                       betas=(opt.beta1, 0.999))


    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_criterion_D(self, opt):
        if opt.loss_name_D == 'wgan':
            # TODO implement wgan loss
            raise NotImplementedError('Need to implement WGAN loss')
        elif opt.loss_name_D == 'mse':
            loss_D = nn.MSELoss()
        elif opt.loss_name_D == 'bce':
            loss_D = nn.BCELoss()
        else:
            raise NotImplementedError('Invalid net_D loss')
        return loss_D

    def set_input(self, input):
        self.real_A = input['A'].cuda()
        self.label_A = input['A_label'].cuda()

        self.real_B = input['B'].cuda()
        self.label_B = input['B_label'].cuda()

    def prepare_models(self, run_mode):
        if run_mode == 'train':
            self.net_D.train()
            self.net_G.train()
        else:
            self.net_D.eval()
            self.net_G.eval()

    def cat_label(self, feature, label):
        batch_size, num_channels, height, width = feature.shape
        # feature_con = torch.ones((batch_size, num_channels + self.num_classes, height, width), dtype=torch.float32).to(feature.device)

        label_array = torch.ones((batch_size, self.num_classes, height, width), dtype=torch.float32,
                                 requires_grad=False).to(feature.device)

        for index_batch in range(batch_size):
            for index in range(self.num_classes):
                current_label = label[index_batch].item()
                # TODO add randomness
                if index == current_label:
                    label_multiplier = 1.0
                else:
                    label_multiplier = 0.0
                current_channel = torch.ones((height, width), dtype=torch.float32) * label_multiplier
                label_array[index_batch, index] = current_channel

        feature_con = torch.cat((feature, label_array), dim=1)

        return feature_con
    def get_num_correct(self, pred, target):
        pred = torch.argmax(pred, dim=1)
        return torch.sum(pred == target).item()
        # return torch.sum(torch.argmax(pred, dim=1) == target.long()).item()


    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).cuda()
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

    def generate_label_tensors(self, feature, label):

        batch_size, num_channels, height, width = feature.shape
        label_array = torch.ones((batch_size, self.num_classes, height, width), dtype=torch.float32,
                                 requires_grad=False).to(feature.device)
        for index_batch in range(batch_size):
            for index in range(self.num_classes):
                current_label = label[index_batch].item()
                # TODO add randomness
                if index == current_label:
                    label_multiplier = 1.0
                else:
                    label_multiplier = 0.0
                current_channel = torch.ones((height, width), dtype=torch.float32) * label_multiplier
                label_array[index_batch, index] = current_channel
        return label_array

    def generate_label_one_hot(self, labels):
        batch_size = len(labels)

        label_array = torch.ones((batch_size, self.num_classes), dtype=torch.float32,
                                 requires_grad=False).to(labels.device)
        for index_batch in range(batch_size):
            for index in range(self.num_classes):
                current_label = labels[index_batch].item()
                # TODO add randomness
                if index == current_label:
                    label_multiplier = 1.0
                else:
                    label_multiplier = 0.0
                current_channel = 1 * label_multiplier
                label_array[index_batch, index] = current_channel
        return label_array

    def get_label(self, features, type):
        batch_size = len(features)
        if type == 'real':
            max = 0.8
            min = 0.999
        elif type == 'fake':
            max = 0.2
            min = 0.0

        target = (max - min) * torch.rand((batch_size)) + min
        target_tensor = torch.ones_like(features)
        for index in range(batch_size):
            target_tensor[index] *= target[index]
        return target_tensor.cuda()

    def optimize_parameters(self, run_mode):
        if run_mode == 'train':
            self.net_D.train()
            self.net_G.train()
            self.net_cls.train()
            self.net_ad_cls.train()
        else:
            self.net_D.eval()
            self.net_G.eval()
            self.net_cls.eval()
            self.net_ad_cls.eval()

        # create local copies
        real_A = self.real_A
        real_B = self.real_B
        label_A = self.label_A
        label_B = self.label_B

        label_oh_A = self.generate_label_one_hot(label_A)
        label_oh_B = self.generate_label_one_hot(label_B)

        # generate fake image in domain B
        fake_B = self.net_G(real_A, label_oh_B)
        # generate reconstructed image in domain A
        rec_A = self.net_G(fake_B, label_oh_A)

        # =================================================================================== #
        #                               1. Train the discriminator                            #
        # =================================================================================== #
        if run_mode == 'train':
            for _ in range(self.opt.num_loops_D):
                self.optimizer_D.zero_grad()

                # train for real images for domain A
                D_real_A = self.net_D(real_A.detach(), label_oh_A.detach())
                label_real = self.get_label(D_real_A, 'real')
                loss_D_real_A = self.criterion_D(D_real_A, label_real)
                conf_D_real_A = torch.mean(D_real_A.detach()).item()

                # # train for real images for domain B
                # D_real_B = self.net_D(real_B.detach(), label_oh_B.detach())
                # label_real = self.get_label(D_real_B, 'real')
                # loss_D_real_B = self.criterion_D(D_real_B, label_real)
                # conf_D_real_B = torch.mean(D_real_B.detach()).item()

                # train for fake images in domain B
                D_fake_B = self.net_D(fake_B.detach(), label_oh_B.detach())
                label_fake = self.get_label(D_fake_B, 'fake')
                loss_D_fake_B = self.criterion_D(D_fake_B, label_fake)
                conf_D_fake_B = torch.mean(D_fake_B.detach()).item()

                # train for reconstructed images in domain A
                D_rec_A = self.net_D(rec_A.detach(), label_oh_A.detach())
                label_fake = self.get_label(D_rec_A, 'fake')
                loss_D_rec_A = self.criterion_D(D_rec_A, label_fake)
                conf_D_rec_A = torch.mean(D_rec_A.detach()).item()

                loss_D_adv = loss_D_real_A + loss_D_fake_B + loss_D_rec_A
                # loss_D_adv = loss_D_real_A + loss_D_real_B + loss_D_fake_B + loss_D_rec_A
                loss_D = loss_D_adv
                loss_D.backward()
                self.optimizer_D.step()

        # =================================================================================== #
        #                      2. Train the generator and classifier                          #
        # =================================================================================== #
        if run_mode == 'train':
            self.optimizer_G.zero_grad()
            self.optimizer_cls.zero_grad()

        # train Generator for fake images in domain B
        D_G_fake_B = self.net_D(fake_B, label_oh_B.detach())
        label_real = self.get_label(D_G_fake_B, 'real')
        loss_G_fake_B = self.criterion_D(D_G_fake_B, label_real)
        conf_D_G_fake_B = torch.mean(D_G_fake_B.detach()).item()

        # train Generator for reconstructed images in domain A
        D_G_rec_A = self.net_D(rec_A, label_oh_A.detach())
        label_real = self.get_label(D_G_rec_A, 'real')
        loss_G_rec_A = self.criterion_D(D_G_rec_A, label_real)
        conf_D_G_rec_A = torch.mean(D_G_rec_A.detach()).item()

        # train classifier for fake_B
        cls_fake_B = self.net_cls(fake_B)
        loss_cls_fake_B = self.criterion_cls(cls_fake_B, label_B)
        correct_cls_fake_B = self.get_num_correct(cls_fake_B.detach(), label_B.detach())

        # train classifier for rec_A
        cls_rec_A = self.net_cls(rec_A)
        loss_cls_rec_A = self.criterion_cls(cls_rec_A, label_A)
        correct_cls_rec_A = self.get_num_correct(cls_rec_A.detach(), label_A.detach())

        # train classifier for real A
        cls_real_A = self.net_cls(real_A)
        loss_cls_real_A = self.criterion_cls(cls_real_A, label_A)
        correct_cls_real_A = self.get_num_correct(cls_real_A.detach(), label_A.detach())

        # # train classifier for real B
        # cls_real_B = self.net_cls(real_B)
        # loss_cls_real_B = self.criterion_cls(cls_real_B, label_B)
        # correct_cls_real_B = self.get_num_correct(cls_real_B.detach(), label_B.detach())

        # l1 losses
        loss_l1_fake_B = self.criterion_l1(fake_B, real_B)
        loss_l1_rec_A = self.criterion_l1(rec_A, real_A)
        loss_G_cyc = loss_l1_fake_B + loss_l1_rec_A

        loss_G_adv = loss_G_fake_B + loss_G_rec_A

        # loss_G_cls_real = correct_cls_real_A
        # loss_G_cls_real = (correct_cls_real_A + correct_cls_real_B)
        loss_G_cls_fake = (loss_cls_fake_B + loss_cls_rec_A)
        loss_G_cls = loss_G_cls_fake
        # loss_G_cls = loss_G_cls_real + loss_G_cls_fake

        loss_G = loss_G_adv + loss_G_cls + loss_G_cyc

        if run_mode == 'train':
            loss_G.backward()
            self.optimizer_G.step()
            self.optimizer_cls.step()

        # =================================================================================== #
        #                            3. Train Additional Classifier                           #
        # =================================================================================== #
        # if run_mode == 'train':
        #     self.optimizer_ad_cls.zero_grad()

        # train additional classifier for real A
        with torch.no_grad():
            self.net_ad_cls.eval()
            ad_cls_real_A = self.net_ad_cls(real_A.detach())
            loss_ad_cls_real_A = self.criterion_cls(ad_cls_real_A, label_A)
            correct_ad_cls_real_A = self.get_num_correct(ad_cls_real_A.detach(), label_A.detach())

        # train additional classifier for fake images in domain B
        with torch.no_grad():
            self.net_ad_cls.eval()
            ad_cls_fake_B = self.net_ad_cls(fake_B.detach())
            loss_ad_cls_fake_B = self.criterion_cls(ad_cls_fake_B.detach(), label_B.detach())
            correct_ad_cls_fake_B = self.get_num_correct(ad_cls_fake_B.detach(), label_B.detach())

        # train additional classifier for reconstructed images in domain A
        with torch.no_grad():
            self.net_ad_cls.eval()
            ad_cls_rec_A = self.net_ad_cls(rec_A.detach())
            loss_ad_cls_rec_A = self.criterion_cls(ad_cls_rec_A.detach(), label_A.detach())
            correct_ad_cls_rec_A = self.get_num_correct(ad_cls_rec_A.detach(), label_A.detach())

        # loss_ad_cls = loss_ad_cls_real_A
        #
        # if run_mode == 'train':
        #     self.net_ad_cls.train()
        #     loss_ad_cls.backward()
        #     self.optimizer_ad_cls.step()


        # =================================================================================== #
        #                                      Logging                                        #
        # =================================================================================== #

        loss = {}
        # D
        loss['D/real_A'] = loss_D_real_A.item() if run_mode == 'train' else 0
        # loss['D/real_B'] = loss_D_real_B.item() if run_mode == 'train' else 0
        loss['D/fake_B'] = loss_D_fake_B.item() if run_mode == 'train' else 0
        loss['D/rec_A'] = loss_D_rec_A.item() if run_mode == 'train' else 0
        # loss['D/adv'] = loss_D_adv.item() if run_mode == 'train' else 0

        # G
        loss['G/fake_B'] = loss_G_fake_B.item()
        loss['G/rec_A'] = loss_G_rec_A.item()
        # loss['G/adv'] = loss_G_adv.item()

        # cyc
        # loss['G/l1_fake_B'] = loss_l1_fake_B.item()
        loss['G/l1_rec_A'] = loss_l1_rec_A.item()
        loss['G/cyc'] = loss_G_cyc.item()

        # cls
        loss['cls/real_A'] = loss_cls_real_A.item()
        # loss['cls/real_B'] = loss_cls_real_B.item()
        loss['cls/fake_B'] = loss_cls_fake_B.item()
        loss['cls/rec_A'] = loss_cls_rec_A.item()

        # ad cls
        loss['ad_cls/real_A'] = loss_ad_cls_real_A.item()
        loss['ad_cls/fake_B'] = loss_ad_cls_fake_B.item()
        loss['ad_cls/rec_A'] = loss_ad_cls_rec_A.item()

        self.loss = loss

        # correct
        correct = {}
        correct['correct/cls_real_A'] = correct_cls_real_A
        # correct['correct/cls_real_B'] = correct_cls_real_B

        correct['correct/cls_fake_B'] = correct_cls_fake_B
        correct['correct/cls_rec_A'] = correct_cls_rec_A

        correct['correct/ad_cls_real_A'] = correct_ad_cls_real_A
        correct['correct/ad_cls_fake_B'] = correct_ad_cls_fake_B
        correct['correct/ad_cls_rec_A'] = correct_ad_cls_rec_A

        self.correct = correct

        # confidence
        confidence = {}
        confidence['conf_D/real_A'] = conf_D_real_A * 100 if run_mode == 'train' else 0
        # confidence['conf_D/real_B'] = conf_D_real_B * 100 if run_mode == 'train' else 0
        confidence['conf_D/fake_B'] = conf_D_fake_B * 100 if run_mode == 'train' else 0
        confidence['conf_D/rec_A'] = conf_D_rec_A * 100 if run_mode == 'train' else 0
        # confidence['conf_D/total_D'] = (((conf_D_real_A + conf_D_real_B + conf_D_fake_B + conf_D_rec_A) / 4) * 100) \
        #     if run_mode == 'train' else 0

        confidence['conf_G/fake_B'] = conf_D_G_fake_B * 100
        confidence['conf_G/rec_A'] = conf_D_G_rec_A * 100
        # confidence['conf_G/total_G'] = ((conf_D_G_fake_B + conf_D_G_rec_A) / 2) * 100

        self.confidence = confidence

        # for displaying purposes
        self.fake_B = fake_B
        self.rec_A = rec_A









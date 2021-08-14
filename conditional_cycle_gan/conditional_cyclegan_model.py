import torch
import itertools
import numpy as np
from utils.image_pool import ImagePool
from conditional_cycle_gan.base_model import BaseModel
from conditional_cycle_gan import networks

class ConditionalCycleGANModel(BaseModel):
    def name(self):
        return 'ConditionalCycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.correct_batch_A = np.NINF
        self.correct_batch_B = np.NINF

        # self.acc_epoch_A = np.NINF
        # self.acc_epoch_B = np.NINF

        self.num_classes = opt.num_classes
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'cls_A', 'cls_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.is_train and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.is_train:
            self.model_names = ['G_AtoB', 'G_BtoA', 'D_A', 'D_B', 'cls_A', 'cls_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        use_sigmoid = True if opt.gan_loss_type == 'mse' else False
        self.net_G_AtoB = networks.define_G(opt.input_nc, opt.output_nc,
                                            opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type,
                                            self.gpu_ids, normalize_output=use_sigmoid)
        self.net_G_BtoA = networks.define_G(opt.input_nc, opt.output_nc,
                                            opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type,
                                            self.gpu_ids, normalize_output=use_sigmoid)

        if self.is_train:
            self.net_D_A = networks.define_D(opt.output_nc, opt.ndf,
                                             opt.which_model_netD,
                                             opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.net_D_B = networks.define_D(opt.output_nc, opt.ndf,
                                             opt.which_model_netD,
                                             opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.net_cls_A = networks.define_C(opt.num_channels, opt.num_classes, 32, opt.init_type, self.gpu_ids)
            self.net_cls_B = networks.define_C(opt.num_channels, opt.num_classes, 32, opt.init_type, self.gpu_ids)


        if self.is_train:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_loss_type=opt.gan_loss_type).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionCls = torch.nn.CrossEntropyLoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.net_G_AtoB.parameters(), self.net_G_BtoA.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.net_D_A.parameters(), self.net_D_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_cls = torch.optim.Adam(itertools.chain(self.net_cls_A.parameters(), self.net_cls_B.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_cls)


    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device, dtype=torch.float)
        self.label_A = input['A_label' if AtoB else 'B_label'].to(self.device, dtype=torch.float)

        self.real_B = input['B' if AtoB else 'A'].to(self.device, dtype=torch.float)
        self.label_B = input['B_label' if AtoB else 'A_label'].to(self.device, dtype=torch.float)

    def cat_con_feature(self, feature, label):
        batch_size, num_channels, height, width = feature.shape
        feature_con = torch.ones((batch_size, num_channels + self.num_classes, height, width), dtype=torch.float32).to(feature.device)

        for index_batch in range(batch_size):
            feature_con[index_batch, :num_channels] = feature[index_batch]
            for index in range(self.num_classes):
                # TODO add randomness
                if index == label[index_batch].item():
                    label_multiplier = 1.0
                else:
                    label_multiplier = 0.0
                current_channel = torch.ones((height, width)) * label_multiplier
                feature_con[index_batch, index+num_channels] = current_channel
        return feature_con

    def forward(self):
        self.fake_B = self.net_G_AtoB(self.cat_con_feature(self.real_A, self.label_B))
        self.rec_A = self.net_G_BtoA(self.cat_con_feature(self.fake_B, self.label_A))

        self.fake_A = self.net_G_BtoA(self.cat_con_feature(self.real_B, self.label_A))
        self.rec_B = self.net_G_AtoB(self.cat_con_feature(self.fake_A, self.label_B))

        # TODO look into this whether to run classifier on all three variants
        self.cls_B_real = 0 # self.net_cls_B(self.real_B)
        self.cls_B_fake = self.net_cls_B(self.fake_B)
        self.cls_B_rec = 0 # self.net_cls_B(self.rec_B)

        self.cls_A_real = 0 # self.net_cls_A(self.real_A)
        self.cls_A_fake = self.net_cls_A(self.fake_A)
        self.cls_A_rec = 0 # self.net_cls_A(self.rec_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_B = self.net_G_AtoB(self.cat_con_feature(self.real_B, self.label_B))
            self.loss_idt_A = self.criterionIdt(self.idt_B, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_A = self.net_G_BtoA(self.cat_con_feature(self.real_A, self.label_A))
            self.loss_idt_B = self.criterionIdt(self.idt_A, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        D_B_fake = self.net_D_B(self.fake_B)
        self.loss_G_B = self.criterionGAN(D_B_fake, True)
        # GAN loss D_B(G_B(B))
        D_A_fake = self.net_D_A(self.fake_A)
        self.loss_G_A = self.criterionGAN(D_A_fake, True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        # classification loss
        # self.loss_cls_A = self.criterionCls(self.cls_A_real, self.label_A.long()) \
        #                   + self.criterionCls(self.cls_A_fake, self.label_A.long()) \
        #                   + self.criterionCls(self.cls_A_rec, self.label_A.long())
        #
        # self.loss_cls_B = self.criterionCls(self.cls_B_real, self.label_B.long()) \
        #                   + self.criterionCls(self.cls_B_fake, self.label_B.long()) \
        #                   + self.criterionCls(self.cls_B_rec , self.label_B.long())

        self.loss_cls_A = self.criterionCls(self.cls_A_fake, self.label_A.long())
        self.loss_cls_B = self.criterionCls(self.cls_B_fake, self.label_B.long())

        self.correct_batch_A = torch.sum(torch.argmax(self.cls_A_fake, dim=1) == self.label_A.long()).item()
        self.correct_batch_B = torch.sum(torch.argmax(self.cls_B_fake, dim=1) == self.label_B.long()).item()

        self.loss_G += self.loss_cls_A + self.loss_cls_B

        self.loss_G.backward()

    def backward_D_basic(self, netD, real, fake):

        # Real
        pred_real = netD(real)

        # Fake
        pred_fake = netD(fake.detach())

        # wasserestein GAN
        if self.opt.gan_loss_type == 'w':
            loss_D = -(torch.mean(pred_real) - torch.mean(pred_fake))
        else:

            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)

            # Combined loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_B = self.backward_D_basic(self.net_D_B, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_A = self.backward_D_basic(self.net_D_A, self.real_A, fake_A)

    def clip_D_grads(self):
        if self.opt.gan_loss_type == 'w':
            for p in self.net_D_A.parameters():
                p.data.clamp_(-0.01, 0.01)

            for p in self.net_D_B.parameters():
                p.data.clamp_(-0.01, 0.01)

    def optimize_parameters(self):
        # forward
        self.forward()

        # G_A and G_B
        self.set_requires_grad([self.net_D_A, self.net_D_B], False)
        self.optimizer_G.zero_grad()
        self.optimizer_cls.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_cls.step()

        for _ in range(self.opt.num_D_loops):
            # D_A and D_B
            self.set_requires_grad([self.net_D_A, self.net_D_B], True)
            self.optimizer_D.zero_grad()
            self.backward_D_A()
            self.backward_D_B()
            self.optimizer_D.step()

            self.clip_D_grads()

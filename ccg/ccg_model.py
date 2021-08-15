from ccg import networks
import torch
import torch.nn as nn
import itertools

def get_model(opt):
    model = CCGModel(opt)
    print(f'model [{model.get_model_name()}] was created')
    return model

class CCGModel:
    def get_model_name(self):
        return self.model_name

    def __init__(self, opt):
        self.model_name = 'Conditional CycleGAN'
        self.num_classes = opt.num_classes

        self.loss_names = ['D_A', 'D_B', 'G_AtoB', 'G_BtoA', 'cycle_A', 'cycle_B', 'idt_A', 'idt_B', 'cls_A', 'cls_B']

        if opt.mode_run == 'train':
            self.model_names = ['G_AtoB', 'G_BtoA', 'D_A', 'D_B', 'cls_A', 'cls_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_AtoB', 'G_BtoA']

        self.define_generators(opt)
        self.define_discriminators(opt)
        self.define_classifiers(opt)
        self.define_criterions(opt)
        self.define_optimizers(opt)
        self.opt = opt


    def define_generators(self, opt):
        activation_G = 'tanh'
        self.net_G_AtoB = networks.define_G(input_nc=opt.num_channels,
                                            output_nc=opt.num_channels,
                                            ngf=opt.ngf,
                                            which_model_netG=opt.model_name_G,
                                            norm=opt.norm_type,
                                            use_dropout=opt.use_dropout,
                                            init_type=opt.weight_init_type,
                                            gpu_ids=opt.gpu_ids,
                                            activation=activation_G)

        self.net_G_BtoA = networks.define_G(input_nc=opt.num_channels+opt.num_classes,
                                            output_nc=opt.num_channels,
                                            ngf=opt.ngf,
                                            which_model_netG=opt.model_name_G,
                                            norm=opt.norm_type,
                                            use_dropout=False,
                                            init_type=opt.weight_init_type,
                                            gpu_ids=opt.gpu_ids,
                                            activation=activation_G)

    def define_discriminators(self, opt):
        if opt.mode_run == 'train':
            activation_D = 'none' if opt.loss_name_D == 'wgan' else 'sigmoid'
            in_channels_D = opt.num_channels if (opt.model_name_D == 'internal_cond') else (opt.num_channels + opt.num_classes)
            internal_cond_D = True if (opt.model_name_D == 'internal_cond') else False
            num_classes = opt.num_classes if (opt.model_name_D == 'internal_cond') else False
            self.net_D_A = networks.define_D(input_nc=in_channels_D,
                                             ndf=opt.ndf,
                                             which_model_netD=opt.model_name_D,
                                             n_layers_D=opt.num_layers_D,
                                             norm=opt.norm_type,
                                             activation=activation_D,
                                             init_type=opt.weight_init_type,
                                             gpu_ids=opt.gpu_ids,
                                             internal_cond_D=internal_cond_D,
                                             num_classes=num_classes
                                             )

            self.net_D_B = networks.define_D(input_nc=opt.num_channels,
                                             ndf=opt.ndf,
                                             which_model_netD=opt.model_name_D,
                                             n_layers_D=opt.num_layers_D,
                                             norm=opt.norm_type,
                                             activation=activation_D,
                                             init_type=opt.weight_init_type,
                                             gpu_ids=opt.gpu_ids,
                                             internal_cond_D=internal_cond_D,
                                             num_classes=num_classes)
            # TODO implement image pool
            # self.pool_fake_A = ImagePool(opt.size_image_pool)
            # self.pool_fake_B = ImagePool(opt.size_image_pool)

    def define_classifiers(self, opt):
        if opt.mode_run == 'train':
            activation_cls = 'none'
            self.net_cls_A = networks.define_C(output_nc=opt.num_channels,
                                               num_classes=opt.num_classes,
                                               ndf=opt.ngf,
                                               init_type=opt.weight_init_type,
                                               gpu_ids=opt.gpu_ids,
                                               activation=activation_cls)

            self.net_cls_B = networks.define_C(output_nc=opt.num_channels,
                                               num_classes=opt.num_classes,
                                               ndf=opt.ngf,
                                               init_type=opt.weight_init_type,
                                               gpu_ids=opt.gpu_ids,
                                               activation=activation_cls)

    def define_criterions(self, opt):
        self.criterion_D = self.get_criterion_D(opt)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_cyc = nn.L1Loss()
        self.criterion_id = nn.L1Loss()

    def define_optimizers(self, opt):
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.net_G_AtoB.parameters(), self.net_G_BtoA.parameters()),
                                            lr=opt.lr,
                                            betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.net_D_A.parameters(), self.net_D_B.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_cls = torch.optim.Adam(itertools.chain(self.net_cls_A.parameters(), self.net_cls_B.parameters()),
                                              lr=opt.lr,
                                              betas=(opt.beta1, 0.999))

        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.optimizers.append(self.optimizer_cls)

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_criterion_D(self, opt):
        loss_D = None
        if opt.loss_name_D == 'wgan':
            # TODO implement wgan loss
            raise NotImplementedError('Need to implement WGAN loss')
        elif opt.loss_name_D == 'mse':
            loss_D = nn.MSELoss()
        elif opt.loss_name_D == 'bce':
            loss_D = nn.BCELoss()
        else:
            raise NotImplementedError('Invalid D loss')
        return loss_D

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].cuda()
        self.label_A = input['A_label' if AtoB else 'B_label'].cuda()

        self.real_B = input['B' if AtoB else 'A'].cuda()
        self.label_B = input['B_label' if AtoB else 'A_label'].cuda()

    def load_model(self, opt):
        # TODO implement load model
        pass

    def prepare_models(self, run_mode):
        if run_mode == 'train':
            self.net_D_A.train()
            self.net_D_B.train()
            self.net_G_AtoB.train()
            self.net_G_BtoA.train()
            self.net_cls_A.train()
            self.net_cls_B.train()
        else:
            self.net_D_A.eval()
            self.net_D_B.eval()
            self.net_G_AtoB.eval()
            self.net_G_BtoA.eval()
            self.net_cls_A.eval()
            self.net_cls_B.eval()

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
    def get_num_correct(self, pred, label):
        return torch.sum(torch.argmax(pred, dim=1) == label.long()).item()

    def optimize_parameters(self):

        # perform forward passes for both Generators
        # ------------------------------------------------------------------------------------------------------------
        # A -> fake_B
        self.fake_B = self.net_G_AtoB(self.real_A)
        # fake_B -> rec_A
        self.rec_A = self.net_G_BtoA(self.cat_label(self.fake_B, self.label_A))

        # TODO inspect the labels
        # B -> fake_A
        self.fake_A = self.net_G_BtoA(self.cat_label(self.real_B, self.label_A))
        # fake_A -> rec_B
        self.rec_B = self.net_G_AtoB(self.fake_A)

        # optimize discriminators
        # ------------------------------------------------------------------------------------------------------------
        self.optimizer_D.zero_grad()
        self.set_requires_grad([self.net_D_A, self.net_D_B], True)

        if self.opt.model_name_D == 'internal_cond':
            D_B_real = self.net_D_B(self.real_B, None)
            D_B_fake = self.net_D_B(self.fake_B.detach(), None)
            D_A_real_r = self.net_D_A(self.real_A, self.label_A)
            D_A_fake_r = self.net_D_A(self.fake_A.detach(), self.label_A)
            D_A_real_w = self.net_D_A(self.real_A, self.label_B) # WRONG??
        else:
            D_B_real = self.net_D_B(self.real_B)
            D_B_fake = self.net_D_B(self.fake_B.detach())
            D_A_real_r = self.net_D_A(self.cat_label(self.real_A, self.label_A))
            D_A_fake_r = self.net_D_A(self.cat_label(self.fake_A.detach(), self.label_A))
            D_A_real_w = self.net_D_A(self.cat_label(self.real_A, self.label_B))  # WRONG??

        # D_B loss
        # TODO deal with the mean
        self.loss_D_B = torch.mean(torch.log(D_B_real) + torch.log(1 - D_B_fake))
        self.loss_D_A = torch.mean(torch.log(D_A_real_r) + ((torch.log(1 - D_A_fake_r) + torch.log(1 - D_A_real_w)) / 2))
        self.loss_D = self.loss_D_A + self.loss_D_B
        self.loss_D.backward()
        self.optimizer_D.step()

        # optimize generators
        # ------------------------------------------------------------------------------------------------------------
        self.optimizer_G.zero_grad()
        self.set_requires_grad([self.net_D_A, self.net_D_B], False)

        # TODO this block is redundant
        if self.opt.model_name_D == 'internal_cond':
            # D_B_real = self.net_D_B(self.real_B, None)
            D_B_fake = self.net_D_B(self.fake_B.detach(), None)
            # D_A_real_r = self.net_D_A(self.real_A, self.label_A)
            D_A_fake_r = self.net_D_A(self.fake_A.detach(), self.label_A)
            # D_A_real_w = self.net_D_A(self.real_A, self.label_B) # WRONG??
        else:
            # D_B_real = self.net_D_B(self.real_B)
            D_B_fake = self.net_D_B(self.fake_B.detach())
            # D_A_real_r = self.net_D_A(self.cat_label(self.real_A, self.label_A))
            D_A_fake_r = self.net_D_A(self.cat_label(self.fake_A.detach(), self.label_A))
            # D_A_real_w = self.net_D_A(self.cat_label(self.real_A, self.label_B))  # WRONG??

        # cycle consistency loss
        # ------------------------------------------------------------------------------------------------------------
        self.cyc_A = self.criterion_cyc(self.fake_A, self.real_A)
        self.cyc_B = self.criterion_cyc(self.fake_B, self.real_B)
        self.cyc = self.cyc_A + self.cyc_B

        # classifier loss
        # TODO whether to classify both real and fake
        self.cls_A_real = self.net_cls_A(self.real_A)
        self.loss_cls_A_real = self.criterion_cls(self.cls_A_real, self.label_A)

        self.cls_A_fake = self.net_cls_A(self.fake_A)
        self.loss_cls_A_fake = self.criterion_cls(self.cls_A_fake, self.label_A)

        self.cls_B_real = self.net_cls_B(self.real_B)
        self.loss_cls_B_real = self.criterion_cls(self.cls_B_real, self.label_B)

        self.cls_B_fake = self.net_cls_B(self.fake_B)
        self.loss_cls_B_fake = self.criterion_cls(self.cls_B_fake, self.label_B)

        self.loss_cls_A = self.loss_cls_A_real + self.loss_cls_A_fake
        self.loss_cls_B = self.loss_cls_B_real + self.loss_cls_B_fake
        self.loss_cls = self.loss_cls_A + self.loss_cls_B

        # classifier num correct
        self.correct_cls_A = self.get_num_correct(self.cls_A_real, self.label_A) + self.get_num_correct(self.cls_A_fake, self.label_A)
        self.correct_cls_B = self.get_num_correct(self.cls_B_real, self.label_B) + self.get_num_correct(self.cls_B_fake, self.label_B)

        # generator loss
        # ------------------------------------------------------------------------------------------------------------
        # TODO deal with the mean
        self.loss_G_AtoB = torch.mean(torch.log(D_B_fake) + self.cyc)
        self.loss_G_BtoA = torch.mean(torch.log(D_A_fake_r) + self.cyc)
        self.loss_G = self.loss_G_AtoB + self.loss_G_BtoA
        self.loss_G += self.loss_cls
        self.loss_G.backward()
        self.optimizer_G.step()








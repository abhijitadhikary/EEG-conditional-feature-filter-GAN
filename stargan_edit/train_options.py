import argparse
import os
from utils import utils
import torch
from datetime import datetime


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
        self.initialize()

    def initialize(self):
        self.initialized = True

        load_model = False
        load_epoch = 10

        if load_model:
            timestamp = '23_Aug_2021__22_54_39'
        else:
            timestamp = datetime.now().strftime("%d_%b_%Y__%H_%M_%S")

        directory_name = 'stargan_edit'
        experiment_name = f'baseline_{timestamp}'
        self.parser.add_argument('--timestamp', type=str, default=f'{timestamp}', help='timestamp of current experiment')

        # paths
        self.parser.add_argument('--name_experiment', type=str, default=f'{experiment_name}', help='experiment name_experiment')
        self.parser.add_argument('--name_directory', type=str, default=f'{directory_name}', help='path to source code directory')
        self.parser.add_argument('--path_dataset', type=str, default=os.path.join('datasets', 'eeg'), help='path to dataset directory')
        self.parser.add_argument('--path_checkpoint', type=str, default=os.path.join(f'{directory_name}', 'checkpoints', f'{experiment_name}'), help='models are saved here')
        self.parser.add_argument('--path_save_image', type=str, default=os.path.join(f'{directory_name}', 'output', f'{experiment_name}'), help='directory to save the images')
        self.parser.add_argument('--path_tensorboard', type=str, default=os.path.join(f'{directory_name}', 'runs'), help='directory to save the images')

        # which mode to run the model
        self.parser.add_argument('--mode_run', type=str, default='train', help='train, val, test')
        self.parser.add_argument('--mode_split', type=str, default='within', help='within | across')

        # network architectures
        self.parser.add_argument('--model_name_D', type=str, default='internal_cond', help='basic | internal_cond -- name of discriminator')
        self.parser.add_argument('--model_name_G', type=str, default='resnet_9blocks', help='name of generator')
        self.parser.add_argument('--model_name_cls', type=str, default='baseline', help='which classifier to use')
        self.parser.add_argument('--loss_name_D', type=str, default='mse', help='| wgan, mse, bce| which loss to use for generator')
        self.parser.add_argument('--activation_D', type=str, default='none', help='sigmoid | tanh | None - Which activation function to use on the last layer of net_D')
        self.parser.add_argument('--activation_G', type=str, default='none', help='sigmoid | tanh | None - Which activation function to use on the last layer of net_D')
        # self.parser.add_argument('--internal_cond_D', type=bool, default=False, help='whether to insert condition in the middle of the discriminator')

        # key hyperparameters
        self.parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        self.parser.add_argument('--num_epochs', type=int, default=200, help='Total number of epochs to train')
        self.parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate for adam')
        self.parser.add_argument('--ngf', type=int, default=64, help='number of filters in the first generator layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='number of filters in the first discriminator layer')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--num_workers', default=4, type=int, help='number of workers for dataloader')
        self.parser.add_argument('--shuffle_dataloaders', default=True, type=bool, help='number of workers for dataloader')
        self.parser.add_argument('--num_loops_D', type=int, default=10, help='how many more times to train the discriminator than the generator')
        self.parser.add_argument('--min_value_feature', default=-1, type=int, help='minimum value to rescale input feature to')
        self.parser.add_argument('--max_value_feature', default=1, type=int, help='minimum value to rescale input feature to')
        self.parser.add_argument('--num_layers_D', default=3, type=int, help='number of discriminator layers')
        self.parser.add_argument('--num_resnet_blocks', default=3, type=int, help='number of resnet blocks')

        # loss multipliers
        self.parser.add_argument('--lambda_cyc', type=float, default=20.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_cls_real', type=float, default=10.0, help='weight for classifier loss (for real samples)')
        self.parser.add_argument('--lambda_cls_fake', type=float, default=5.0, help='weight for classifier loss (for fake samples)')
        self.parser.add_argument('--lambda_identity', type=float, default=0.5, help='weight for identity loss, only effective if the value is greater than 0')
        self.parser.add_argument('--size_image_pool', type=int, default=50, help='number of generated images to store in image pool')



        self.parser.add_argument('--epoch_start', type=int, default=0, help='start training from which epoch')

        # auxilary hyperparameters
        self.parser.add_argument('--image_size', type=int, default=32, help='resize images to this size')
        self.parser.add_argument('--num_classes', type=int, default=8, help='how many classes does the feature has (8)')
        self.parser.add_argument('--num_channels', type=int, default=3, help='how many channels does the feature has')
        self.parser.add_argument('--norm_type', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--lr_decay_type', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_start', type=int, default=100, help='from which epoch to start decaying learning rate')
        self.parser.add_argument('--use_dropout', type=bool, default=False, help='Whether to use dropout layers')
        self.parser.add_argument('--weight_init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--num_imsave', type=int, default=8, help='how many images to save in the image grid')
        self.parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name_experiment = opt.name_experiment + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')

        # # stargan
        # self.parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
        # self.parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
        # self.parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

        # save/load
        self.parser.add_argument('--load_model', type=bool, default=load_model, help='Whether to load pretrained model')
        self.parser.add_argument('--load_epoch', type=int, default=load_epoch, help='From which epoch to load model')
        self.parser.add_argument('--num_keep_best_ckpt', type=int, default=3, help='How many checkpoints to store')

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        # set gpu ids
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        args = vars(opt)

        print(f'------------ Options: {opt.mode_run}  -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name_experiment = opt.name_experiment + suffix

        # save to the disk
        expr_dir = os.path.join(opt.path_checkpoint)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        self.opt = opt
        return self.opt


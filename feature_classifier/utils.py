import os
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152



def create_model(self):
    # self.model_dict = {}
    if self.model_name == 'ResNet18':
        self.model = ResNet18(self.num_classes)
    elif self.model_name == 'ResNet34':
        self.model = ResNet34(self.num_classes)
    elif self.model_name == 'ResNet50':
        self.model = ResNet50(self.num_classes)
    elif self.model_name == 'ResNet101':
        self.model = ResNet101(self.num_classes)
    elif self.model_name == 'ResNet152':
        self.model = ResNet152(self.num_classes)
    else:
        raise NotImplementedError(f'model_name [{self.model_name}] not implemented.')

def create_dirs(self):
    dir_list = [
        ['.', 'feature_classifier'],
        ['.', 'feature_classifier', 'checkpoints'],
        ['.', 'feature_classifier', 'checkpoints', self.feature, self.model_name]
    ]

    for current_dir in dir_list:
        current_path = current_dir[0]
        if len(current_dir) > 1:
            for sub_dir_index in range(1, len(current_dir)):
                current_path = os.path.join(current_path, current_dir[sub_dir_index])
        if not os.path.exists(current_path):
            os.makedirs(current_path)

def remove_previous_checkpoints(self):
    all_files = os.listdir(self.checkpoint_path)
    if len(all_files) >= self.num_keep_best:
        current_full_path = os.path.join(self.checkpoint_path, all_files[0])
        os.remove(current_full_path)
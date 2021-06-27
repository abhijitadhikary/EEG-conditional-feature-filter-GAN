from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

def get_num_classes(self):
    if self.feature == 'alcoholism':
        self.num_classes = 2
    elif self.feature == 'stimulus':
        self.num_classes = 5
    elif self.feature == 'id':
        self.num_classes = 122
    else:
        raise ValueError(f'feature [{self.feature}] not recognized.')

def define_model(self):
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
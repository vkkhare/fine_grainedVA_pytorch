from datasets import *
from train_utils import *
from models import *

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
data_transforms = {
    'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    'val': transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
}
data_transforms_fine = {
    'train': transforms.Compose([
#             transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    'val': transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
}
# for training coarse classifier
options = {'base_lr': 1e-2,
           'batch_size': 64*2,
           'epochs': 600,
           'weight_decay': 1e-5}

dsets_coarse = {x: Dataset_Creator_IND('/data1/varun/assign2_split/',transform=data_transforms[x],test_tr_val=x).get_dataset()
             for x in ['train','val']}
data_loader_coarse = {x:torch.utils.data.DataLoader(dset,batch_size=options["batch_size"], 
                    shuffle=True, num_workers=8,pin_memory=True) for x,dset in dsets_coarse.items()}

coarse_class = dsets_coarse["train"].coarse_class

fine_class = {x: len(items) for x, items in dsets_coarse["train"].description_dict.items()}
# creating test, train, val for aircraft fine grained classifier
dsets_fine = {}
data_loaders_fine = {}
for x in ['train','val']:
    dsets_fine[x] = {y : Dataset_Creator_IND('/data1/varun/assign2_split/',transform=data_transforms_fine[x],
                                              test_tr_val=x,coarse=y).get_dataset() for y in coarse_class } 
    data_loaders_fine[x] = {y : torch.utils.data.DataLoader(dset,batch_size=int(options["batch_size"]/2), shuffle=True,
                                                            num_workers=8,pin_memory=True) for y,dset in dsets_fine[x].items()}

bcnn_module = BCNNWrapper(data_loader_coarse,data_loaders_fine,coarse_class,fine_class,options) 
train(bcnn_module)

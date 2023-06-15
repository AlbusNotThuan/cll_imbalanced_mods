from .cl_cifar import CLCIFAR10
from torch.utils.data import DataLoader, Dataset

def prepare_dataset(dataset, max_train_samples=None, multi_label=False, augment=False, imb_type=None, imb_factor=1.0):

    if dataset == "cifar10":
        trainset = CLCIFAR10(
            root="./data/cifar10",
            train=True,
            max_train_samples=max_train_samples,
            multi_label=multi_label,
            augment=augment,
            imb_type=imb_type,
            imb_factor=imb_factor,
        )
        testset = CLCIFAR10(root="./data/cifar10", train=False)
    else:
        raise NotImplementedError
    # import pdb
    # pdb.set_trace()
    return trainset, testset, trainset.input_dim, trainset.num_classes

class CLCustomDataset(Dataset):
    def __init__(self, data, targets, true_targets):
        self.data = data
        self.targets = targets
        self.true_targets = true_targets
        print('Init CLCustomDataset')
        breakpoint()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        print('getitem')
        breakpoint()
        return self.data[index], self.targets[index], self.true_targets[index]

# class CustomDataset(Dataset):
#     def __init__(self, data, targets):
#         self.data = data
#         self.targets = targets

#     def __len__(self):
#         return len(self.targets)
    
#     def __getitem__(self, index):
#         return self.data[index], self.targets[index]
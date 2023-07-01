# from .cl_cifar import CLCIFAR10
from .clcifar_cluster_label import CLCIFAR10
# from torch.utils.data import DataLoader, Dataset

def prepare_dataset(dataset, data_type, max_train_samples=None, multi_label=False, augment=False, imb_type=None, imb_factor=1.0):

    if dataset == "cifar10":
        if data_type == "train":
            dataset = CLCIFAR10(
                root="./data/cifar10",
                data_type=data_type,
                train=True,
                max_train_samples=max_train_samples,
                multi_label=multi_label,
                augment=augment,
                imb_type=imb_type,
                imb_factor=imb_factor,
            )
        else:
            dataset = CLCIFAR10(root="./data/cifar10", data_type = data_type, train=False)
    else:
        raise NotImplementedError
    return dataset, dataset.input_dim, dataset.num_classes

def prepare_cluster_dataset(dataset, data_type=None, kmean_cluster= None, max_train_samples=None, multi_label=False, augment=False, imb_type=None, imb_factor=1.0, pretrain=None):

    if dataset == "cifar10":
        if data_type == "train":
            dataset = CLCIFAR10(
                root="./data/cifar10",
                train=True,
                data_type=data_type,
                download=True,
                kmean_cluster=kmean_cluster,
                max_train_samples=max_train_samples,
                multi_label=multi_label,
                augment=augment,
                imb_type=imb_type,
                imb_factor=imb_factor,
                pretrain=pretrain
            )
        else:
            dataset = CLCIFAR10(root="./data/cifar10", train=False, data_type=data_type)
    else:
        raise NotImplementedError
    # import pdb
    # pdb.set_trace()
    return dataset, dataset.num_classes

# class CLCustomDataset(Dataset):
#     def __init__(self, data, targets, true_targets):
#         self.data = data
#         self.targets = targets
#         self.true_targets = true_targets
#         print('Init CLCustomDataset')
#         breakpoint()

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, index):
#         print('getitem')
#         breakpoint()
#         return self.data[index], self.targets[index], self.true_targets[index]

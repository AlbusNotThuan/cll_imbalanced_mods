# from .cl_cifar import CLCIFAR10
from .clcifar_cluster_label import CLCIFAR20, CLCIFAR10, CLCIFAR100
from .clmnist_cluster_label import CLMNIST, CLFashionMNIST, CLKMNIST
from .clcifar_nn_label import NCLCIFAR10, NCLCIFAR20, NCLCIFAR100
from .clmnist_nn_label import NCLMNIST
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

def prepare_neighbour_dataset(input_dataset, data_type=None, max_train_samples=None, multi_label=False, weight=None, imb_type=None, imb_factor=1.0, pretrain=None):
    if input_dataset == "CIFAR10":
        if data_type == "train":
            dataset = NCLCIFAR10(
                root="./data/cifar10",
                train=True,
                data_type=data_type,
                download=True,
                max_train_samples=max_train_samples,
                multi_label=multi_label,
                weight=weight,
                imb_type=imb_type,
                imb_factor=imb_factor,
                pretrain=pretrain,
                input_dataset=input_dataset,
            )
        else:
            dataset = NCLCIFAR10(root="./data/cifar10", train=False, data_type=data_type, input_dataset=input_dataset)
    elif input_dataset == "CIFAR20":
        if data_type == "train":
            dataset = NCLCIFAR20(
                root="./data/cifar20",
                train=True,
                data_type=data_type,
                download=True,
                max_train_samples=max_train_samples,
                multi_label=multi_label,
                weight=weight,
                imb_type=imb_type,
                imb_factor=imb_factor,
                pretrain=pretrain,
                input_dataset=input_dataset,
            )
        else:
            dataset = NCLCIFAR20(root="./data/cifar20", train=False, data_type=data_type, input_dataset=input_dataset)
    elif input_dataset == "MNIST":
        if data_type == "train":
            dataset = NCLMNIST(
                root="./data/mnist",
                train=True,
                data_type=data_type,
                download=True,
                max_train_samples=max_train_samples,
                multi_label=multi_label,
                weight=weight,
                imb_type=imb_type,
                imb_factor=imb_factor,
                pretrain=pretrain,
                input_dataset=input_dataset,
            )
        else:
            dataset = NCLMNIST(root="./data/mnist", train=False, data_type=data_type, input_dataset=input_dataset)
    else:
        raise NotImplementedError
    
    return dataset, dataset.input_dim, dataset.num_classes

def prepare_cluster_dataset(input_dataset, data_type=None, kmean_cluster= None, max_train_samples=None, multi_label=False, augment=False, imb_type=None, imb_factor=1.0, pretrain=None):
    if input_dataset == "CIFAR10":
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
                pretrain=pretrain,
                input_dataset=input_dataset,
            )
        else:
            dataset = CLCIFAR10(root="./data/cifar10", train=False, data_type=data_type, input_dataset=input_dataset)
    elif input_dataset == "CIFAR20":
        if data_type == "train":
            dataset = CLCIFAR20(
                root="./data/cifar20",
                train=True,
                data_type=data_type,
                download=True,
                kmean_cluster=kmean_cluster,
                max_train_samples=max_train_samples,
                multi_label=multi_label,
                augment=augment,
                imb_type=imb_type,
                imb_factor=imb_factor,
                pretrain=pretrain,
                input_dataset=input_dataset,
            )
        else:
            dataset = CLCIFAR20(root="./data/cifar20", train=False, data_type=data_type, input_dataset=input_dataset)
    elif input_dataset == "MNIST":
        if data_type == "train":
            dataset = CLMNIST(
                root="./data/mnist",
                train=True,
                data_type=data_type,
                download=True,
                kmean_cluster=kmean_cluster,
                max_train_samples=max_train_samples,
                multi_label=multi_label,
                augment=augment,
                imb_type=imb_type,
                imb_factor=imb_factor,
                pretrain=pretrain,
                input_dataset=input_dataset,
            )
        else:
            dataset = CLMNIST(root="./data/mnist", train=False, data_type=data_type, input_dataset=input_dataset)
    elif input_dataset == "FashionMNIST":
        if data_type == "train":
            dataset = CLFashionMNIST(
                root="./data/FashionMNIST",
                train=True,
                data_type=data_type,
                download=True,
                kmean_cluster=kmean_cluster,
                max_train_samples=max_train_samples,
                multi_label=multi_label,
                augment=augment,
                imb_type=imb_type,
                imb_factor=imb_factor,
                pretrain=pretrain,
                input_dataset=input_dataset,
            )
        else:
            dataset = CLFashionMNIST(root="./data/FashionMNIST", train=False, data_type=data_type, input_dataset=input_dataset)

    elif input_dataset == "KMNIST":
        if data_type == "train":
            dataset = CLKMNIST(
                root="./data/KMNIST",
                train=True,
                data_type=data_type,
                download=True,
                kmean_cluster=kmean_cluster,
                max_train_samples=max_train_samples,
                multi_label=multi_label,
                augment=augment,
                imb_type=imb_type,
                imb_factor=imb_factor,
                pretrain=pretrain,
                input_dataset=input_dataset,
            )
        else:
            dataset = CLKMNIST(root="./data/KMNIST", train=False, data_type=data_type, input_dataset=input_dataset)
    else:
        raise NotImplementedError
    
    return dataset, dataset.input_dim, dataset.num_classes

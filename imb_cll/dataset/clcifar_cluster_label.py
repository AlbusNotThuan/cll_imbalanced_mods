from PIL import Image
import os
import os.path
import numpy as np
from scipy import sparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.models import resnet18
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
import copy
from .base_dataset import BaseDataset
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

class CLCIFAR10(VisionDataset, BaseDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self,
        root="../data/cifar10",
        train=True,
        data_type=None,
        transform=None,
        validate=False,
        target_transform=None,
        download=True,
        kmean_cluster=None,
        max_train_samples=None,
        multi_label=False,
        augment=False,
        imb_type=None,
        imb_factor=1.0,
        pretrain=None,
        seed=1126
    ):
        self.data_type = data_type
        self.num_classes = 10
        self.input_dim = 3 * 32 * 32
        self.multi_label = multi_label
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.kmean_cluster = kmean_cluster # Number of clustering with K mean method.

        super(CLCIFAR10, self).__init__(
            root, train, transform, target_transform)
        
        self.train = train
        self.validate = validate
        self.pretrain = pretrain
        self.seed = seed

        if seed is None:
            raise RuntimeError('Seed is not specified.')

        if self.data_type == "train" and imb_factor > 0 and not imb_type in ["exp", "step"]:
            raise RuntimeError(f'Imb_type method {imb_type} is invalid.')
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.data_type in ("train", "val"):
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.data_type =="train":
            if self.imb_type is not None:
                self.img_num_list = self.get_img_num_per_cls(self.num_classes, self.imb_type, self.imb_factor)
                self.gen_imbalanced_data(self.img_num_list)
            
            if max_train_samples: #limit the size of the training dataset to max_train_samples
                train_len = min(len(self.data), max_train_samples)
                self.data = self.data[:train_len]
                self.targets = self.targets[:train_len]

            self.gen_complementary_target()
        
        # self.rng = np.random.default_rng(self.seed)
        # self.idx = self.rng.permutation(len(self.data))
        # self.idx = self.rng.permutation(len(self.data))
        # self.idx_train = self.idx[range(0, 12406)]
        # self.idx_train = self.idx[:12406]

        self.idx_train = len(self.data)
        # print("The range of index {}".format(self.idx_train[:10]))

        self.mean, self.std = [0.4914, 0.4822, 0.4465], [0.247,  0.2435, 0.2616]

        if self.data_type == "train" and not validate:
            if augment:
                self.transform=Compose([
                    RandomHorizontalFlip(),
                    RandomCrop(32, 4, padding_mode='reflect'),
                    ToTensor(),
                    Normalize(mean=self.mean, std=self.std),
                ])
            else:
                self.transform=Compose([
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])

        else:
            self.transform=Compose([
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])

        self._load_meta()
        if self.data_type =="train":
            self.true_targets = self.features_space()
            print("Done: K_Mean Cluster")

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.data_type == "train":
            img, target, true_target = self.data[index], self.targets[index], self.true_targets[index]
        
        if self.data_type == "test":
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.data_type == "train":
            return img, target, true_target
        else:
            return img, target
    
    def __len__(self):
        if self.data_type == "train":
            return len(self.data)
        else:
            return len(self.data)
    
    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    @torch.no_grad()
    def features_space(self):
        if self.data_type == "train":
            # pretrain = "../CIFAR10_checkpoint_0799_-0.7960.pth.tar"
            model_simsiam = resnet18()
            model_simsiam.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model_simsiam.maxpool = nn.Identity()

            transform=Compose([
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])
            ###NOTED: Need to create imbalanced dataset first and get the idx of training
            tensor = torch.stack([transform(self.data[i]) for i in range(0, self.idx_train)])  
            ds = torch.utils.data.TensorDataset(tensor)
            dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False)
            print(self.pretrain)

            checkpoint = torch.load(self.pretrain, map_location="cpu")
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            model_simsiam.load_state_dict(state_dict, strict=False)
            model_simsiam.fc = nn.Identity()
            model_simsiam.cpu()

            features = []

            model_simsiam.eval()
            for input in dl:
                # features.append(F.normalize(model_simsiam(input.cpu())).cpu())
                features.append(F.normalize(model_simsiam(torch.cat(input).cpu())).cpu())
            features = torch.cat(features, dim=0).cpu().detach().numpy()

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.kmean_cluster, random_state=42)
        cluster_labels = kmeans.fit_predict(features)

        classes, class_counts = np.unique(cluster_labels, return_counts=True)
        sorted_list = sorted(class_counts, reverse=True)
        print("The number of each sample into each cluster is {}".format(sorted_list))

        return cluster_labels
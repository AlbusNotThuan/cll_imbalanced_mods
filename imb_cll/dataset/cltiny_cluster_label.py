from PIL import Image
import os
import os.path
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.models import resnet18
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandAugment, Resize
from .base_dataset import BaseDataset
from imb_cll.utils.autoaugment import AutoAugment, Cutout


class CLTiny200(VisionDataset, BaseDataset):
    """TinyImageNet-200 Dataset for Complementary Label Learning.
    
    TinyImageNet contains 200 classes from ImageNet, each with 500 training images,
    50 validation images, and 50 test images. Images are 64x64 pixels.
    
    Args:
        root (string): Root directory of dataset where directory
            ``tiny-imagenet-200`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from validation set.
        data_type (str): Type of data - 'train' or 'test'
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet.
    """
    
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    base_folder = "tiny-imagenet-200"
    
    def __init__(self,
        root=None,
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
        seed=1126,
        input_dataset=None,
        transition_bias=1.0,
        setup_type=None,
        aug_type=None,
        cll_type='random',
        noise=False,
        transition_matrix=None,
        pretrained_mode=0,
        ba_config=None,
        mi_config=None,
        ord_num=None
    ):
        self.root = root
        self.data_type = data_type
        self.num_classes = 200
        self.input_dim = 3 * 64 * 64
        self.multi_label = multi_label
        self.input_dataset = input_dataset
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.kmean_cluster = kmean_cluster
        self.transition_bias = transition_bias
        self.setup_type = setup_type
        self.cll_type = cll_type
        self.noise = noise
        self.transition_matrix = transition_matrix
        self.dataset_name = "Tiny200"
        self.pretrained_mode = pretrained_mode
        
        # Store BA and MI configurations
        self.ba_config = ba_config
        self.mi_config = mi_config
        self.ord_num = ord_num

        super(CLTiny200, self).__init__(
            root, transform, target_transform)
        
        self.train = train
        self.validate = validate
        self.pretrain = pretrain
        self.seed = seed
        self.max_train_samples = max_train_samples

        if seed is None:
            raise RuntimeError('Seed is not specified.')

        if self.data_type == "train" and imb_factor > 0 and not imb_type in ["exp", "step", None]:
            raise RuntimeError(f'Imb_type method {imb_type} is invalid.')
        
        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        # Load class names and create class_to_idx mapping
        self._load_class_names()
        
        # Load the data
        self.data = []
        self.targets = []
        
        if self.data_type in ("train", "val"):
            self._load_train_data()
        else:
            self._load_val_data()  # Use validation as test set
        
        self.data = np.array(self.data)
        
        if self.data_type == "train":
            if self.imb_type is not None and self.imb_factor < 1.0:
                self.img_num_list, self.img_max = self.get_img_num_per_cls(self.num_classes, self.imb_type, self.imb_factor)
                self.gen_imbalanced_data(self.img_num_list)
                print("Done: Generate imbalanced data")
            else:
                self.img_max = len(self.data) / self.num_classes

            if self.max_train_samples:
                train_len = min(len(self.data), self.max_train_samples)
                self.data = self.data[:train_len]
                self.targets = self.targets[:train_len]
                print(f"Training dataset limited to {train_len} samples.")

            # Only generate ordinary/CLL split if ord_num is specified (for comb-oc)
            if self.ord_num is not None and self.ord_num > 0:
                self.gen_few_ordinary_target()
                print(f"Number of ordinary samples per class: {self.ord_num}")
            
            if self.setup_type == "setup 1":
                self.gen_complementary_target()
            elif self.setup_type == "setup 2":
                self.gen_bias_complementary_label()
            elif self.setup_type in ["transition_matrix", "Dbar[prompt]_T", "Dbar[prompt]_T[prompt]"]:
                print("Using Dbar[prompt]")
                self.generate_cl_from_matrix(
                    self.transition_matrix,
                    ba_config=self.ba_config,
                    mi_config=self.mi_config
                )
            elif self.setup_type in ["Dbar_T[prompt]", "Dbar_T"]:
                print("Using Dbar")
                self.gen_complementary_target()
        
        self.rng = np.random.default_rng(self.seed)
        self.idx = self.rng.permutation(len(self.data))
        self.idx_train = len(self.data)

        # TinyImageNet normalization values (ImageNet statistics are commonly used)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Set up transforms
        if self.data_type == "train" and not validate:
            if augment:
                if aug_type == "randaug":
                    self.transform = Compose([
                        RandAugment(3, 5),
                        RandomHorizontalFlip(),
                        RandomCrop(64, 8, padding_mode='reflect'),
                        ToTensor(),
                        Normalize(mean=self.mean, std=self.std),
                    ])
                elif aug_type == "autoaug":
                    self.transform = Compose([
                        RandomHorizontalFlip(),
                        RandomCrop(64, 8, padding_mode='reflect'),
                        AutoAugment(),
                        ToTensor(),
                        Normalize(mean=self.mean, std=self.std),
                    ])
                elif aug_type == "cutout":
                    self.transform = Compose([
                        RandomHorizontalFlip(),
                        RandomCrop(64, 8, padding_mode='reflect'),
                        Cutout(),
                        ToTensor(),
                        Normalize(mean=self.mean, std=self.std),
                    ])
                elif aug_type == "flipflop":
                    self.transform = Compose([
                        RandomHorizontalFlip(),
                        RandomCrop(64, 8, padding_mode='reflect'),
                        ToTensor(),
                        Normalize(mean=self.mean, std=self.std),
                    ])
            else:
                self.transform = Compose([
                    RandomCrop(64, padding=8),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize(mean=self.mean, std=self.std),
                ])
        else:
            self.transform = Compose([
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])

        if self.data_type == "train":
            if self.kmean_cluster != 0:
                self.k_mean_targets = self.features_space()
                print("Done: K_Mean Cluster")

    def _load_class_names(self):
        """Load class name to ID mapping from wnids.txt and words.txt"""
        # Load WordNet IDs (class identifiers)
        wnids_path = os.path.join(self.root, self.base_folder, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            self.wnids = [line.strip() for line in f.readlines()]
        
        # Create mapping from WordNet ID to class index
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        
        # Load human-readable class names from words.txt
        words_path = os.path.join(self.root, self.base_folder, 'words.txt')
        self.wnid_to_words = {}
        with open(words_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    wnid = parts[0]
                    words = parts[1]
                    self.wnid_to_words[wnid] = words
        
        # Create class names list (in order of class index)
        self.classes = []
        for wnid in self.wnids:
            if wnid in self.wnid_to_words:
                # Take only the first word/phrase before comma for cleaner names
                class_name = self.wnid_to_words[wnid].split(',')[0].strip()
                self.classes.append(class_name)
            else:
                self.classes.append(wnid)  # Fallback to wnid if no word found
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def _load_train_data(self):
        """Load training data from the train folder"""
        train_dir = os.path.join(self.root, self.base_folder, 'train')
        
        for wnid in self.wnids:
            class_dir = os.path.join(train_dir, wnid, 'images')
            class_idx = self.wnid_to_idx[wnid]
            
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.JPEG'):
                        img_path = os.path.join(class_dir, img_name)
                        img = Image.open(img_path).convert('RGB')
                        img_array = np.array(img)
                        self.data.append(img_array)
                        self.targets.append(class_idx)
        
        print(f"Loaded {len(self.data)} training samples")

    def _load_val_data(self):
        """Load validation data (used as test set)"""
        val_dir = os.path.join(self.root, self.base_folder, 'val')
        val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')
        
        # Parse validation annotations
        val_annotations = {}
        with open(val_annotations_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                img_name = parts[0]
                wnid = parts[1]  # This is the WordNet ID
                val_annotations[img_name] = wnid
        
        # Load validation images
        val_images_dir = os.path.join(val_dir, 'images')
        for img_name, wnid in val_annotations.items():
            if wnid in self.wnid_to_idx:
                img_path = os.path.join(val_images_dir, img_name)
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    img_array = np.array(img)
                    self.data.append(img_array)
                    self.targets.append(self.wnid_to_idx[wnid])
        
        print(f"Loaded {len(self.data)} validation/test samples")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.data_type == "train":
            img, targets, true_targets, k_mean_targets = (
                self.data[index], 
                self.targets[index], 
                self.true_targets[index], 
                self.k_mean_targets[index]
            )

            # Get label type: 1 = ordinary, 0 = complementary
            if hasattr(self, 'label_type'):
                label_type = self.label_type[index]
            else:
                label_type = 0  # Default: all complementary
        
        if self.data_type == "test":
            img, targets = self.data[index], self.targets[index]

        # Convert to PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        if self.data_type == "train":
            return img, targets, true_targets, k_mean_targets, self.img_max, label_type
        else:
            return img, targets
    
    def __len__(self):
        return len(self.data)
    
    def _check_exists(self):
        """Check if the dataset folder exists"""
        return os.path.exists(os.path.join(self.root, self.base_folder))
    
    def _download(self):
        """Download the TinyImageNet dataset if it doesn't exist"""
        if self._check_exists():
            print('Files already downloaded and verified')
            return
        
        os.makedirs(self.root, exist_ok=True)
        download_and_extract_archive(self.url, self.root, filename=self.filename)
        print("Downloaded and extracted TinyImageNet-200")

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    @torch.no_grad()
    def features_space(self):
        if self.data_type == "train":
            model_simsiam = resnet18()
            num_channel = 3
            model_simsiam.conv1 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)

            transform = Compose([
                Resize((64, 64)),
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])
            
            tensor = torch.stack([transform(Image.fromarray(self.data[i])) for i in range(0, self.idx_train)])
            ds = torch.utils.data.TensorDataset(tensor)
            dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False)
            print(self.pretrain)

            if self.pretrain:
                checkpoint = torch.load(self.pretrain, map_location="cuda", weights_only=False)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                for k in list(state_dict.keys()):
                    if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                        state_dict[k[len("module.encoder."):]] = state_dict[k]
                    del state_dict[k]
                model_simsiam.load_state_dict(state_dict, strict=False)
            
            model_simsiam.fc = nn.Identity()
            model_simsiam.cpu()

            features = []
            for batch in dl:
                x = batch[0]
                feat = model_simsiam(x)
                features.append(feat.cpu().numpy())
            
            features = np.vstack(features)
            
            if self.kmean_cluster and self.kmean_cluster > 0:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=self.kmean_cluster, random_state=self.seed)
                cluster_labels = kmeans.fit_predict(features)
                return cluster_labels
            
            return self.targets.copy() if isinstance(self.targets, list) else self.targets.tolist()
        
        return []

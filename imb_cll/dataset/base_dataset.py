import numpy as np
import copy
import torch
import torch.nn.functional as F


class BaseDataset:
    def gen_complementary_target(self):
        # This will NOT shuffle the dataset
        np.random.seed(1126)
        self.true_targets = copy.deepcopy(self.targets)
        self.targets = [
            np.random.choice(
                [j for j in range(self.num_classes) if j != self.targets[i]],
                3 if self.multi_label else 1,
                False,
            ) # generates new complementary target values for each original target value
            for i in range(len(self.targets))
        ]

    #        Q = [[0 for i in range(self.num_classes)] for i in range(self.num_classes)]
    #        for i in range(len(self.true_targets)):
    #            Q[self.true_targets[i]][int(self.targets[i][0])] += 1
    #        Q = torch.Tensor(Q)
    #        V = torch.sum(Q, dim=1, keepdim=True)
    #        Q = Q.div(V)
    #        print(Q)

    def estimate_Q(self, module, model_path):
        module.load_state_dict(torch.load(model_path))
        rng = np.random.default_rng(seed=1126)
        idx = rng.permutation(len(self.true_targets))
        anchor_set = [[] for i in range(self.num_classes)]
        for i in range(len(idx)):
            if len(anchor_set[self.true_targets[idx[i]]]) < 10:
                anchor_set[self.true_targets[idx[i]]].append(
                    self.__getitem__(idx[i])[0]
                )
        Q = torch.zeros((self.num_classes, self.num_classes))
        for i, anchor in enumerate(anchor_set):
            x = torch.stack(anchor).float()
            output = module(x)
            output = F.softmax(output, dim=1)
            Q[i] += output.mean(dim=0)
        # print(Q)
        return Q
    

    # Base dataset for creating imbalanced dataset
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        # cifar10, cifar100, svhn
        if hasattr(self, "data"):
            img_max = len(self.data) / cls_num
        # cinic10, tiny-imagenet
        elif hasattr(self, "samples"):
            img_max = len(self.samples) / cls_num
        else:
            raise AttributeError("[Warning] Check your data or customize !")
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([
                the_class,
            ] * the_img_num)
        new_data = np.vstack(new_data)
        assert new_data.shape[0] == len(new_targets)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

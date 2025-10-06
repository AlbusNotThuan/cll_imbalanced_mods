import numpy as np
import copy
import torch
import torch.nn.functional as F
import pdb
import os


class BaseDataset:
    def gen_complementary_target(self):
        # This will NOT shuffle the dataset

        print("Generating complementary targets...")

        np.random.seed(1126)
        self.true_targets = copy.deepcopy(self.targets)
        self.k_mean_targets = copy.deepcopy(self.targets)

        if self.cll_type == 'random':
            print("Using random complementary labels")
            self.targets = [
                np.random.choice(
                    [j for j in range(self.num_classes) if j != self.targets[i]],
                    3 if self.multi_label else 1,
                    False,
                ) # generates new complementary target values for each original target value
                for i in range(len(self.targets))
            ]
            
            # Save generated labels to file
            dataset_type = getattr(self, 'dataset_name', 'CIFAR10')
            organized_file_path = f"generated_labels/{dataset_type.lower()}/{self.cll_type}.txt"
            os.makedirs(f"generated_labels/{dataset_type.lower()}", exist_ok=True)
            with open(organized_file_path, 'w') as f:
                for label in self.targets:
                    if isinstance(label, np.ndarray):
                        f.write(f"array([{label[0]}])\n")
                    else:
                        f.write(f"array([{label}])\n")
            print(f"Saved {len(self.targets)} {self.cll_type} complementary labels to {organized_file_path}")


        elif self.cll_type in ['least', 'most', 'most_no_noise', 'from_matrix_least', 'from_matrix_most', 'most+rand', 'least+rand']:
            # Use image predictor based on dataset type
            from imb_cll.utils.image_predictor import create_predictor

            # if self.cll_type in ['from_matrix_least']: self.cll_type = 'least'
            # if self.cll_type in ['from_matrix_most']: self.cll_type = 'most'

            if self.cll_type in ['most+rand']: self.cll_type = 'most'
            if self.cll_type in ['least+rand']: self.cll_type = 'least'

            # Switch cases for training dataset generated from pretrained model and specific transition matrix
            # This is for 2 cases: Dbar and Dbar[prompt]
            # Determine label filename for 2 cases: Dbar_T and Dbar[prompt]_T[prompt]
            label_filename = (
                self.cll_type
            )
            
            # Determine dataset type
            dataset_type = getattr(self, 'dataset_name', 'CIFAR10')
            
            # Set noise flag based on mode
            noise_flag = (self.cll_type == 'most_no_noise')
            
            # Check if labels already exist in organized structure
            organized_file_path = f"generated_labels/{dataset_type.lower()}/{label_filename}.txt"
            if os.path.exists(organized_file_path):
                print(f"Loading {self.cll_type} complementary labels from {organized_file_path}")
                with open(organized_file_path, 'r') as f:
                    content = f.read().strip()
            else:
                print(f"Generating {self.cll_type} complementary labels for {dataset_type}...")
                print(f"Using image predictor type: {self.cll_type} for dataset: {dataset_type}")
                self.image_predictor = create_predictor(
                    device=torch.device('cuda:0'), 
                    mode=self.cll_type if self.cll_type != 'most_no_noise' else 'most',
                    debug=False, 
                    noise=noise_flag,
                    dataset_type=dataset_type,
                    pretrained_mode=self.pretrained_mode
                )
                
                # Generate new labels
                generated_labels = []
                for i in range(len(self.targets)):
                    self.image_predictor.set_true_label(self.true_targets[i])
                    predicted = self.image_predictor.predict_single_image(self.data[i])
                    generated_labels.append(np.array([predicted['predicted_class']]))
                self.targets = generated_labels

                # Save to organized structure
                os.makedirs(f"generated_labels/{dataset_type.lower()}", exist_ok=True)
                with open(organized_file_path, 'w') as f:
                    for label in generated_labels:
                        f.write(f"array([{label[0]}])\n")
                
                print(f"Saved {len(generated_labels)} {self.cll_type} complementary labels to {organized_file_path}")

                pdb.set_trace()
                return  # Skip parsing since we already have the labels
            
            # Parse the string representation of arrays
            import re
            pattern = r'array\(\[(\d+)\]\)'
            matches = re.findall(pattern, content)
            
            self.targets = [np.array([int(match)]) for match in matches]
            print(f"Loaded {len(self.targets)} {self.cll_type} complementary labels")
        
        elif self.cll_type in ['bias_most', 'bias_least', 'bias_random']:
            print(f"Using {self.cll_type} complementary labels")
            
            # Determine dataset type
            dataset_type = getattr(self, 'dataset_name', 'CIFAR10').upper()
            
            # Define bias mappings based on dataset and bias type
            if dataset_type == 'CIFAR10':
                if self.cll_type == 'bias_most':
                    bias_mapping = {
                        0: 8, 1: 9, 2: 0, 3: 5, 4: 3, 
                        5: 3, 6: 3, 7: 5, 8: 0, 9: 1
                    }
                elif self.cll_type == 'bias_least':
                    bias_mapping = {
                        0: 6, 1: 4, 2: 1, 3: 2, 4: 2,
                        5: 8, 6: 7, 7: 8, 8: 7, 9: 4
                    }
                elif self.cll_type == 'bias_random':
                    # Create a deterministic random mapping where each true label maps to a unique CL label
                    np.random.seed(42)
                    bias_mapping = {}
                    # Simple approach: create a random permutation and use it as mapping
                    all_labels = list(range(10))
                    np.random.shuffle(all_labels)
                    for true_label in range(10):
                        # Find the next available label that's not the true label
                        for candidate in all_labels:
                            if candidate != true_label:
                                bias_mapping[true_label] = candidate
                                all_labels.remove(candidate)
                                break
                        # If we run out of shuffled labels, use any remaining valid label
                        if true_label not in bias_mapping:
                            remaining = [x for x in range(10) if x != true_label and x not in bias_mapping.values()]
                            if remaining:
                                bias_mapping[true_label] = remaining[0]
                            
            elif dataset_type == 'CIFAR20':
                if self.cll_type == 'bias_most':
                    raise NotImplementedError("CIFAR-20 bias_most mapping not yet implemented")
                elif self.cll_type == 'bias_least':
                    raise NotImplementedError("CIFAR-20 bias_least mapping not yet implemented")
                elif self.cll_type == 'bias_random':
                    np.random.seed(42)
                    bias_mapping = {}
                    all_labels = list(range(20))
                    np.random.shuffle(all_labels)
                    for true_label in range(20):
                        for candidate in all_labels:
                            if candidate != true_label:
                                bias_mapping[true_label] = candidate
                                all_labels.remove(candidate)
                                break
                        if true_label not in bias_mapping:
                            remaining = [x for x in range(20) if x != true_label and x not in bias_mapping.values()]
                            if remaining:
                                bias_mapping[true_label] = remaining[0]
                            
            elif dataset_type == 'CIFAR100':
                if self.cll_type == 'bias_most':
                    raise NotImplementedError("CIFAR-100 bias_most mapping not yet implemented")
                elif self.cll_type == 'bias_least':
                    raise NotImplementedError("CIFAR-100 bias_least mapping not yet implemented")
                elif self.cll_type == 'bias_random':
                    np.random.seed(42)
                    bias_mapping = {}
                    all_labels = list(range(100))
                    np.random.shuffle(all_labels)
                    for true_label in range(100):
                        for candidate in all_labels:
                            if candidate != true_label:
                                bias_mapping[true_label] = candidate
                                all_labels.remove(candidate)
                                break
                        if true_label not in bias_mapping:
                            remaining = [x for x in range(100) if x != true_label and x not in bias_mapping.values()]
                            if remaining:
                                bias_mapping[true_label] = remaining[0]
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
            self.targets = [
                np.array([bias_mapping[self.targets[i]]])
                for i in range(len(self.targets))
            ]
            print(f"Applied {self.cll_type} mapping to {len(self.targets)} labels")
            
            # Save generated labels to file
            organized_file_path = f"generated_labels/{dataset_type.lower()}/{self.cll_type}.txt"
            os.makedirs(f"generated_labels/{dataset_type.lower()}", exist_ok=True)
            with open(organized_file_path, 'w') as f:
                for label in self.targets:
                    f.write(f"array([{label[0]}])\n")
            print(f"Saved {len(self.targets)} {self.cll_type} complementary labels to {organized_file_path}")

        pdb.set_trace()

        # T = np.array(torch.full([self.num_classes, self.num_classes], 1/(self.num_classes -1)))
        # for i in range(self.num_classes):
        #     T[i][i] = 0
        
        # for i in range(len(self.targets)):
        #     self.ord_labels = self.targets[i]
        #     self.targets[i] = np.random.choice(list(range(self.num_classes)), p=T[self.ord_labels])
        
    # Q = [[0 for i in range(self.num_classes)] for i in range(self.num_classes)]
    # for i in range(len(self.true_targets)):
    #     Q[self.true_targets[i]][int(self.targets[i][0])] += 1
    # Q = torch.Tensor(Q)
    # V = torch.sum(Q, dim=1, keepdim=True)
    # Q = Q.div(V)
    # print(Q)

    def gen_bias_complementary_label(self):
        cls_num = self.num_classes
        transition_bias = 1/self.transition_bias
        weight_max = 100
        img_num_per_cls = []

        for cls_idx in range(cls_num):
            num = weight_max * (transition_bias**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))

        T_bias = img_num_per_cls.copy()
        for i in range(cls_num - 1):
            T_bias =  np.vstack((T_bias, img_num_per_cls))
        for i in range(cls_num):
            T_bias[i][i] = 0.0

        # Need to add dtype=float, otherwise gets all 0 matrix
        T_bias = np.array(T_bias, dtype=float)
        for i in range(cls_num):
            T_bias[i, :] = T_bias[i, :] / np.sum(T_bias[i, :])

        np.random.seed(1126)
        self.true_targets = copy.deepcopy(self.targets)
        self.k_mean_targets = copy.deepcopy(self.targets)
        for i in range(len(self.targets)):
            self.ord_labels = self.targets[i]
            self.targets[i] = np.random.choice(list(range(cls_num)), p=T_bias[self.ord_labels])

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
        # cifar10, cifar100, svhn, mnist
        if hasattr(self, "data"):
            img_max = len(self.data) / cls_num
            # check for mnist, just take 5900 samples for maximum
            if self.input_dataset == "MNIST":
                img_max = 5000
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
        print("The number samples of each class: {}".format(img_num_per_cls))
        return img_num_per_cls, img_max

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
        print(new_data.shape[0], len(new_targets))
        assert new_data.shape[0] == len(new_targets)
        self.data = new_data
        self.targets = new_targets
    
    def generate_cl_from_matrix(self, transition_matrix):
        """
        Generates complementary labels based on a given transition matrix.
        :param transition_matrix: A numpy array of shape (num_classes, num_classes)
                                 Row sums should equal 1, diagonal elements can be non-zero
        """
        print("Generating complementary labels from transition matrix...")
        
        num_classes = self.num_classes
        if transition_matrix.shape != (num_classes, num_classes):
            raise ValueError(f"Transition matrix must have shape ({num_classes}, {num_classes})")

        # Ensure rows sum to 1 (normalize if needed)
        T = transition_matrix.copy().astype(float)
        row_sums = T.sum(axis=1, keepdims=True)
        # Avoid division by zero for rows that are all zero
        row_sums[row_sums == 0] = 1
        T = T / row_sums

        # Verify row sums are 1
        actual_sums = T.sum(axis=1)
        if not np.allclose(actual_sums, 1.0, atol=1e-6):
            print(f"Warning: Row sums are not exactly 1: {actual_sums}")

        np.random.seed(self.seed)
        self.true_targets = copy.deepcopy(self.targets)
        self.k_mean_targets = copy.deepcopy(self.targets)

        # import pdb
        # pdb.set_trace()
        
        new_targets = []
        for target in self.targets:
            prob = T[target, :]
            # Sample one label based on the probability distribution
            cl_label = np.random.choice(np.arange(num_classes), p=prob)
            new_targets.append(np.array([cl_label]))
        
        self.targets = new_targets
        # import pdb
        # pdb.set_trace()
        
        # Save the generated labels to file
        dataset_type = getattr(self, 'dataset_name', 'CIFAR10')
        save_dir = os.path.join(os.getcwd(), 'generated_labels', dataset_type.lower())
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{self.cll_type}[prompt].txt')

        # Convert list of arrays to a single array for saving
        labels_to_save = np.array([label[0] for label in new_targets])
        np.savetxt(save_path, labels_to_save, fmt='%d')
        print(f"Saved transition matrix generated labels to: {save_path}")
        
        print(f"Done generating complementary labels from transition matrix. Generated {len(new_targets)} labels.")

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

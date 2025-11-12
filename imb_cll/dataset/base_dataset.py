import numpy as np
import copy
import torch
import torch.nn.functional as F
import pdb
import os


class BaseDataset:
    def gen_complementary_target(self):
        # This will NOT shuffle the dataset
        # print("Generating complementary targets...")

        # Use the CURRENT dataset size (already limited by max_train_samples if specified)
        num_samples = len(self.targets)
        print(f"Generating complementary labels for {num_samples} samples")

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
            
            # # Save generated labels to file
            # dataset_type = getattr(self, 'dataset_name', 'CIFAR10')
            # organized_file_path = f"generated_labels/{dataset_type.lower()}/{self.cll_type}.txt"
            # os.makedirs(f"generated_labels/{dataset_type.lower()}", exist_ok=True)
            # with open(organized_file_path, 'w') as f:
            #     for label in self.targets:
            #         if isinstance(label, np.ndarray):
            #             f.write(f"array([{label[0]}])\n")
            #         else:
            #             f.write(f"array([{label}])\n")
            # print(f"Saved {len(self.targets)} {self.cll_type} complementary labels to {organized_file_path}")

        elif self.cll_type in ['least', 'most', 'most_no_noise',
                                'from_matrix_least', 'from_matrix_most', 
                                'most+rand', 'least+rand', 'third', 'fourth',
                                'eighth']:
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

        # Preserve true labels for ordinary samples
        if hasattr(self, 'label_type'):
            print("Preserving true labels for ordinary samples...")
            for i in range(len(self.targets)):
                if self.label_type[i] == 1:  # If this is an ordinary sample
                    self.targets[i] = np.array([self.true_targets[i]]) if not isinstance(self.true_targets[i], np.ndarray) else self.true_targets[i]
            print(f"Preserved true labels for {int(np.sum(self.label_type))} ordinary samples")

        # pdb.set_trace()

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
    def gen_few_ordinary_target(self):
        np.random.seed(self.seed)
        ord_num = self.ord_num
        
        # Create a label_type array: 0 = complementary, 1 = ordinary
        self.label_type = np.zeros(len(self.data), dtype=np.int32)
        
        for cls in range(self.num_classes):
            cls_indices = np.where(np.array(self.targets) == cls)[0]
            selected_cls_indices = np.random.choice(cls_indices, size=ord_num, replace=False)
            
            # Mark selected indices as ordinary samples
            self.label_type[selected_cls_indices] = 1
        
        print(f"Generated {np.sum(self.label_type)} ordinary samples out of {len(self.data)} total samples")

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
    
    def generate_cl_from_matrix(self, transition_matrix, ba_config=None, mi_config=None):
        """
        Generates complementary labels based on a given transition matrix.
        
        Parameters:
        -----------
        transition_matrix : np.ndarray
            A numpy array of shape (num_classes, num_classes).
            Row sums should equal 1, diagonal elements can be non-zero.
        ba_config : dict, optional
            Configuration dictionary for Blahut-Arimoto augmentation.
            If None, BA augmentation is disabled. Keys:
            - 'use_blahut' (bool): Enable BA augmentation
            - 'strength' (float): Global augmentation strength in [0, 1]
            - 'row_mode' (str): Target distribution ('uniform' or 'q')
            - 'gamma' (float): Per-row strength contrast exponent
            - 'max_iters' (int): Maximum iterations
            - 'tol' (float): Convergence tolerance
            - 'preserve_diagonal' (bool): Preserve diagonal elements
            - 'save' (bool): Save augmented matrix to file
        mi_config : dict, optional
            Configuration dictionary for MI optimization.
            If None, MI optimization is disabled. Keys:
            - 'use_mi_optimization' (bool): Enable MI optimization
            - 'learning_rate' (float): Gradient ascent step size
            - 'epsilon' (float): Convergence tolerance
            - 'max_iters' (int): Maximum iterations
            - 'budget' (float): Frobenius norm budget constraint (None = no limit)
            - 'P_y' (np.ndarray): Prior distribution over labels (None = uniform)
            - 'save' (bool): Save optimized matrix to file
        
        Note:
        -----
        If both BA and MI are enabled, BA augmentation is applied first,
        then MI optimization is applied to the BA-augmented matrix.
        """
        print("Generating complementary labels from transition matrix...")
        
        # Extract BA configuration
        use_blahut = ba_config.get('use_blahut', False) if ba_config else False
        ba_strength = ba_config.get('strength', 0.1) if ba_config else 0.1
        ba_row_mode = ba_config.get('row_mode', 'uniform') if ba_config else 'uniform'
        ba_gamma = ba_config.get('gamma', 1.0) if ba_config else 1.0
        ba_max_iters = ba_config.get('max_iters', 500) if ba_config else 500
        ba_tol = ba_config.get('tol', 1e-6) if ba_config else 1e-6
        ba_preserve_diagonal = ba_config.get('preserve_diagonal', False) if ba_config else False
        ba_save = ba_config.get('save', True) if ba_config else True
        
        # Extract MI configuration
        use_mi_optimization = mi_config.get('use_mi_optimization', False) if mi_config else False
        mi_learning_rate = mi_config.get('learning_rate', 0.05) if mi_config else 0.05
        mi_epsilon = mi_config.get('epsilon', 1e-6) if mi_config else 1e-6
        mi_max_iters = mi_config.get('max_iters', 2000) if mi_config else 2000
        mi_budget = mi_config.get('budget', None) if mi_config else None
        mi_P_y = mi_config.get('P_y', None) if mi_config else None
        mi_save = mi_config.get('save', True) if mi_config else True
        
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

        # Apply Blahut-Arimoto augmentation if requested
        if use_blahut:
            from imb_cll.utils.ba_utils import augment_transition_matrix_with_ba
            
            print(f"Applying Blahut-Arimoto augmentation (strength={ba_strength}, "
                  f"mode={ba_row_mode}, gamma={ba_gamma})...")
            T_aug, ba_info = augment_transition_matrix_with_ba(
                T, 
                strength=ba_strength,
                row_mode=ba_row_mode,
                gamma=ba_gamma,
                tol=ba_tol,
                max_iters=ba_max_iters,
                preserve_diagonal=ba_preserve_diagonal,
                seed=getattr(self, 'seed', 1126)
            )
            
            # Print BA statistics
            print(f"  p* (optimal input dist): {ba_info['p_star']}")
            print(f"  Row strengths: {ba_info['row_strengths']}")
            
            # Save augmented matrix if requested
            if ba_save:
                dataset_type = getattr(self, 'dataset_name', 'CIFAR10')
                matrix_save_dir = os.path.join(os.getcwd(), 'transition_matrix', 
                                               dataset_type.lower())
                os.makedirs(matrix_save_dir, exist_ok=True)
                matrix_save_path = os.path.join(matrix_save_dir, 
                                                f'{self.cll_type}_augmented_ba.txt')
                np.savetxt(matrix_save_path, T_aug, fmt='%.6f')
                print(f"  Saved augmented transition matrix to: {matrix_save_path}")
            
            # Use augmented matrix for label generation
            T = T_aug
        
        # Apply MI optimization if requested
        if use_mi_optimization:
            from imb_cll.utils.mi_optimization import optimize_cll_matrix_mi
            
            print(f"\nApplying MI optimization (learning_rate={mi_learning_rate}, "
                  f"budget={mi_budget})...")
            T_opt, max_mi, mi_history = optimize_cll_matrix_mi(
                T,
                learning_rate=mi_learning_rate,
                epsilon_convergence=mi_epsilon,
                max_iterations=mi_max_iters,
                budget_B=mi_budget,
                P_y_true=mi_P_y,
                verbose=True
            )
            
            # Print MI statistics
            print(f"\n  MI optimization complete:")
            print(f"  Initial MI: {mi_history[0]:.6f} nats ({mi_history[0]/np.log(2):.6f} bits)")
            print(f"  Final MI: {max_mi:.6f} nats ({max_mi/np.log(2):.6f} bits)")
            print(f"  Improvement: {max_mi - mi_history[0]:.6f} nats "
                  f"({(max_mi - mi_history[0])/np.log(2):.6f} bits)")
            
            # Save optimized matrix if requested
            if mi_save:
                dataset_type = getattr(self, 'dataset_name', 'CIFAR10')
                matrix_save_dir = os.path.join(os.getcwd(), 'transition_matrix', 
                                               dataset_type.lower())
                os.makedirs(matrix_save_dir, exist_ok=True)
                
                # Create filename based on what augmentations were applied
                if use_blahut:
                    suffix = '_augmented_ba_mi_optimized.txt'
                else:
                    suffix = '_mi_optimized.txt'
                matrix_save_path = os.path.join(matrix_save_dir, 
                                                f'{self.cll_type}{suffix}')
                np.savetxt(matrix_save_path, T_opt, fmt='%.6f')
                print(f"  Saved MI-optimized transition matrix to: {matrix_save_path}")
            
            # Use optimized matrix for label generation
            T = T_opt
        
        # Use RNG for reproducibility (prefer modern API)
        rng = np.random.default_rng(getattr(self, 'seed', 1126))
        self.true_targets = copy.deepcopy(self.targets)
        self.k_mean_targets = copy.deepcopy(self.targets)

        # import pdb
        # pdb.set_trace()
        
        new_targets = []
        for target in self.targets:
            # Support targets stored as ints or as arrays like np.array([label])
            if isinstance(target, (np.ndarray, list, tuple)):
                idx = int(np.asarray(target).item())
            else:
                idx = int(target)
            prob = T[idx, :].ravel()  # Ensure 1D array
            # Sample one label based on the probability distribution
            cl_label = int(rng.choice(np.arange(num_classes), p=prob))
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
        
        # Preserve true labels for ordinary samples
        if hasattr(self, 'label_type'):
            print("Preserving true labels for ordinary samples (from transition matrix)...")
            for i in range(len(self.targets)):
                if self.label_type[i] == 1:  # If this is an ordinary sample
                    self.targets[i] = np.array([self.true_targets[i]]) if not isinstance(self.true_targets[i], np.ndarray) else self.true_targets[i]
            print(f"Preserved true labels for {int(np.sum(self.label_type))} ordinary samples")
        
        print(f"Done generating complementary labels from transition matrix. Generated {len(new_targets)} labels.")

    def generate_biased_transition_matrix(self, num_classes, bias_strength=0.7, 
                                         distribution_type='uniform', seed=None):
        """
        Generate a pseudo-random biased transition matrix for complementary label generation.
        
        Parameters:
        -----------
        num_classes : int
            Number of classes in the dataset
        bias_strength : float, optional
            Probability assigned to the biased class. Must be in (0, 1).
            Default: 0.7
        distribution_type : str, optional
            How to distribute remaining probability among non-bias classes:
            - 'uniform': Equal distribution
            - 'random': Random distribution (Dirichlet sampling)
            - 'exponential': Exponentially decreasing distribution
            Default: 'uniform'
        seed : int, optional
            Random seed for reproducibility. If None, uses self.seed
            Default: None
        
        Returns:
        --------
        T : np.ndarray
            Row-stochastic transition matrix of shape (num_classes, num_classes)
            where T[i,j] = P(complementary_label=j | true_label=i)
        
        Properties:
        -----------
        - All diagonal elements are 0 (no self-complementary labels)
        - Each row has exactly one biased class (randomly chosen)
        - The biased class has probability = bias_strength
        - Remaining probability (1 - bias_strength) is distributed among other classes
        - Each row sums to 1 (row-stochastic)
        """
        if not 0 < bias_strength < 1:
            raise ValueError(f"bias_strength must be in (0, 1), got {bias_strength}")
        
        if seed is None:
            seed = getattr(self, 'seed', 1126)
        
        rng = np.random.default_rng(seed)
        
        # Initialize matrix
        T = np.zeros((num_classes, num_classes), dtype=float)
        
        # For each true class (row)
        for true_class in range(num_classes):
            # Get list of valid complementary classes (all except true_class)
            valid_classes = [c for c in range(num_classes) if c != true_class]
            
            # Randomly select one class as the biased class
            bias_class = rng.choice(valid_classes)
            
            # Assign bias_strength probability to the biased class
            T[true_class, bias_class] = bias_strength
            
            # Distribute remaining probability among other valid classes
            remaining_prob = 1.0 - bias_strength
            other_classes = [c for c in valid_classes if c != bias_class]
            
            if len(other_classes) == 0:
                # Edge case: only 2 classes total
                continue
            
            if distribution_type == 'uniform':
                # Equal distribution
                prob_per_class = remaining_prob / len(other_classes)
                for c in other_classes:
                    T[true_class, c] = prob_per_class
                    
            elif distribution_type == 'random':
                # Random distribution using Dirichlet
                # Alpha = 1 gives uniform Dirichlet (similar to random simplex sampling)
                alphas = np.ones(len(other_classes))
                random_probs = rng.dirichlet(alphas)
                random_probs = random_probs * remaining_prob  # Scale to sum to remaining_prob
                for i, c in enumerate(other_classes):
                    T[true_class, c] = random_probs[i]
                    
            elif distribution_type == 'exponential':
                # Exponentially decreasing distribution
                # Shuffle to make which class gets highest prob random
                shuffled_classes = other_classes.copy()
                rng.shuffle(shuffled_classes)
                
                # Generate exponentially decreasing weights
                weights = np.array([np.exp(-0.5 * i) for i in range(len(shuffled_classes))])
                weights = weights / weights.sum()  # Normalize
                weights = weights * remaining_prob  # Scale to sum to remaining_prob
                
                for i, c in enumerate(shuffled_classes):
                    T[true_class, c] = weights[i]
            else:
                raise ValueError(f"Unknown distribution_type: {distribution_type}. "
                               f"Must be 'uniform', 'random', or 'exponential'")
        
        # Verify properties
        assert np.allclose(np.diag(T), 0), "Diagonal elements must be 0"
        row_sums = T.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10), f"Rows must sum to 1, got {row_sums}"
        
        return T

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


    

#!/usr/bin/env python3
"""
Image Prediction Module using Pretrained ResNet32 for CIFAR20

This module provides functionality to predict CIFAR20 classes using a pretrained ResNet32 model
trained on CIFAR100, with mapping from CIFAR100 to CIFAR20 classes.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import warnings

from .resnet32_cifar100 import resnet32_cifar100


def _cifar100_to_cifar20(target):
    """Map CIFAR100 class to CIFAR20 class"""
    _dict = {
        0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3,
        10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11,
        20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15,
        30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5,
        40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10,
        50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17,
        60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19,
        70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13,
        80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19,
        90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13,
    }
    return _dict[target]


def _cifar20_to_cifar100_classes(cifar20_class):
    """Get all CIFAR100 classes that map to a given CIFAR20 class"""
    cifar100_classes = []
    for cifar100_class in range(100):
        if _cifar100_to_cifar20(cifar100_class) == cifar20_class:
            cifar100_classes.append(cifar100_class)
    return cifar100_classes


class ImagePredictorCIFAR20:
    """
    Image prediction module using pretrained ResNet32 for CIFAR-20 classification.
    
    This class loads a pretrained ResNet32 model trained on CIFAR100 and maps
    predictions to CIFAR20 classes.
    """

    def __init__(self, device=None, pretrained=True, mode='most', debug=False, noise=False):
        """
        Initialize the image predictor.
        
        Args:
            device (torch.device, optional): Device to run the model on. 
                                           If None, automatically selects GPU if available.
            pretrained (bool): Whether to load pretrained weights. Default: True.
            mode (str): Prediction mode ('most', 'least', 'most_no_noise'). Default: 'most'
            debug (bool): Whether to print debug messages. Default: False.
            noise (bool): If False, remove true label before prediction. If True, keep true label. Default: False.
        """
        self.debug = debug
        self.mode = mode
        self.noise = noise
        
        # Automatically select device if not provided
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
                if self.debug:
                    print(f"🔧 DEBUG: Auto-selected device: {device} (CUDA not available)")
        
        self.device = device
        
        # Load the ResNet32 model trained on CIFAR100
        try:
            self.model = resnet32_cifar100(pretrained=pretrained, device=self.device)
            self.model.eval()  # Set to evaluation mode
            
            if self.debug:
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"🔧 DEBUG: ResNet32 model loaded successfully on {self.device}")
                print(f"🔧 DEBUG: Total parameters: {total_params:,}")
                print(f"🔧 DEBUG: Model architecture: ResNet32 for CIFAR100→CIFAR20 mapping")
                
        except Exception as e:
            print(f"❌ ERROR: Failed to load model: {e}")
            raise
        
        # Define CIFAR-20 class names for reference
        self.class_names = [
            'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
            'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
            'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
            'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
            'trees', 'vehicles_1', 'vehicles_2'
        ]
        
        # Define normalization parameters for CIFAR (same as CIFAR100)
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        
        # Create transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts [0,255] to [0,1] and changes HWC to CHW
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # Create a mapping from CIFAR-20 to CIFAR-100 classes
        self.cifar20_to_100_map = {i: [] for i in range(20)}
        for c100 in range(100):
            c20 = _cifar100_to_cifar20(c100)
            self.cifar20_to_100_map[c20].append(c100)
        
        if self.debug:
            print(f"🔧 DEBUG: Preprocessing pipeline initialized")
            print(f"🔧 DEBUG: Normalization - Mean: {self.mean}, Std: {self.std}")
            print(f"🔧 DEBUG: CIFAR20 class names: {self.class_names}")

    def set_mode(self, mode):
        """Set the mode of the image predictor."""
        self.mode = mode

    def set_true_label(self, true_label):
        """Set the true label for the current image (CIFAR20 class)"""
        self.true_label = true_label

    def preprocess_image(self, image_array):
        """
        Preprocess a numpy image array for model input.
        
        Args:
            image_array (np.ndarray): Image array in HWC format with shape (H, W, 3)
                                    and dtype uint8 (values 0-255)
        
        Returns:
            torch.Tensor: Preprocessed image tensor ready for model input
        """
        # Validate input
        if not isinstance(image_array, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image_array)}")
        
        if image_array.shape[-1] != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {image_array.shape[-1]}")
        
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        # Convert numpy array to PIL Image for torchvision transforms
        if len(image_array.shape) == 3:  # Single image
            pil_image = Image.fromarray(image_array)
        else:
            raise ValueError(f"Expected 3D array (H, W, C), got shape {image_array.shape}")
        
        # Apply transforms
        tensor_image = self.transform(pil_image)
        return tensor_image
    
    def predict_single_image(self, image_array):
        """
        Predict the CIFAR20 class of a single image.
        
        Args:
            image_array (np.ndarray): Image array in HWC format (H, W, 3) with uint8 values
        
        Returns:
            dict: Dictionary containing predicted_class (CIFAR20 class)
        """
        if self.debug:
            print(f"🔧 DEBUG: Starting CIFAR20 single image prediction...")
        
        # Preprocess the image
        tensor_image = self.preprocess_image(image_array)
        
        # Add batch dimension and move to device
        tensor_image = tensor_image.unsqueeze(0).to(self.device)
        
        if self.debug:
            print(f"🔧 DEBUG: Input tensor shape: {tensor_image.shape}")
            print(f"🔧 DEBUG: Input tensor device: {tensor_image.device}")
        
        # Perform inference with ResNet32 (CIFAR100 predictions)
        with torch.no_grad():
            logits = self.model(tensor_image)
        
        if self.debug:
            print(f"🔧 DEBUG: Model output (logits) shape: {logits.shape}")
        
        # Calculate probabilities for all 100 CIFAR100 classes
        probabilities = F.softmax(logits, dim=1)
        probs_array = probabilities.cpu().numpy().flatten()
        
        if self.debug:
            print(f"🔧 DEBUG: Got {len(probs_array)} CIFAR100 class probabilities")
        
        # Prepare results based on mode
        if self.mode == 'most_no_noise':
            results = {
                'predicted_class': self._get_nth_highest_cifar20_class(probs_array, 1),
            }
        elif self.mode == 'least':
            results = {
                'predicted_class': self._get_lowest_cifar20_class(probs_array)
            }
        elif self.mode == 'most':
            # For most, we want the highest class without removing true label
            old_noise = self.noise
            self.noise = True
            results = {
                'predicted_class': self._get_nth_highest_cifar20_class(probs_array, 1)
            }
            self.noise = old_noise
        else:  # default mode returns full results
            predicted_class = torch.argmax(probabilities, dim=1).item()
            mapped_class = _cifar100_to_cifar20(predicted_class)
            confidence = probabilities[0, predicted_class].item()
            
            results = {
                'logits': logits.cpu().numpy().flatten(),
                'predicted_class': mapped_class,
                'predicted_class_name': self.class_names[mapped_class],
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy().flatten()
            }
        
        return results
    
    def _get_cifar20_probabilities(self, cifar100_probabilities):
        """
        Aggregate CIFAR-100 probabilities into CIFAR-20 probabilities.
        """
        cifar20_probs = np.zeros(20)
        for c20_class, c100_classes in self.cifar20_to_100_map.items():
            cifar20_probs[c20_class] = np.sum(cifar100_probabilities[c100_classes])
        return cifar20_probs

    def _get_nth_highest_cifar20_class(self, cifar100_probabilities, n):
        """
        Get the nth highest CIFAR20 class by aggregating probabilities.
        
        Args:
            cifar100_probabilities (np.ndarray): Array of 100 CIFAR100 class probabilities
            n (int): The rank to retrieve (1-based)
        
        Returns:
            int: CIFAR20 class index, or -1 if n is out of bounds
        """
        cifar20_probs = self._get_cifar20_probabilities(cifar100_probabilities)
        
        available_cifar20_classes = list(range(20))

        if not self.noise and hasattr(self, 'true_label'):
            if self.true_label in available_cifar20_classes:
                available_cifar20_classes.remove(self.true_label)
            if self.debug:
                print(f"🔧 DEBUG: Removed true CIFAR20 label {self.true_label} from consideration.")

        if not available_cifar20_classes:
            return -1

        prob_dict = {c: cifar20_probs[c] for c in available_cifar20_classes}
        
        sorted_classes = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
        
        # When noise=True, return n+1 label since highest is likely true label
        target_n = n + 1 if self.noise else n
        
        if 1 <= target_n <= len(sorted_classes):
            nth_highest_cifar20 = sorted_classes[target_n-1][0]
            if self.debug:
                print(f"🔧 DEBUG: {target_n}-th highest CIFAR20 class: {nth_highest_cifar20}")
            return nth_highest_cifar20
        else:
            if self.debug:
                warnings.warn(f"Warning: target_n={target_n} is out of bounds for {len(sorted_classes)} available classes.")
            return -1
    
    def _get_lowest_cifar20_class(self, cifar100_probabilities):
        """
        Get the CIFAR20 class with the lowest aggregated probability.
        
        Args:
            cifar100_probabilities (np.ndarray): Array of 100 CIFAR100 class probabilities
        
        Returns:
            int: CIFAR20 class index with lowest probability
        """
        cifar20_probs = self._get_cifar20_probabilities(cifar100_probabilities)

        available_cifar20_classes = list(range(20))
        
        if not self.noise and hasattr(self, 'true_label'):
            if self.true_label in available_cifar20_classes:
                available_cifar20_classes.remove(self.true_label)
            if self.debug:
                print(f"🔧 DEBUG: Removed true CIFAR20 label {self.true_label} from consideration.")

        if not available_cifar20_classes:
            return -1
        
        min_prob = float('inf')
        lowest_cifar20_class = -1
        
        for c in available_cifar20_classes:
            if cifar20_probs[c] < min_prob:
                min_prob = cifar20_probs[c]
                lowest_cifar20_class = c
        
        if self.debug:
            print(f"🔧 DEBUG: Lowest CIFAR20 class: {lowest_cifar20_class}")
        
        return lowest_cifar20_class
        
    def get_model_info(self):
        """Get information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': 'ResNet32',
            'num_classes': 20,
            'dataset': 'CIFAR-20',
            'backbone_trained_on': 'CIFAR-100',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'class_names': self.class_names
        }

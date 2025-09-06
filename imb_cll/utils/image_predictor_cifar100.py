#!/usr/bin/env python3
"""
Image Prediction Module using Pretrained ResNet32 for CIFAR100

This module provides functionality to predict CIFAR100 classes using a pretrained ResNet32 model
trained on CIFAR100, working directly with all 100 classes without any mapping.
"""
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image

from .resnet32_cifar100 import resnet32_cifar100


class ImagePredictorCIFAR100:
    """
    Image prediction module using pretrained ResNet32 for CIFAR-100 classification.
    
    This class loads a pretrained ResNet32 model trained on CIFAR100 and provides
    direct predictions for all 100 classes without any label mapping.
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
        
        self.device = device
        
        # Load the ResNet32 model trained on CIFAR100
        try:
            self.model = resnet32_cifar100(pretrained=pretrained, device=self.device)
            self.model.eval()  # Set to evaluation mode
            
            if self.debug:
                print(f"✅ Model loaded successfully on {self.device}")
                print(f"🔧 DEBUG: Model type: ResNet32-CIFAR100")
                
        except Exception as e:
            print(f"❌ ERROR: Failed to load model: {e}")
            raise
        
        # Define CIFAR-100 class names (fine labels)
        self.class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
        
        # Define normalization parameters for CIFAR-100
        self.mean = [0.5071, 0.4865, 0.4409]
        self.std = [0.2673, 0.2564, 0.2762]
        
        # Create transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts [0,255] to [0,1] and changes HWC to CHW
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        if self.debug:
            print(f"🔧 DEBUG: Preprocessing pipeline initialized")
            print(f"🔧 DEBUG: Normalization - Mean: {self.mean}, Std: {self.std}")
            print(f"🔧 DEBUG: Number of classes: {len(self.class_names)}")

    def set_mode(self, mode):
        """Set the mode of the image predictor."""
        self.mode = mode

    def set_true_label(self, true_label):
        """Set the true label for the current image (CIFAR100 class)"""
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
        Predict the CIFAR100 class of a single image.
        
        Args:
            image_array (np.ndarray): Image array in HWC format (H, W, 3) with uint8 values
        
        Returns:
            dict: Dictionary containing predicted_class (CIFAR100 class)
        """
        if self.debug:
            print(f"🔧 DEBUG: Starting CIFAR100 single image prediction...")
        
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
                'predicted_class': self._get_nth_highest_class(probs_array, 1),
            }
        elif self.mode == 'least':
            results = {
                'predicted_class': self._get_lowest_class(probs_array)
            }
        elif self.mode == 'most':
            # For most_no_noise, we want the highest class without removing true label
            old_noise = self.noise
            self.noise = True
            results = {
                'predicted_class': self._get_nth_highest_class(probs_array, 1)
            }
            self.noise = old_noise
        else:  # default mode returns full results
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            results = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_name': self.class_names[predicted_class]
            }
        
        return results

    def _get_nth_highest_class(self, probabilities, n):
        """
        Get the nth highest CIFAR100 class.
        
        Args:
            probabilities (np.ndarray): Array of 100 CIFAR100 class probabilities
            n (int): The rank to retrieve (1-based)
        
        Returns:
            int: CIFAR100 class index, or -1 if n is out of bounds
        """
        available_classes = list(range(100))

        if not self.noise and hasattr(self, 'true_label'):
            if self.true_label in available_classes:
                available_classes.remove(self.true_label)

        if not available_classes:
            return -1

        prob_dict = {c: probabilities[c] for c in available_classes}
        
        sorted_classes = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
        
        # When noise=True, return n+1 label since highest is likely true label
        target_n = n + 1 if self.noise else n
        
        if 1 <= target_n <= len(sorted_classes):
            return sorted_classes[target_n-1][0]
        else:
            return -1
    
    def _get_lowest_class(self, probabilities):
        """
        Get the CIFAR100 class with the lowest probability.
        
        Args:
            probabilities (np.ndarray): Array of 100 CIFAR100 class probabilities
        
        Returns:
            int: CIFAR100 class index with lowest probability
        """
        available_classes = list(range(100))
        
        if not self.noise and hasattr(self, 'true_label'):
            if self.true_label in available_classes:
                available_classes.remove(self.true_label)

        if not available_classes:
            return -1
        
        min_prob = float('inf')
        lowest_class = -1
        
        for c in available_classes:
            if probabilities[c] < min_prob:
                min_prob = probabilities[c]
                lowest_class = c
        
        if self.debug:
            print(f"🔧 DEBUG: Lowest class: {lowest_class} with probability {min_prob:.6f}")
        
        return lowest_class
        
    def get_model_info(self):
        """Get information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': 'ResNet32',
            'num_classes': 100,
            'dataset': 'CIFAR-100',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'class_names': self.class_names
        }

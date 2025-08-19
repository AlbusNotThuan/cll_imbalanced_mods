#!/usr/bin/env python3
"""
Image Prediction Module using Pretrained ResNet18

This module provides functionality to predict image classes using a pretrained ResNet18 model.
It handles image preprocessing and returns prediction probabilities for all classes.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import warnings

from .resnet18 import resnet18


class ImagePredictor:
    """
    Image prediction module using pretrained ResNet18 for CIFAR-10 classification.
    
    This class loads a pretrained ResNet18 model and provides methods to predict
    image classes from numpy arrays in HWC format (Height, Width, Channels).
    """
    
    def __init__(self, device=None, pretrained=True, mode='most', debug=False):
        """
        Initialize the image predictor.
        
        Args:
            device (torch.device, optional): Device to run the model on. 
                                           If None, automatically selects GPU if available.
            pretrained (bool): Whether to load pretrained weights. Default: True.
            debug (bool): Whether to print debug messages. Default: True.
        """
        self.debug = debug
        self.mode = mode
        
        # Automatically select device if not provided
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                # if self.debug:
                #     print(f"🔧 DEBUG: Auto-selected device: {device} (CUDA available)")
            else:
                device = torch.device('cpu')
                if self.debug:
                    print(f"🔧 DEBUG: Auto-selected device: {device} (CUDA not available)")
        
        self.device = device
        
        # Load the model
        # if self.debug:
        #     print(f"🔧 DEBUG: Loading ResNet18 model (pretrained={pretrained})...")
        
        try:
            self.model = resnet18(pretrained=pretrained, device=self.device, num_classes=10)
            self.model.eval()  # Set to evaluation mode
            
            if self.debug:
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"🔧 DEBUG: Model loaded successfully on {self.device}")
                print(f"🔧 DEBUG: Total parameters: {total_params:,}")
                print(f"🔧 DEBUG: Model architecture: ResNet18 for CIFAR-10 (10 classes)")
                
        except Exception as e:
            print(f"❌ ERROR: Failed to load model: {e}")
            raise
        
        # Define CIFAR-10 class names for reference
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # Define normalization parameters for CIFAR-10
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        
        # Create transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts [0,255] to [0,1] and changes HWC to CHW
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        if self.debug:
            print(f"🔧 DEBUG: Preprocessing pipeline initialized")
            print(f"🔧 DEBUG: Normalization - Mean: {self.mean}, Std: {self.std}")
            print(f"🔧 DEBUG: Class names: {self.class_names}")

    def set_mode(self, mode):
        """
        Set the mode of the image predictor.

        Args:
            mode (str): Mode to set ('most' or 'least')
        """
        self.mode = mode
    
    def preprocess_image(self, image_array):
        """
        Preprocess a numpy image array for model input.
        
        Args:
            image_array (np.ndarray): Image array in HWC format with shape (H, W, 3)
                                    and dtype uint8 (values 0-255)
        
        Returns:
            torch.Tensor: Preprocessed image tensor ready for model input
        """
        # if self.debug:
        #     print(f"🔧 DEBUG: Preprocessing image...")
        #     print(f"🔧 DEBUG: Input shape: {image_array.shape}, dtype: {image_array.dtype}")
        #     print(f"🔧 DEBUG: Input value range: [{image_array.min()}, {image_array.max()}]")
        
        # Validate input
        if not isinstance(image_array, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image_array)}")
        
        if image_array.shape[-1] != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {image_array.shape[-1]}")
        
        if image_array.dtype != np.uint8:
            # if self.debug:
            #     print(f"⚠️  DEBUG: Converting from {image_array.dtype} to uint8")
            image_array = image_array.astype(np.uint8)
        
        # Convert numpy array to PIL Image for torchvision transforms
        if len(image_array.shape) == 3:  # Single image
            pil_image = Image.fromarray(image_array)
        else:
            raise ValueError(f"Expected 3D array (H, W, C), got shape {image_array.shape}")
        
        # Apply transforms
        tensor_image = self.transform(pil_image)
        
        # if self.debug:
        #     print(f"🔧 DEBUG: After preprocessing:")
        #     print(f"🔧 DEBUG: - Tensor shape: {tensor_image.shape}")
        #     print(f"🔧 DEBUG: - Tensor dtype: {tensor_image.dtype}")
        #     print(f"🔧 DEBUG: - Tensor value range: [{tensor_image.min():.4f}, {tensor_image.max():.4f}]")
        
        return tensor_image
    
    def predict_single_image(self, image_array):
        """
        Predict the class of a single image.
        
        Args:
            image_array (np.ndarray): Image array in HWC format (H, W, 3) with uint8 values
            return_probabilities (bool): If True, return softmax probabilities. 
                                       If False, return raw logits.
        
        Returns:
            dict: Dictionary containing:
                - 'logits': Raw model output (before softmax)
                - 'probabilities': Softmax probabilities (if return_probabilities=True)
                - 'predicted_class': Index of predicted class
                - 'predicted_class_name': Name of predicted class
                - 'confidence': Highest probability value
        """
        if self.debug:
            print(f"🔧 DEBUG: Starting single image prediction...")
        
        # Preprocess the image
        tensor_image = self.preprocess_image(image_array)
        
        # Add batch dimension and move to device
        tensor_image = tensor_image.unsqueeze(0).to(self.device)
        
        if self.debug:
            print(f"🔧 DEBUG: Input tensor shape: {tensor_image.shape}")
            print(f"🔧 DEBUG: Input tensor device: {tensor_image.device}")
        
        # Perform inference
        with torch.no_grad():
            logits = self.model(tensor_image)
        
        if self.debug:
            print(f"🔧 DEBUG: Model output (logits) shape: {logits.shape}")
            print(f"🔧 DEBUG: Raw logits: {logits.cpu().numpy().flatten()}")
        
        # Calculate probabilities
        probabilities = F.softmax(logits, dim=1)
        
        # Get predictions
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        predicted_class_name = self.class_names[predicted_class]
        probs_array = probabilities.cpu().numpy().flatten()
        
        if self.debug:
            pass
            
        # Prepare results based on mode
        if self.mode == 'most':
            results = {
                'predicted_class': self._get_second_highest_class(probs_array) 
            }
        elif self.mode == 'least':
            results = {
                'predicted_class': self._get_lowest_class(probs_array)
            }
        else:  # default mode returns full results
            results = {
                'logits': logits.cpu().numpy().flatten(),
                'predicted_class': predicted_class,
                'predicted_class_name': predicted_class_name,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy().flatten()
            }

            # Add second highest and lowest class predictions
            results['second_highest_class'] = self._get_second_highest_class(probs_array)
            results['lowest_class'] = self._get_lowest_class(probs_array)
            results['second_highest_class_name'] = self.class_names[results['second_highest_class']]
            results['lowest_class_name'] = self.class_names[results['lowest_class']]
        
        return results
    
    def _get_second_highest_class(self, probabilities):
        """
        Get the class with the second highest probability.
        
        Args:
            probabilities (np.ndarray): Array of class probabilities
            
        Returns:
            int: Index of class with second highest probability
        """
        # Get indices sorted by probability (descending)
        sorted_indices = np.argsort(probabilities)[::-1]
        
        if self.debug:
            print(f"🔧 DEBUG: Sorted class indices by probability: {sorted_indices}")
            print(f"🔧 DEBUG: Second highest class: {sorted_indices[1]} with prob {probabilities[sorted_indices[1]]:.6f}")
        
        return sorted_indices[1]  # Second highest
    
    def _get_lowest_class(self, probabilities):
        """
        Get the class with the lowest probability.
        
        Args:
            probabilities (np.ndarray): Array of class probabilities
            
        Returns:
            int: Index of class with lowest probability
        """
        # Get index of minimum probability
        lowest_idx = np.argmin(probabilities)
        
        if self.debug:
            print(f"🔧 DEBUG: Lowest class: {lowest_idx} with prob {probabilities[lowest_idx]:.6f}")
        
        return lowest_idx
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': 'ResNet18',
            'num_classes': 10,
            'dataset': 'CIFAR-10',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'class_names': self.class_names
        }


def create_predictor(device=None, pretrained=True, mode='most', debug=True):
    """
    Convenience function to create an ImagePredictor instance.
    
    Args:
        device (torch.device, optional): Device to run the model on
        pretrained (bool): Whether to load pretrained weights
        debug (bool): Whether to enable debug messages
    
    Returns:
        ImagePredictor: Initialized predictor instance
    """
    return ImagePredictor(device=device, pretrained=pretrained, mode=mode, debug=debug)




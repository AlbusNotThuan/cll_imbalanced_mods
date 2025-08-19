#!/usr/bin/env python3
"""
Test script for the Image Predictor Module

This script tests the image predictor with sample data similar to your input format.
"""

import sys
import numpy as np

# Add the project root to Python path
sys.path.append('/home/hamt/thuan_cll')

from imb_cll.utils.image_predictor import create_predictor


def create_sample_image():
    """Create a sample image similar to your input format"""
    # Create a sample 32x32x3 image with similar values to your example
    image = np.full((32, 32, 3), [177, 208, 255], dtype=np.uint8)
    
    # Add some variation to make it more realistic
    # Add some noise
    noise = np.random.randint(-10, 11, size=(32, 32, 3))
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    # Add some pattern variation like in your example
    image[16:, :, :] = [180, 208, 255]  # Bottom half slightly different
    image[:, 16:, :] = np.clip(image[:, 16:, :].astype(int) + [5, -2, -2], 0, 255).astype(np.uint8)
    
    return image


def test_predictor():
    """Test the image predictor module"""
    print("🧪 Testing Image Predictor Module")
    print("=" * 60)
    
    # Create the predictor
    print("1. Creating predictor...")
    predictor = create_predictor(debug=True)
    
    print("\n" + "=" * 60)
    print("2. Creating sample image...")
    
    # Create a sample image
    sample_image = create_sample_image()
    
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample image dtype: {sample_image.dtype}")
    print(f"Sample image value range: [{sample_image.min()}, {sample_image.max()}]")
    print(f"Sample of pixel values:")
    print(f"  Top-left corner: {sample_image[0, 0]}")
    print(f"  Top-right corner: {sample_image[0, -1]}")
    print(f"  Bottom-left corner: {sample_image[-1, 0]}")
    print(f"  Bottom-right corner: {sample_image[-1, -1]}")
    
    print("\n" + "=" * 60)
    print("3. Making prediction...")
    
    # Make prediction
    results = predictor.predict_single_image(sample_image, return_probabilities=True)
    
    print("\n" + "=" * 60)
    print("4. Results:")
    print(f"Predicted class: {results['predicted_class']}")
    print(f"Predicted class name: {results['predicted_class_name']}")
    print(f"Confidence: {results['confidence']:.6f}")
    
    print(f"\nAll class probabilities:")
    class_names = predictor.get_class_names()
    probabilities = results['probabilities']
    
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        marker = " ← PREDICTED" if i == results['predicted_class'] else ""
        print(f"  {i}: {class_name:12} = {prob:.6f}{marker}")
    
    print(f"\nRaw logits (before softmax):")
    logits = results['logits']
    for i, (class_name, logit) in enumerate(zip(class_names, logits)):
        marker = " ← HIGHEST" if i == results['predicted_class'] else ""
        print(f"  {i}: {class_name:12} = {logit:.6f}{marker}")
    
    print("\n" + "=" * 60)
    print("5. Testing with your exact input format...")
    
    # Create an image that matches your exact description
    your_format_image = np.array([[[177, 208, 255],
                                  [177, 208, 255],
                                  [177, 208, 255]],
                                 [[177, 208, 255],
                                  [177, 208, 255],
                                  [177, 208, 255]],
                                 [[180, 208, 255],
                                  [180, 208, 255],
                                  [180, 208, 255]]], dtype=np.uint8)
    
    # Expand to 32x32x3
    your_format_image = np.tile(your_format_image, (11, 11, 1))[:32, :32, :]
    
    print(f"Your format image shape: {your_format_image.shape}")
    print(f"Your format image dtype: {your_format_image.dtype}")
    
    # Make prediction
    results2 = predictor.predict_single_image(your_format_image, return_probabilities=True)
    
    print(f"\nPrediction for your format:")
    print(f"Predicted class: {results2['predicted_class']} ({results2['predicted_class_name']})")
    print(f"Confidence: {results2['confidence']:.6f}")
    print(f"Probability array: {results2['probabilities']}")
    
    print("\n" + "=" * 60)
    print("6. Model information:")
    model_info = predictor.get_model_info()
    for key, value in model_info.items():
        if key != 'class_names':
            print(f"{key}: {value}")
    
    print("\n🎉 Test completed successfully!")


def test_batch_prediction():
    """Test batch prediction functionality"""
    print("\n" + "=" * 60)
    print("7. Testing batch prediction...")
    
    predictor = create_predictor(debug=False)  # Disable debug for batch test
    
    # Create multiple test images
    images = []
    for i in range(3):
        img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
        images.append(img)
        print(f"Created test image {i+1}: shape {img.shape}")
    
    # Predict batch
    results = predictor.predict_batch(images, return_probabilities=True)
    
    print(f"\nBatch prediction results:")
    for i, result in enumerate(results):
        print(f"Image {i+1}: {result['predicted_class_name']} (confidence: {result['confidence']:.4f})")


if __name__ == "__main__":
    try:
        test_predictor()
        test_batch_prediction()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

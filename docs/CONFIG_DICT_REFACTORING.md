# Configuration Dictionary Refactoring

## Overview
Refactored BA (Blahut-Arimoto) and MI (Mutual Information) optimization parameters to use dictionary-based configuration for cleaner, more maintainable code.

## Changes Made

### 1. train.py
**Before:** 14 individual parameters passed through multiple function calls
```python
use_blahut=use_blahut, ba_strength=args.ba_strength, ba_row_mode=args.ba_row_mode, 
ba_gamma=args.ba_gamma, ba_max_iters=args.ba_max_iters, ba_tol=args.ba_tol, 
ba_preserve_diagonal=ba_preserve_diagonal, ba_save=ba_save,
use_mi_optimization=use_mi_optimization, mi_learning_rate=args.mi_learning_rate, 
mi_epsilon=args.mi_epsilon, mi_max_iters=args.mi_max_iters, 
mi_budget=args.mi_budget, mi_P_y=None, mi_save=mi_save
```

**After:** 2 configuration dictionaries
```python
# Create BA configuration dictionary
ba_config = {
    'use_blahut': use_blahut,
    'strength': args.ba_strength,
    'row_mode': args.ba_row_mode,
    'gamma': args.ba_gamma,
    'max_iters': args.ba_max_iters,
    'tol': args.ba_tol,
    'preserve_diagonal': ba_preserve_diagonal,
    'save': ba_save
}

# Create MI configuration dictionary
mi_config = {
    'use_mi_optimization': use_mi_optimization,
    'learning_rate': args.mi_learning_rate,
    'epsilon': args.mi_epsilon,
    'max_iters': args.mi_max_iters,
    'budget': args.mi_budget,
    'P_y': None,  # Will use uniform distribution by default
    'save': mi_save
}

# Pass to dataset preparation
trainset, input_dim, num_classes = prepare_cluster_dataset(
    ...,
    ba_config=ba_config,
    mi_config=mi_config
)
```

### 2. dataset.py (`prepare_cluster_dataset`)
**Before:** 14 individual parameters in function signature
```python
def prepare_cluster_dataset(..., use_blahut=False, ba_strength=0.1, ...)
```

**After:** 2 dictionary parameters
```python
def prepare_cluster_dataset(..., ba_config=None, mi_config=None):
    if input_dataset == "CIFAR10":
        if data_type == "train":
            dataset = CLCIFAR10(
                ...,
                ba_config=ba_config,
                mi_config=mi_config
            )
```

### 3. clcifar_cluster_label.py (CLCIFAR10/20/100 classes)
**Before:** 14 individual parameters in __init__ and stored as instance variables
```python
def __init__(self, ..., use_blahut=False, ba_strength=0.1, ...):
    self.use_blahut = use_blahut
    self.ba_strength = ba_strength
    ...  # 12 more parameters
```

**After:** 2 dictionary parameters
```python
def __init__(self, ..., ba_config=None, mi_config=None):
    # Store BA and MI configurations
    self.ba_config = ba_config
    self.mi_config = mi_config

# Updated generate_cl_from_matrix call
self.generate_cl_from_matrix(
    self.transition_matrix,
    ba_config=self.ba_config,
    mi_config=self.mi_config
)
```

### 4. base_dataset.py (`generate_cl_from_matrix`)
**Before:** 14 individual parameters in method signature
```python
def generate_cl_from_matrix(self, transition_matrix, use_blahut=False, 
                            ba_strength=0.1, ba_row_mode='uniform', ...)
```

**After:** 2 dictionary parameters with smart defaults
```python
def generate_cl_from_matrix(self, transition_matrix, ba_config=None, mi_config=None):
    """
    Parameters:
    -----------
    ba_config : dict, optional
        Configuration dictionary for Blahut-Arimoto augmentation.
        If None, BA augmentation is disabled. Keys:
        - 'use_blahut' (bool): Enable BA augmentation
        - 'strength' (float): Global augmentation strength
        - 'row_mode' (str): Target distribution
        - 'gamma' (float): Per-row strength contrast
        - 'max_iters' (int): Maximum iterations
        - 'tol' (float): Convergence tolerance
        - 'preserve_diagonal' (bool): Preserve diagonal
        - 'save' (bool): Save matrix to file
    
    mi_config : dict, optional
        Configuration dictionary for MI optimization.
        If None, MI optimization is disabled. Keys:
        - 'use_mi_optimization' (bool): Enable MI optimization
        - 'learning_rate' (float): Gradient ascent step size
        - 'epsilon' (float): Convergence tolerance
        - 'max_iters' (int): Maximum iterations
        - 'budget' (float): Frobenius norm budget
        - 'P_y' (np.ndarray): Prior distribution
        - 'save' (bool): Save matrix to file
    """
    # Extract BA configuration with defaults
    use_blahut = ba_config.get('use_blahut', False) if ba_config else False
    ba_strength = ba_config.get('strength', 0.1) if ba_config else 0.1
    ba_row_mode = ba_config.get('row_mode', 'uniform') if ba_config else 'uniform'
    # ... extract all BA params with defaults
    
    # Extract MI configuration with defaults
    use_mi_optimization = mi_config.get('use_mi_optimization', False) if mi_config else False
    mi_learning_rate = mi_config.get('learning_rate', 0.05) if mi_config else 0.05
    # ... extract all MI params with defaults
```

## Benefits

### 1. **Cleaner Code**
- Reduced parameter count from 14 to 2 in function signatures
- Easier to read and understand function calls
- Clear grouping of related configuration options

### 2. **Better Maintainability**
- Adding new BA/MI parameters only requires updating the dictionary
- No need to modify multiple function signatures
- Centralized configuration management

### 3. **Improved Flexibility**
- Easy to disable entire feature sets by passing `None`
- Can create different configuration presets
- Simple to validate configuration at one place

### 4. **Type Safety & Documentation**
- Dictionary keys are self-documenting
- Clear structure for what parameters belong together
- Easy to add validation for configuration dictionaries

### 5. **Backwards Compatible Pattern**
- Existing code behavior unchanged (same defaults)
- `None` config means feature disabled (safe default)
- Easy to extend in the future

## Usage Examples

### Basic Usage (No Optimization)
```python
trainset, _, _ = prepare_cluster_dataset(
    input_dataset="CIFAR10",
    data_type="train",
    ba_config=None,      # BA disabled
    mi_config=None       # MI disabled
)
```

### Enable Only BA
```python
ba_config = {
    'use_blahut': True,
    'strength': 0.2,
    'row_mode': 'uniform',
    'gamma': 1.0,
    'max_iters': 500,
    'tol': 1e-6,
    'preserve_diagonal': False,
    'save': True
}

trainset, _, _ = prepare_cluster_dataset(
    input_dataset="CIFAR10",
    data_type="train",
    ba_config=ba_config,
    mi_config=None
)
```

### Enable Only MI
```python
mi_config = {
    'use_mi_optimization': True,
    'learning_rate': 0.05,
    'epsilon': 1e-6,
    'max_iters': 2000,
    'budget': 0.5,
    'P_y': None,
    'save': True
}

trainset, _, _ = prepare_cluster_dataset(
    input_dataset="CIFAR10",
    data_type="train",
    ba_config=None,
    mi_config=mi_config
)
```

### Enable Both (BA then MI)
```python
ba_config = {...}  # BA configuration
mi_config = {...}  # MI configuration

trainset, _, _ = prepare_cluster_dataset(
    input_dataset="CIFAR10",
    data_type="train",
    ba_config=ba_config,
    mi_config=mi_config  # MI applied after BA
)
```

### CLI Integration
The CLI arguments remain unchanged. The refactoring only affects internal code organization:
```bash
python train.py --dataset CIFAR10 \
    --use_blahut true --ba_strength 0.2 \
    --use_mi_optimization true --mi_learning_rate 0.05
```

## Files Modified
1. `train.py` - Create config dicts from CLI args
2. `imb_cll/dataset/dataset.py` - Update function signatures
3. `imb_cll/dataset/clcifar_cluster_label.py` - Update all CIFAR classes
4. `imb_cll/dataset/base_dataset.py` - Update generate_cl_from_matrix

## Testing Recommendations
1. Test with both BA and MI disabled (None configs)
2. Test with only BA enabled
3. Test with only MI enabled
4. Test with both BA and MI enabled (chained)
5. Verify saved matrices have correct filenames
6. Check that all defaults work correctly

## Migration Notes
- No CLI changes required
- No changes to saved model format
- Fully backward compatible with existing scripts
- Test set automatically gets `None` configs (optimization disabled)

# Blahut-Arimoto Transition Matrix Augmentation

## Overview

This feature enables augmentation of transition matrices used for complementary label generation using the **Blahut-Arimoto (BA) algorithm**. The BA algorithm is a classical iterative method from information theory that computes channel capacity and optimal input distributions for discrete memoryless channels.

## Purpose

When generating complementary labels from a transition matrix, you may want to "smooth" or augment the matrix to:
- Reduce overfitting to a specific transition pattern
- Increase diversity in label generation
- Leverage information-theoretic insights about the channel structure
- Create more robust training data by controlled interpolation toward uniform or optimal distributions

## How It Works

### Algorithm Overview

1. **Blahut-Arimoto Computation**
   - Input: Transition matrix `T[x,y]` representing `P(complementary_label=y | true_label=x)`
   - Computes: Optimal input distribution `p*(x)` that maximizes mutual information `I(X;Y)`
   - Returns: Channel capacity `C`, optimal distribution `p*`, and output distribution `q(y)`

2. **Row-Specific Augmentation Strengths**
   - Uses `p*(x)` to compute per-row augmentation strength
   - Formula: `s_x = normalize((max(p*) - p*(x))^gamma)`
   - Interpretation: Classes with lower `p*(x)` (less informative) get stronger augmentation
   - Parameter `gamma` controls contrast between rows (higher = more contrast)

3. **Matrix Augmentation**
   - For each row `x`, interpolate between original `T[x,:]` and target distribution `U[x,:]`
   - Formula: `T_aug[x,:] = (1 - alpha * s_x) * T[x,:] + (alpha * s_x) * U[x,:]`
   - Where `alpha` is global strength parameter
   - Renormalize each row to ensure valid probability distribution

4. **Target Distributions**
   - **Uniform mode** (`ba_row_mode='uniform'`): Target is uniform distribution (default)
   - **Q mode** (`ba_row_mode='q'`): Target is BA output distribution `q(y)`

5. **Label Generation**
   - Use augmented matrix `T_aug` instead of original `T` for sampling complementary labels
   - Save augmented matrix for reproducibility and analysis

## Usage

### Basic Usage

```python
from imb_cll.dataset.cifar import CIFAR10_CLL
import numpy as np

# Load your transition matrix
T = np.loadtxt('transition_matrix/cifar10/my_matrix.txt')

# Create dataset with BA augmentation enabled
dataset = CIFAR10_CLL(
    root='./data',
    train=True,
    cll_type='from_matrix_least',
    # ... other parameters ...
)

# Generate complementary labels with BA augmentation
dataset.generate_cl_from_matrix(
    transition_matrix=T,
    use_blahut=True,          # Enable BA augmentation
    ba_strength=0.1,          # Global strength (0-1)
    ba_row_mode='uniform',    # Target: uniform distribution
    ba_gamma=1.0,             # Per-row contrast
    ba_save=True              # Save augmented matrix
)
```

### Advanced Usage

```python
# High augmentation with increased row contrast
dataset.generate_cl_from_matrix(
    transition_matrix=T,
    use_blahut=True,
    ba_strength=0.3,              # Stronger smoothing
    ba_row_mode='uniform',        
    ba_gamma=2.0,                 # Higher contrast between rows
    ba_preserve_diagonal=True,    # Keep self-label probabilities
    ba_max_iters=1000,            # More iterations for convergence
    ba_tol=1e-8,                  # Tighter tolerance
    ba_save=True
)
```

```python
# Use BA output distribution as target
dataset.generate_cl_from_matrix(
    transition_matrix=T,
    use_blahut=True,
    ba_strength=0.15,
    ba_row_mode='q',              # Use q(y) from BA as target
    ba_gamma=1.5,
    ba_save=True
)
```

## Parameters

### `use_blahut` (bool, default=False)
- Enable/disable BA augmentation
- When `False`, uses original transition matrix directly
- When `True`, applies BA augmentation before label generation

### `ba_strength` (float, default=0.1)
- Global augmentation strength in range [0, 1]
- 0.0 = no augmentation (same as `use_blahut=False`)
- 1.0 = maximum augmentation (full interpolation to target)
- Typical values: 0.05-0.3
- **Recommendation**: Start with 0.1 and adjust based on validation performance

### `ba_row_mode` (str, default='uniform')
- Target distribution for augmentation
- **'uniform'**: Smooth toward uniform distribution over all classes
  - Good for: Reducing bias, increasing exploration
- **'q'**: Smooth toward BA output distribution q(y) = Σ_x p*(x)T[x,y]
  - Good for: Information-theoretic optimal smoothing
- **Recommendation**: Use 'uniform' for most cases

### `ba_gamma` (float, default=1.0)
- Exponent controlling per-row strength contrast
- Formula: `s_x ∝ (max(p*) - p*(x))^gamma`
- gamma < 1.0: Reduces contrast (more uniform augmentation across rows)
- gamma = 1.0: Linear contrast
- gamma > 1.0: Increases contrast (stronger differentiation)
- Typical values: 0.5-2.0
- **Recommendation**: Use 1.0 unless you want to emphasize/de-emphasize class differences

### `ba_max_iters` (int, default=500)
- Maximum iterations for BA algorithm convergence
- Typical convergence: 10-100 iterations
- 500 is conservative; increase only if convergence warnings appear

### `ba_tol` (float, default=1e-6)
- Convergence tolerance (L1 norm of input distribution change)
- Smaller = tighter convergence, more iterations
- 1e-6 is standard; use 1e-8 for high precision

### `ba_preserve_diagonal` (bool, default=False)
- Whether to preserve diagonal elements (self-label probabilities) during augmentation
- `False`: Augment all elements including diagonal
- `True`: Keep `T[x,x]` unchanged, only augment off-diagonal
- **Recommendation**: Use `False` unless you specifically want to preserve self-label rates

### `ba_save` (bool, default=True)
- Whether to save augmented transition matrix to file
- Saved to: `transition_matrix/{dataset}/{cll_type}_augmented_ba.txt`
- Format: Space-delimited text, 6 decimal places
- **Recommendation**: Keep `True` for reproducibility

## Output Files

### Augmented Transition Matrix
- **Path**: `transition_matrix/{dataset}/{cll_type}_augmented_ba.txt`
- **Format**: n×n matrix, space-delimited, 6 decimal precision
- **Example**: `transition_matrix/cifar10/least_augmented_ba.txt`

### Generated Labels
- **Path**: `generated_labels/{dataset}/{cll_type}[prompt].txt`
- **Format**: One integer per line (complementary label for each sample)
- **Note**: This is the standard output; augmentation only affects the transition matrix used

## Example Workflow

```python
import numpy as np
from imb_cll.dataset.cifar import CIFAR10_CLL

# 1. Load or define transition matrix
T_original = np.loadtxt('transition_matrix/cifar10/least.txt')
print("Original matrix shape:", T_original.shape)

# 2. Create dataset
dataset = CIFAR10_CLL(
    root='./data',
    train=True,
    cll_type='from_matrix_least',
    imb_type='exp',
    imb_factor=0.01
)

# 3. Generate CL with BA augmentation
dataset.generate_cl_from_matrix(
    transition_matrix=T_original,
    use_blahut=True,
    ba_strength=0.15,
    ba_row_mode='uniform',
    ba_gamma=1.0,
    ba_save=True
)

# 4. Check outputs
print(f"Generated {len(dataset.targets)} complementary labels")
print(f"True targets preserved: {len(dataset.true_targets)}")

# 5. Load augmented matrix for analysis
T_augmented = np.loadtxt('transition_matrix/cifar10/from_matrix_least_augmented_ba.txt')
print("\nDifference between original and augmented:")
print("Max absolute diff:", np.abs(T_original - T_augmented).max())
print("Mean absolute diff:", np.abs(T_original - T_augmented).mean())

# 6. Visualize (optional)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(T_original, cmap='viridis')
axes[0].set_title('Original Matrix')
axes[1].imshow(T_augmented, cmap='viridis')
axes[1].set_title('Augmented Matrix')
axes[2].imshow(T_augmented - T_original, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
axes[2].set_title('Difference')
plt.colorbar(axes[2].images[0])
plt.show()
```

## Algorithm Details

### Blahut-Arimoto Iteration

Given transition matrix `T[x,y]` (rows are input, columns are output):

```
Initialize: p(x) = uniform distribution over x

Repeat until convergence:
  1. Compute output distribution: q(y) = Σ_x p(x) * T[x,y]
  2. Compute exponential weights: r(x) = exp(Σ_y T[x,y] * log(T[x,y]/q(y)))
  3. Update input: p(x) = r(x) / Σ_x r(x)
  4. Check: ||p_new - p_old|| < tolerance

Compute capacity: C = Σ_x p(x) * Σ_y T[x,y] * log(T[x,y]/q(y))
```

Convergence: Typically 10-100 iterations; guaranteed to converge to global optimum.

### Row Strength Computation

```python
p_max = max(p_star)
raw_strengths = (p_max - p_star) ** gamma
row_strengths = raw_strengths / max(raw_strengths)  # Normalize to [0, 1]
```

Interpretation:
- Rows with high `p*(x)` (more informative) get low strength (less augmentation)
- Rows with low `p*(x)` (less informative) get high strength (more augmentation)
- `gamma` controls how much this difference is amplified

### Augmentation Formula

For each row `x`:
```python
alpha_x = ba_strength * row_strengths[x]  # Per-row effective strength
T_aug[x,:] = (1 - alpha_x) * T[x,:] + alpha_x * U[x,:]
T_aug[x,:] = T_aug[x,:] / sum(T_aug[x,:])  # Renormalize
```

Where `U[x,:]` is the target distribution (uniform or q).

## Interpretation & Intuition

### Why Blahut-Arimoto?

The BA algorithm finds the input distribution that maximizes information flow through the channel `T`. Classes with higher `p*(x)` are more "informative" or "important" for the channel. We use this as a principled way to decide which rows to augment more/less.

### Information-Theoretic View

- **Capacity**: Maximum bits transmitted per channel use
- **p*(x)**: Optimal code book (how often to use each class)
- High `p*(x)` → class is important for channel → preserve its transition pattern
- Low `p*(x)` → class is less important → can safely smooth it more

### Practical Effects

- **ba_strength=0.1**: Subtle smoothing, mostly preserves original structure
- **ba_strength=0.3**: Moderate smoothing, noticeable movement toward target
- **ba_strength=0.5**: Strong smoothing, significant deviation from original
- **ba_row_mode='uniform'**: Reduces class-specific bias in transitions
- **ba_row_mode='q'**: Preserves information-theoretic structure while smoothing

## When to Use

### Use BA Augmentation When:
- ✓ Transition matrix is estimated/noisy and may overfit
- ✓ Want to increase robustness via controlled smoothing
- ✓ Exploring augmentation strategies systematically
- ✓ Working with imbalanced classes (BA respects importance)
- ✓ Have validation set to tune `ba_strength` and `ba_gamma`

### Don't Use BA Augmentation When:
- ✗ Transition matrix is ground truth and should be preserved exactly
- ✗ Already using strong data augmentation (may be redundant)
- ✗ Computational budget is extremely tight (BA adds ~1-2 seconds per call)
- ✗ Dataset is very small (augmentation may not help)

## Tuning Guide

### Step 1: Baseline
Run without BA augmentation to get baseline performance.

### Step 2: Conservative Start
```python
use_blahut=True, ba_strength=0.05, ba_gamma=1.0
```

### Step 3: Grid Search (optional)
- `ba_strength`: [0.05, 0.1, 0.15, 0.2, 0.3]
- `ba_gamma`: [0.5, 1.0, 1.5, 2.0]
- `ba_row_mode`: ['uniform', 'q']

### Step 4: Validate
Check validation accuracy and compare transition matrices visually.

### Step 5: Production
Use best settings from validation; ensure `ba_save=True` for reproducibility.

## Implementation Details

### Files Modified/Created

1. **`imb_cll/utils/ba_utils.py`** (new)
   - `blahut_arimoto()`: Core BA algorithm
   - `augment_transition_matrix_with_ba()`: Matrix augmentation wrapper

2. **`imb_cll/dataset/base_dataset.py`** (modified)
   - `generate_cl_from_matrix()`: Extended with BA parameters

### Dependencies
- NumPy (standard in project)
- No new external dependencies

### Numerical Stability
- Small epsilon (1e-12) added to prevent log(0)
- Row normalization enforced after augmentation
- Convergence monitoring with warnings

### Reproducibility
- Uses `np.random.default_rng(seed)` for deterministic label sampling
- BA algorithm is deterministic (no randomness)
- Saved matrices enable exact reproduction

## References

### Original Papers
- Blahut, R. (1972). "Computation of channel capacity and rate-distortion functions." IEEE Transactions on Information Theory.
- Arimoto, S. (1972). "An algorithm for computing the capacity of arbitrary discrete memoryless channels." IEEE Transactions on Information Theory.

### Related Concepts
- Shannon channel capacity
- Mutual information maximization
- Rate-distortion theory
- Complementary label learning

## Troubleshooting

### Warning: BA did not converge
**Cause**: `max_iters` too low or `tol` too tight
**Solution**: Increase `ba_max_iters` to 1000 or relax `ba_tol` to 1e-5

### Warning: Row sums are not exactly 1
**Cause**: Numerical errors in input matrix
**Solution**: Normal; algorithm automatically renormalizes

### Labels look too uniform
**Cause**: `ba_strength` too high
**Solution**: Reduce to 0.05-0.15

### No visible difference in augmented matrix
**Cause**: `ba_strength` too low or `ba_gamma` too small
**Solution**: Increase `ba_strength` to 0.2-0.3 or `ba_gamma` to 1.5-2.0

### TypeError: target is np.array
**Cause**: Old code didn't handle array targets
**Solution**: Updated code now handles both int and array targets automatically

## Examples

See `notebooks/ba_augmentation_demo.ipynb` (to be created) for:
- Visual comparison of original vs augmented matrices
- Effect of different `ba_strength` values
- Training curves with/without augmentation
- Analysis of `p*` and row strengths

## Future Enhancements

Potential extensions (not yet implemented):
- [ ] Class-specific strength multipliers
- [ ] Adaptive strength based on class imbalance
- [ ] Multiple augmented matrices with ensemble
- [ ] Learned augmentation parameters
- [ ] Integration with other matrix augmentation methods

## Contact

For questions or issues with BA augmentation:
- Check this documentation first
- Review code comments in `ba_utils.py`
- Consult original BA papers for theoretical details

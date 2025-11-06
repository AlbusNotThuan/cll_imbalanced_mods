# Mutual Information Maximization for CLL Transition Matrices

## Overview

This module implements **gradient ascent optimization** to maximize the mutual information $I(Y;C)$ between true labels $Y$ and complementary labels $C$ for a given transition matrix $Q(c|y)$, while respecting all Complementary Label Learning (CLL) constraints.

## Purpose

When working with CLL transition matrices, you may want to optimize them to:
- **Maximize information content**: More informative complementary labels lead to better learning
- **Preserve CLL constraints**: Zero diagonal, row-stochastic, non-negative
- **Control deviation**: Optional budget constraint limits how much the matrix can change
- **Improve model performance**: Higher $I(Y;C)$ can lead to better downstream classification

## Theoretical Background

### Mutual Information

The mutual information between true labels $Y$ and complementary labels $C$ is:

$$I(Y;C) = \sum_{y} \sum_{c} P(y) \cdot Q(c|y) \cdot \log\frac{Q(c|y)}{P(c)}$$

where:
- $P(y)$ is the prior distribution over true labels
- $Q(c|y)$ is the transition matrix (probability of complementary label $c$ given true label $y$)
- $P(c) = \sum_y P(y) \cdot Q(c|y)$ is the marginal distribution over complementary labels

### Gradient

The gradient of $I(Y;C)$ with respect to $Q(c|y)$ is:

$$\frac{\partial I}{\partial Q(c|y)} = P(y) \cdot \left[\log Q(c|y) - \log P(c)\right]$$

## CLL Constraints

### Hard Constraints (Always Enforced)

1. **Zero Diagonal**: $Q(y|y) = 0$ for all $y$
   - A complementary label cannot be the true label
   
2. **Non-Negativity**: $Q(c|y) \geq 0$ for all $c, y$
   - Probabilities must be non-negative
   
3. **Row-Stochastic**: $\sum_c Q(c|y) = 1$ for all $y$
   - Each row must sum to 1 (valid probability distribution)

### Soft Constraint (Optional)

4. **Budget Constraint**: $\|Q - Q_{\text{baseline}}\|_F \leq B$
   - Limits the Frobenius norm of the deviation from the baseline
   - Controls how much the optimization can modify the original matrix

## Implementation

### Core Function: `optimize_cll_matrix_mi()`

```python
from imb_cll.utils.mi_optimization import optimize_cll_matrix_mi

Q_optimized, max_mi, history = optimize_cll_matrix_mi(
    Q_initial,                    # Initial transition matrix (N×N)
    learning_rate=0.01,           # Gradient ascent step size
    epsilon_convergence=1e-6,     # Convergence tolerance
    max_iterations=1000,          # Maximum iterations
    budget_B=None,                # Optional budget constraint
    P_y_true=None,                # Optional true label prior (default: uniform)
    verbose=True                  # Print progress
)
```

### Algorithm Steps

1. **Initial Validation**
   - Enforce zero diagonal
   - Ensure non-negativity
   - Normalize rows → creates $Q_{\text{baseline}}$

2. **Iterative Optimization Loop**
   ```
   for iteration in range(max_iterations):
       a. Calculate P(c) = sum_y P(y) * Q(c|y)
       b. Calculate gradient: ∇I[y,c] = P(y) * (log(Q[y,c]) - log(P[c]))
       c. Update: Q_temp = Q_current + alpha * ∇I
       d. Project onto CLL constraints (zero diagonal, non-negative, row-stochastic)
       e. Apply budget constraint if specified
       f. Check convergence: if ΔMI < ε, stop
   ```

3. **Return Results**
   - Optimized matrix $Q$
   - Maximum MI achieved
   - History of MI values per iteration

## Usage Examples

### Example 1: Basic Optimization (No Budget)

```python
import numpy as np
from imb_cll.utils.mi_optimization import optimize_cll_matrix_mi

# Load transition matrix
Q = np.loadtxt('transition_matrix/cifar10/random.txt')

# Optimize without budget constraint
Q_opt, mi_max, history = optimize_cll_matrix_mi(
    Q,
    learning_rate=0.05,
    epsilon_convergence=1e-6,
    max_iterations=2000,
    verbose=True
)

# Save optimized matrix
np.savetxt('transition_matrix/cifar10/random_mi_optimized.txt', Q_opt, fmt='%.6f')

print(f"MI improvement: {mi_max - history[0]:.6f} nats")
```

### Example 2: With Budget Constraint

```python
# Optimize with budget constraint (limits deviation from original)
Q_opt, mi_max, history = optimize_cll_matrix_mi(
    Q,
    learning_rate=0.05,
    epsilon_convergence=1e-6,
    max_iterations=2000,
    budget_B=0.5,  # Max Frobenius norm of deviation
    verbose=True
)

# Check final deviation
deviation = np.linalg.norm(Q_opt - Q, 'fro')
print(f"Final deviation: {deviation:.6f} (budget: 0.5)")
```

### Example 3: With Custom Prior Distribution

```python
# Use non-uniform prior (e.g., class-imbalanced dataset)
P_y = np.array([0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1])  # 10 classes

Q_opt, mi_max, history = optimize_cll_matrix_mi(
    Q,
    learning_rate=0.05,
    epsilon_convergence=1e-6,
    max_iterations=2000,
    P_y_true=P_y,
    verbose=True
)
```

### Example 4: Compare Original vs Optimized

```python
from imb_cll.utils.mi_optimization import compare_matrices

stats = compare_matrices(Q_original, Q_optimized, P_y=None)

print(f"MI improvement: {stats['mi_improvement']:.6f} nats")
print(f"Frobenius norm: {stats['frobenius_norm']:.6f}")
print(f"Max difference: {stats['max_abs_diff']:.6f}")
```

## Parameters

### `Q_initial` (required)
- **Type**: np.ndarray of shape (N, N)
- **Description**: Initial transition matrix where Q[y, c] = P(c|y)
- **Note**: May not satisfy CLL constraints; will be automatically corrected

### `learning_rate` (default=0.01)
- **Type**: float
- **Range**: Typically 0.01 to 0.1
- **Description**: Step size for gradient ascent (α)
- **Tuning**: 
  - Too large → oscillations, instability
  - Too small → slow convergence
  - **Recommended**: Start with 0.05, adjust based on convergence

### `epsilon_convergence` (default=1e-6)
- **Type**: float
- **Description**: Convergence tolerance for change in MI
- **Stops when**: $\Delta MI < \epsilon$
- **Recommended**: 1e-6 for standard precision, 1e-8 for high precision

### `max_iterations` (default=1000)
- **Type**: int
- **Description**: Maximum number of iterations to prevent infinite loops
- **Typical**: Converges in 50-500 iterations
- **Recommended**: 1000 to 2000

### `budget_B` (default=None)
- **Type**: float or None
- **Description**: Maximum allowed Frobenius norm $\|Q_{\text{optimized}} - Q_{\text{baseline}}\|_F$
- **Effect**:
  - `None`: No constraint, optimize freely
  - Small (e.g., 0.1-0.3): Conservative changes
  - Medium (e.g., 0.5-1.0): Moderate changes
  - Large (e.g., 2.0+): Aggressive changes
- **Use when**: You want to limit how much the matrix changes from the original

### `P_y_true` (default=None)
- **Type**: np.ndarray of shape (N,) or None
- **Description**: Prior distribution over true labels P(y)
- **Default**: Uniform distribution (1/N for all y)
- **Use when**: Dataset has class imbalance or known prior

### `verbose` (default=True)
- **Type**: bool
- **Description**: Whether to print progress information
- **Prints**: Initial state, iteration progress (every 100 iters), final statistics

## Returns

### `Q_optimized`
- **Type**: np.ndarray of shape (N, N)
- **Description**: Optimized transition matrix
- **Guarantees**: Satisfies all CLL constraints and budget (if specified)

### `max_mutual_information`
- **Type**: float
- **Description**: Maximum MI achieved (in nats)
- **Convert to bits**: Divide by `np.log(2)`

### `history_mi`
- **Type**: list of float
- **Description**: MI value at each iteration
- **Use for**: Plotting convergence, analyzing optimization trajectory

## Output Files

### Optimized Transition Matrix
```python
# Save optimized matrix
np.savetxt('path/to/matrix_mi_optimized.txt', Q_optimized, fmt='%.6f')
```
- **Format**: Space-delimited text, 6 decimal places
- **Properties**: Row-stochastic, zero diagonal, non-negative

### Convergence Plot (Optional)
```python
import matplotlib.pyplot as plt

plt.plot(history_mi)
plt.xlabel('Iteration')
plt.ylabel('Mutual Information (nats)')
plt.title('MI Optimization Convergence')
plt.savefig('convergence.png')
```

## Tuning Guide

### Step 1: Start Conservative
```python
Q_opt, mi_max, history = optimize_cll_matrix_mi(
    Q, learning_rate=0.05, max_iterations=1000
)
```

### Step 2: Check Convergence
- **Converged too early** (< 50 iterations): Increase `learning_rate`
- **Oscillating**: Decrease `learning_rate`
- **Not converging**: Increase `max_iterations` or adjust `learning_rate`

### Step 3: Tune Learning Rate
Try: `[0.01, 0.02, 0.05, 0.1]` and observe:
- MI improvement
- Number of iterations to convergence
- Stability (no large oscillations)

### Step 4: Add Budget (Optional)
```python
# Try different budgets
for budget in [0.2, 0.5, 1.0, 2.0]:
    Q_opt, mi_max, _ = optimize_cll_matrix_mi(
        Q, learning_rate=0.05, budget_B=budget
    )
    print(f"Budget {budget}: MI = {mi_max:.6f}")
```

### Step 5: Validate on Downstream Task
- Train CLL model with original matrix
- Train CLL model with optimized matrix
- Compare validation accuracy

## Integration with Training

### Standalone Script
```bash
# Run test script
python test_mi_optimization.py
```

### In Training Pipeline
```python
# In train.py or dataset preparation
if args.optimize_mi:
    from imb_cll.utils.mi_optimization import optimize_cll_matrix_mi
    
    transition_matrix, _, _ = optimize_cll_matrix_mi(
        transition_matrix,
        learning_rate=args.mi_learning_rate,
        budget_B=args.mi_budget,
        verbose=True
    )
```

## Comparison with Blahut-Arimoto

| Feature | MI Optimization | Blahut-Arimoto |
|---------|-----------------|----------------|
| **Goal** | Maximize $I(Y;C)$ | Compute channel capacity |
| **Method** | Gradient ascent | Iterative algorithm |
| **Output** | Optimized $Q(c\|y)$ | Optimal $p^*(x)$ + augmented $Q$ |
| **Constraints** | CLL + optional budget | Row-stochastic only |
| **Use Case** | Direct matrix optimization | Matrix smoothing/augmentation |
| **Speed** | ~1-2 sec for 10×10 | ~1-2 sec for 10×10 |

## When to Use

### Use MI Optimization When:
- ✓ Want to maximize information content of CL
- ✓ Have a baseline matrix that can be improved
- ✓ Need precise control over deviation (budget)
- ✓ Want theoretically grounded optimization
- ✓ Can validate improvement on downstream task

### Don't Use When:
- ✗ Matrix is already optimal or empirically derived
- ✗ Computational budget is extremely tight
- ✗ No validation set to confirm improvement
- ✗ Matrix structure must be preserved exactly

## Numerical Stability

1. **Epsilon additions**: Prevents log(0) errors
   ```python
   Q_safe = Q + eps  # eps = 1e-10
   ```

2. **Row normalization**: Enforced after every update
   ```python
   Q[y, :] = Q[y, :] / Q[y, :].sum()
   ```

3. **Diagonal enforcement**: Always zero
   ```python
   for i in range(N):
       Q[i, i] = 0.0
   ```

4. **Constraint projection**: Applied after gradient step
   - Zero diagonal → Non-negative → Normalize rows → (Budget)

## Troubleshooting

### Issue: Not Converging
**Symptoms**: Reaches max_iterations without convergence
**Solutions**:
1. Decrease `learning_rate` (try 0.01 or 0.005)
2. Increase `max_iterations` (try 2000 or 5000)
3. Check if matrix is already near-optimal

### Issue: Oscillating MI
**Symptoms**: MI goes up and down, doesn't stabilize
**Solutions**:
1. Decrease `learning_rate` significantly (try 0.005)
2. Check for numerical issues in gradient calculation
3. Try tighter convergence tolerance

### Issue: Small MI Improvement
**Symptoms**: Converges but MI increase is negligible
**Possible Reasons**:
1. Matrix is already well-optimized
2. Budget constraint is too tight
3. Learning rate is too small

**Solutions**:
1. Remove or increase `budget_B`
2. Increase `learning_rate`
3. Check if matrix has inherent structure preventing improvement

### Issue: Budget Constraint Not Satisfied
**Symptoms**: Final deviation > budget_B
**This shouldn't happen** - check implementation
**Debug**: Add assertions to verify budget enforcement

## Testing

### Unit Test Example
```python
import numpy as np
from imb_cll.utils.mi_optimization import (
    optimize_cll_matrix_mi,
    enforce_cll_constraints,
    calculate_mutual_information
)

def test_constraints():
    """Test that CLL constraints are enforced."""
    N = 10
    Q = np.random.rand(N, N)
    
    Q_const = enforce_cll_constraints(Q)
    
    # Check diagonal
    assert np.allclose(np.diag(Q_const), 0), "Diagonal not zero"
    
    # Check row sums
    row_sums = Q_const.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Rows don't sum to 1"
    
    # Check non-negativity
    assert np.all(Q_const >= 0), "Negative elements found"
    
    print("✓ All constraints satisfied")

def test_optimization():
    """Test that MI increases after optimization."""
    Q = np.loadtxt('transition_matrix/cifar10/random.txt')
    P_y = np.ones(10) / 10
    
    mi_before = calculate_mutual_information(Q, P_y)
    
    Q_opt, mi_after, _ = optimize_cll_matrix_mi(
        Q, learning_rate=0.05, max_iterations=500, verbose=False
    )
    
    assert mi_after > mi_before, "MI did not increase"
    print(f"✓ MI increased from {mi_before:.4f} to {mi_after:.4f}")

if __name__ == '__main__':
    test_constraints()
    test_optimization()
```

## References

### Information Theory
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley.
- Chapter on mutual information and channel capacity

### Complementary Label Learning
- Ishida, T., et al. (2017). "Learning from Complementary Labels." NeurIPS.
- Yu, X., et al. (2018). "Learning with Biased Complementary Labels." ECCV.

### Optimization
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge.
- Gradient ascent and constrained optimization

## Future Enhancements

- [ ] Adaptive learning rate (e.g., Adam optimizer)
- [ ] Multi-start optimization (different initializations)
- [ ] Annealing schedule for budget constraint
- [ ] Parallel optimization for multiple matrices
- [ ] Integration into main training pipeline via CLI args
- [ ] Automatic hyperparameter tuning via grid search

---

**Status**: ✅ Implemented and ready for testing
**File**: `imb_cll/utils/mi_optimization.py`
**Test Script**: `test_mi_optimization.py`

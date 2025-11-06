"""
Mutual Information Maximization for CLL Transition Matrix Optimization.

This module implements gradient ascent to maximize I(Y;C) where Y is the true label
and C is the complementary label, subject to CLL constraints:
1. Q(y|y) = 0 (diagonal must be zero)
2. Q(c|y) >= 0 for all c, y
3. sum_c Q(c|y) = 1 for all y (row stochastic)
4. Optional budget constraint: ||Q - Q_baseline||_F <= B
"""

import numpy as np


def calculate_mutual_information(Q, P_y, eps=1e-10):
    """
    Calculate mutual information I(Y;C) for a CLL transition matrix.
    
    I(Y;C) = sum_y sum_c P(y) * Q(c|y) * log(Q(c|y) / P(c))
    where P(c) = sum_y P(y) * Q(c|y)
    
    Parameters:
    -----------
    Q : np.ndarray
        Transition matrix of shape (N, N) where Q[y, c] = P(c|y).
        Row-stochastic with zero diagonal.
    P_y : np.ndarray
        Prior distribution over true labels of shape (N,).
    eps : float, optional
        Small constant for numerical stability. Default: 1e-10
    
    Returns:
    --------
    mi : float
        Mutual information in nats.
    """
    N = Q.shape[0]
    
    # Calculate marginal P(c) = sum_y P(y) * Q(c|y)
    P_c = np.dot(P_y, Q)  # shape (N,)
    
    # Add epsilon for numerical stability
    Q_safe = Q + eps
    P_c_safe = P_c + eps
    
    # Calculate I(Y;C) = sum_y sum_c P(y) * Q(c|y) * log(Q(c|y) / P(c))
    # Vectorized: sum over all elements of P_y[:, None] * Q * log(Q / P_c[None, :])
    log_ratio = np.log(Q_safe / P_c_safe[np.newaxis, :])
    mi = np.sum(P_y[:, np.newaxis] * Q * log_ratio)
    
    return mi


def enforce_cll_constraints(Q, eps=1e-10, min_off_diagonal=0.003):
    """
    Enforce hard CLL constraints on a matrix.
    
    Constraints:
    1. Diagonal elements must be zero: Q[y, y] = 0
    2. Off-diagonal elements must be >= min_off_diagonal: Q[y, c] >= 0.003 for y != c
    3. Rows must sum to 1: sum_c Q(c|y) = 1
    
    Parameters:
    -----------
    Q : np.ndarray
        Matrix of shape (N, N).
    eps : float, optional
        Small positive constant for numerical stability. Default: 1e-10
    min_off_diagonal : float, optional
        Minimum value for off-diagonal elements. Default: 0.003
    
    Returns:
    --------
    Q_constrained : np.ndarray
        Matrix satisfying all CLL constraints.
    """
    N = Q.shape[0]
    Q_constrained = Q.copy()
    
    # Step 1: Enforce zero diagonal
    for i in range(N):
        Q_constrained[i, i] = 0.0
    
    # Step 2: Enforce minimum off-diagonal values
    # All off-diagonal elements must be >= min_off_diagonal
    for y in range(N):
        for c in range(N):
            if y != c:
                Q_constrained[y, c] = max(Q_constrained[y, c], min_off_diagonal)
    
    # Step 3: Re-normalize rows to sum to 1
    for y in range(N):
        row_sum = np.sum(Q_constrained[y, :])
        if row_sum < eps:
            # If row sum is too small, use uniform distribution over off-diagonal
            Q_constrained[y, :] = 1.0 / (N - 1)
            Q_constrained[y, y] = 0.0
        else:
            Q_constrained[y, :] = Q_constrained[y, :] / row_sum
            # Re-enforce zero diagonal after normalization (numerical safety)
            Q_constrained[y, y] = 0.0
            # Re-normalize after setting diagonal to 0
            row_sum = np.sum(Q_constrained[y, :])
            if row_sum > eps:
                Q_constrained[y, :] = Q_constrained[y, :] / row_sum
    
    # Step 4: Final verification - ensure off-diagonal minimums are maintained after normalization
    # This may require iterative adjustment
    max_iterations = 10
    for _ in range(max_iterations):
        needs_adjustment = False
        for y in range(N):
            # Check if any off-diagonal element fell below minimum
            for c in range(N):
                if y != c and Q_constrained[y, c] < min_off_diagonal:
                    Q_constrained[y, c] = min_off_diagonal
                    needs_adjustment = True
            
            # Re-normalize if we made adjustments
            if needs_adjustment:
                Q_constrained[y, y] = 0.0
                row_sum = np.sum(Q_constrained[y, :])
                if row_sum > eps:
                    Q_constrained[y, :] = Q_constrained[y, :] / row_sum
                    Q_constrained[y, y] = 0.0
        
        if not needs_adjustment:
            break
    
    return Q_constrained


def optimize_cll_matrix_mi(Q_initial, learning_rate=0.01, epsilon_convergence=1e-6,
                           max_iterations=1000, budget_B=None, P_y_true=None,
                           verbose=True, eps=1e-10, min_off_diagonal=0.003):
    """
    Optimize a CLL transition matrix to maximize mutual information I(Y;C).
    
    This function performs gradient ascent on the mutual information between
    true labels Y and complementary labels C, subject to CLL constraints.
    
    Parameters:
    -----------
    Q_initial : np.ndarray
        Initial transition matrix of shape (N, N) where Q[y, c] = P(c|y).
        May not satisfy CLL constraints; will be corrected.
    learning_rate : float, optional
        Step size for gradient ascent (alpha). Default: 0.01
        Typical range: 0.01 to 0.1
    epsilon_convergence : float, optional
        Convergence tolerance for change in MI. Default: 1e-6
        Algorithm stops when delta_mi < epsilon_convergence.
    max_iterations : int, optional
        Maximum number of iterations. Default: 1000
    budget_B : float, optional
        Maximum allowed Frobenius norm ||Q_optimized - Q_baseline||_F.
        If None, no budget constraint is applied. Default: None
    P_y_true : np.ndarray, optional
        True label prior distribution of shape (N,).
        If None, assumes uniform distribution. Default: None
    verbose : bool, optional
        Whether to print progress information. Default: True
    eps : float, optional
        Small constant for numerical stability. Default: 1e-10
    min_off_diagonal : float, optional
        Minimum value for off-diagonal elements. Default: 0.003
        All Q[y, c] where y != c must be >= min_off_diagonal.
    
    Returns:
    --------
    Q_optimized : np.ndarray
        Optimized transition matrix of shape (N, N).
    max_mutual_information : float
        Maximum mutual information achieved.
    history_mi : list
        List of mutual information values at each iteration.
    
    Algorithm:
    ----------
    0. Validate and enforce initial CLL constraints to get Q_baseline
    1. Initialize Q_current = Q_baseline
    2. Repeat until convergence or max_iterations:
       a. Calculate P(c) = sum_y P(y) * Q(c|y)
       b. Calculate gradient: ∇I[y,c] = P(y) * (log(Q[y,c]) - log(P[c]))
       c. Update: Q_temp = Q_current + alpha * ∇I
       d. Project onto CLL constraints (diagonal=0, off-diagonal>=0.003, row-stochastic)
       e. Apply budget constraint if specified
       f. Check convergence
    3. Return optimized matrix and history
    
    Example:
    --------
    >>> Q = np.loadtxt('transition_matrix.txt')
    >>> Q_opt, mi_max, history = optimize_cll_matrix_mi(
    ...     Q, learning_rate=0.05, max_iterations=2000)
    >>> print(f"Max MI: {mi_max:.4f} nats ({mi_max/np.log(2):.4f} bits)")
    """
    N = Q_initial.shape[0]
    
    # Validate input shape
    if Q_initial.shape[0] != Q_initial.shape[1]:
        raise ValueError(f"Q_initial must be square, got shape {Q_initial.shape}")
    
    # Step 0: Initial validation and enforce hard constraints
    if verbose:
        print("=" * 60)
        print("MUTUAL INFORMATION OPTIMIZATION FOR CLL MATRIX")
        print("=" * 60)
        print(f"Matrix size: {N}×{N}")
        print(f"Learning rate: {learning_rate}")
        print(f"Convergence tolerance: {epsilon_convergence}")
        print(f"Max iterations: {max_iterations}")
        print(f"Budget constraint: {budget_B if budget_B else 'None'}")
        print(f"Min off-diagonal value: {min_off_diagonal}")
        print()
    
    # Check and enforce diagonal constraint
    diagonal_violations = np.sum(np.diag(Q_initial) != 0)
    if diagonal_violations > 0 and verbose:
        print(f"Warning: {diagonal_violations} diagonal elements are non-zero. Setting to 0.")
    
    # Enforce all CLL constraints to create Q_baseline
    Q_baseline = enforce_cll_constraints(Q_initial, eps=eps, min_off_diagonal=min_off_diagonal)
    
    if verbose:
        print("Initial matrix validated and corrected to Q_baseline.")
        print(f"Q_baseline row sums: min={Q_baseline.sum(axis=1).min():.6f}, "
              f"max={Q_baseline.sum(axis=1).max():.6f}")
        print(f"Q_baseline diagonal sum: {np.trace(Q_baseline):.10f}")
        
        # Check off-diagonal minimum
        off_diag_min = float('inf')
        for i in range(N):
            for j in range(N):
                if i != j:
                    off_diag_min = min(off_diag_min, Q_baseline[i, j])
        print(f"Q_baseline min off-diagonal: {off_diag_min:.6f}")
        print()
    
    # Initialize P_y
    if P_y_true is None:
        P_y = np.ones(N) / N
        if verbose:
            print("Using uniform prior: P(y) = 1/N for all y")
    else:
        P_y = P_y_true.copy()
        if not np.isclose(P_y.sum(), 1.0):
            if verbose:
                print(f"Warning: P_y does not sum to 1 (sum={P_y.sum():.6f}). Normalizing.")
            P_y = P_y / P_y.sum()
        if verbose:
            print(f"Using provided prior: P(y) range [{P_y.min():.4f}, {P_y.max():.4f}]")
    
    # Step 1: Initialization for optimization
    Q_current = Q_baseline.copy()
    current_mi = calculate_mutual_information(Q_current, P_y, eps=eps)
    history_mi = [current_mi]
    
    if verbose:
        print(f"\nInitial MI: {current_mi:.6f} nats ({current_mi/np.log(2):.6f} bits)")
        print("\nStarting optimization...")
        print("-" * 60)
    
    # Step 2: Iterative optimization loop
    for iteration in range(max_iterations):
        # Calculate marginal P(c)
        P_c = np.dot(P_y, Q_current)  # shape (N,)
        P_c_safe = P_c + eps
        
        # Calculate gradient of mutual information
        # ∇I[y, c] = P(y) * (log(Q[y,c]) - log(P[c]))
        Q_safe = Q_current + eps
        log_Q = np.log(Q_safe)
        log_P_c = np.log(P_c_safe)
        
        grad_MI = P_y[:, np.newaxis] * (log_Q - log_P_c[np.newaxis, :])
        
        # Note: Gradient for diagonal should be zero since Q[y,y] must remain 0
        # We'll enforce this in the projection step
        
        # Gradient ascent step
        Q_temp = Q_current + learning_rate * grad_MI
        
        # Projection onto CLL constraints (hard constraints including min_off_diagonal)
        Q_temp = enforce_cll_constraints(Q_temp, eps=eps, min_off_diagonal=min_off_diagonal)
        
        # Apply budget constraint (soft constraint) if specified
        if budget_B is not None:
            diff_from_baseline = Q_temp - Q_baseline
            current_norm = np.linalg.norm(diff_from_baseline, 'fro')
            
            if current_norm > budget_B:
                # Scale back the difference
                scaled_diff = (budget_B / current_norm) * diff_from_baseline
                Q_projected = Q_baseline + scaled_diff
                
                # Re-apply CLL constraints after scaling
                Q_projected = enforce_cll_constraints(Q_projected, eps=eps, min_off_diagonal=min_off_diagonal)
            else:
                Q_projected = Q_temp
        else:
            Q_projected = Q_temp
        
        # Update current matrix
        Q_current = Q_projected
        
        # Calculate new mutual information
        new_mi = calculate_mutual_information(Q_current, P_y, eps=eps)
        delta_mi = new_mi - history_mi[-1]
        history_mi.append(new_mi)
        
        # Print progress
        if verbose and (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1:4d}: MI = {new_mi:.6f} nats "
                  f"({new_mi/np.log(2):.6f} bits), ΔMI = {delta_mi:.6e}")
        
        # Check convergence
        if iteration > 0 and delta_mi < epsilon_convergence:
            if verbose:
                print(f"\nConverged at iteration {iteration + 1}")
                print(f"ΔMI ({delta_mi:.6e}) < ε ({epsilon_convergence:.6e})")
            break
    else:
        if verbose:
            print(f"\nReached maximum iterations ({max_iterations})")
    
    # Final statistics
    max_mutual_information = history_mi[-1]
    
    if verbose:
        print("-" * 60)
        print("\nOptimization complete!")
        print(f"Final MI: {max_mutual_information:.6f} nats "
              f"({max_mutual_information/np.log(2):.6f} bits)")
        print(f"Initial MI: {history_mi[0]:.6f} nats "
              f"({history_mi[0]/np.log(2):.6f} bits)")
        print(f"MI improvement: {max_mutual_information - history_mi[0]:.6f} nats "
              f"({(max_mutual_information - history_mi[0])/np.log(2):.6f} bits)")
        
        if budget_B is not None:
            final_deviation = np.linalg.norm(Q_current - Q_baseline, 'fro')
            print(f"Final deviation from baseline: {final_deviation:.6f}")
            print(f"Budget B: {budget_B:.6f}")
            print(f"Budget utilization: {100 * final_deviation / budget_B:.2f}%")
        
        # Verify constraints
        row_sums = Q_current.sum(axis=1)
        diagonal_sum = np.trace(Q_current)
        
        # Check off-diagonal minimum
        off_diag_min = float('inf')
        off_diag_violations = 0
        for i in range(N):
            for j in range(N):
                if i != j:
                    if Q_current[i, j] < min_off_diagonal:
                        off_diag_violations += 1
                    off_diag_min = min(off_diag_min, Q_current[i, j])
        
        print(f"\nConstraint verification:")
        print(f"  Row sums: min={row_sums.min():.6f}, max={row_sums.max():.6f}")
        print(f"  Diagonal sum: {diagonal_sum:.10f}")
        print(f"  Min off-diagonal element: {off_diag_min:.6f}")
        print(f"  Off-diagonal violations (< {min_off_diagonal}): {off_diag_violations}")
        print(f"  Max element: {Q_current.max():.6f}")
        print("=" * 60)
    
    return Q_current, max_mutual_information, history_mi


def compare_matrices(Q_original, Q_optimized, P_y=None):
    """
    Compare original and optimized transition matrices.
    
    Parameters:
    -----------
    Q_original : np.ndarray
        Original transition matrix.
    Q_optimized : np.ndarray
        Optimized transition matrix.
    P_y : np.ndarray, optional
        Prior distribution. If None, uses uniform.
    
    Returns:
    --------
    stats : dict
        Dictionary containing comparison statistics.
    """
    N = Q_original.shape[0]
    
    if P_y is None:
        P_y = np.ones(N) / N
    
    # Calculate MIs
    mi_original = calculate_mutual_information(Q_original, P_y)
    mi_optimized = calculate_mutual_information(Q_optimized, P_y)
    
    # Calculate differences
    diff = Q_optimized - Q_original
    frobenius_norm = np.linalg.norm(diff, 'fro')
    max_abs_diff = np.abs(diff).max()
    mean_abs_diff = np.abs(diff).mean()
    
    stats = {
        'mi_original': mi_original,
        'mi_optimized': mi_optimized,
        'mi_improvement': mi_optimized - mi_original,
        'frobenius_norm': frobenius_norm,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
    }
    
    print("\nMatrix Comparison:")
    print(f"  Original MI: {mi_original:.6f} nats ({mi_original/np.log(2):.6f} bits)")
    print(f"  Optimized MI: {mi_optimized:.6f} nats ({mi_optimized/np.log(2):.6f} bits)")
    print(f"  MI improvement: {stats['mi_improvement']:.6f} nats "
          f"({stats['mi_improvement']/np.log(2):.6f} bits)")
    print(f"  Frobenius norm ||Q_opt - Q_orig||_F: {frobenius_norm:.6f}")
    print(f"  Max absolute difference: {max_abs_diff:.6f}")
    print(f"  Mean absolute difference: {mean_abs_diff:.6f}")
    
    return stats

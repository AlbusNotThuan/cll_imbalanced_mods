"""
Blahut-Arimoto Algorithm utilities for channel capacity computation.

The Blahut-Arimoto algorithm computes the channel capacity and optimal input
distribution for a discrete memoryless channel with transition matrix T[x,y].
"""

import numpy as np


def blahut_arimoto(T, tol=1e-6, max_iters=500, eps=1e-12):
    """
    Compute channel capacity using the Blahut-Arimoto algorithm.
    
    The algorithm iteratively computes the optimal input distribution p*(x) that
    maximizes the mutual information I(X;Y) for a channel P(y|x) = T[x,y].
    
    Parameters:
    -----------
    T : np.ndarray
        Transition matrix of shape (n, n) where T[x, y] = P(y|x).
        Must be row-stochastic (rows sum to 1) and non-negative.
    tol : float, optional
        Convergence tolerance. Algorithm stops when ||p_new - p|| < tol.
        Default: 1e-6
    max_iters : int, optional
        Maximum number of iterations. Default: 500
    eps : float, optional
        Small constant added for numerical stability (avoid log(0)).
        Default: 1e-12
    
    Returns:
    --------
    p_star : np.ndarray
        Optimal input distribution of shape (n,) that achieves capacity.
    capacity : float
        Channel capacity in nats (natural log). Divide by log(2) for bits.
    q : np.ndarray
        Output distribution q(y) = sum_x p*(x) * T[x,y] of shape (n,).
    converged : bool
        Whether the algorithm converged within max_iters.
    
    Algorithm:
    ----------
    1. Initialize p uniform over n symbols
    2. Repeat until convergence:
       a. Compute output distribution: q(y) = sum_x p(x) * T[x,y]
       b. Compute exponential weights: r(x) = exp(sum_y T[x,y] * log(T[x,y]/q(y)))
       c. Update input distribution: p(x) = r(x) / sum_x r(x)
    3. Compute capacity: C = sum_x p(x) * sum_y T[x,y] * log(T[x,y]/q(y))
    
    References:
    -----------
    - Blahut, R. (1972). "Computation of channel capacity and rate-distortion functions."
    - Arimoto, S. (1972). "An algorithm for computing the capacity of arbitrary discrete memoryless channels."
    """
    n = T.shape[0]
    
    # Validate input
    if T.shape[0] != T.shape[1]:
        raise ValueError(f"Transition matrix must be square, got shape {T.shape}")
    if np.any(T < 0):
        raise ValueError("Transition matrix must be non-negative")
    
    # Ensure rows sum to 1 (with tolerance for numerical errors)
    row_sums = T.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        print(f"Warning: Input matrix rows don't sum to 1. Row sums: {row_sums}")
        # Normalize rows
        T = T.copy()
        row_sums = row_sums.reshape(-1, 1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        T = T / row_sums
    
    # Add epsilon for numerical stability
    T_safe = T + eps
    
    # Initialize uniform input distribution
    p = np.ones(n) / n
    
    converged = False
    for iteration in range(max_iters):
        # Step 1: Compute output distribution q(y) = sum_x p(x) * T[x,y]
        q = np.dot(p, T_safe)  # shape (n,)
        q = q + eps  # Ensure no zeros for log
        
        # Step 2: Compute exponential weights r(x)
        # r(x) = exp(sum_y T[x,y] * log(T[x,y]/q(y)))
        log_ratio = np.log(T_safe / q[np.newaxis, :])  # shape (n, n)
        exponent = np.sum(T_safe * log_ratio, axis=1)  # shape (n,)
        r = np.exp(exponent)  # shape (n,)
        
        # Step 3: Update input distribution
        p_new = r / np.sum(r)
        
        # Check convergence
        delta = np.linalg.norm(p_new - p, ord=1)  # L1 norm
        if delta < tol:
            converged = True
            p = p_new
            break
        
        p = p_new
    
    # Compute final output distribution
    q = np.dot(p, T_safe)
    q = q + eps
    
    # Compute capacity: C = sum_x p(x) * sum_y T[x,y] * log(T[x,y]/q(y))
    log_ratio = np.log(T_safe / q[np.newaxis, :])
    capacity = np.sum(p[:, np.newaxis] * T_safe * log_ratio)
    
    if not converged:
        print(f"Warning: Blahut-Arimoto did not converge after {max_iters} iterations. "
              f"Final delta: {delta:.6e}")
    
    return p, capacity, q, converged


def augment_transition_matrix_with_ba(T, strength=0.1, row_mode='uniform', 
                                     gamma=1.0, tol=1e-6, max_iters=500, 
                                     preserve_diagonal=False, seed=None):
    """
    Augment a transition matrix using Blahut-Arimoto algorithm results.
    
    This function uses the BA algorithm to compute the optimal input distribution p*(x),
    then uses it to compute row-specific augmentation strengths. The matrix is augmented
    by smoothing each row toward a target distribution (e.g., uniform).
    
    Parameters:
    -----------
    T : np.ndarray
        Input transition matrix of shape (n, n). Row-stochastic.
    strength : float, optional
        Global augmentation strength in [0, 1]. Higher values mean more smoothing.
        Default: 0.1
    row_mode : str, optional
        Target distribution for smoothing:
        - 'uniform': Smooth toward uniform distribution (optionally excluding diagonal)
        - 'q': Smooth toward output distribution q from BA
        Default: 'uniform'
    gamma : float, optional
        Exponent controlling contrast in per-row strength computation.
        s_x = normalize((max(p*) - p*(x))^gamma)
        Higher gamma increases contrast between rows. Default: 1.0
    tol : float, optional
        BA convergence tolerance. Default: 1e-6
    max_iters : int, optional
        BA maximum iterations. Default: 500
    preserve_diagonal : bool, optional
        If True, preserve diagonal elements of T and only smooth off-diagonal.
        Default: False
    seed : int, optional
        Random seed for any stochastic operations (currently unused but reserved).
        Default: None
    
    Returns:
    --------
    T_aug : np.ndarray
        Augmented transition matrix of shape (n, n), row-stochastic.
    info : dict
        Dictionary containing:
        - 'p_star': Optimal input distribution from BA
        - 'capacity': Channel capacity
        - 'q': Output distribution
        - 'converged': Whether BA converged
        - 'row_strengths': Per-row augmentation strengths used
    
    Algorithm:
    ----------
    1. Run Blahut-Arimoto to get p*(x)
    2. Compute per-row strength: s_x = normalize((max(p*) - p*(x))^gamma)
       (Higher p*(x) means lower augmentation for that row)
    3. For each row x:
       T_aug[x,:] = (1 - alpha * s_x) * T[x,:] + (alpha * s_x) * U_x
       where U_x is the target distribution and alpha is the global strength
    4. Renormalize each row to ensure row-stochastic property
    """
    n = T.shape[0]
    
    # Run Blahut-Arimoto algorithm
    print("Running Blahut-Arimoto algorithm...")
    p_star, capacity, q, converged = blahut_arimoto(T, tol=tol, max_iters=max_iters)
    print(f"BA converged: {converged}, Capacity: {capacity:.6f} nats ({capacity/np.log(2):.6f} bits)")
    
    # Compute per-row augmentation strengths
    # Rows with lower p*(x) get higher augmentation strength
    p_max = np.max(p_star)
    raw_strengths = (p_max - p_star) ** gamma
    # Normalize to [0, 1] range
    if raw_strengths.max() > 0:
        row_strengths = raw_strengths / raw_strengths.max()
    else:
        row_strengths = np.zeros(n)
    
    # Determine target distribution for each row
    if row_mode == 'uniform':
        # Uniform distribution
        U = np.ones((n, n)) / n
        if preserve_diagonal:
            # Keep diagonal, uniform over off-diagonal
            for i in range(n):
                U[i, :] = 1.0 / (n - 1)
                U[i, i] = 0
    elif row_mode == 'q':
        # Use output distribution q for all rows
        U = np.tile(q, (n, 1))
        if preserve_diagonal:
            for i in range(n):
                U[i, :] = q.copy()
                U[i, i] = 0
                U[i, :] = U[i, :] / U[i, :].sum()  # Renormalize
    else:
        raise ValueError(f"Unknown row_mode: {row_mode}. Choose 'uniform' or 'q'.")
    
    # Augment matrix
    T_aug = T.copy()
    for x in range(n):
        alpha_x = strength * row_strengths[x]
        if preserve_diagonal:
            # Preserve diagonal element
            diag_val = T[x, x]
            # Smooth off-diagonal part
            off_diag_mask = np.ones(n, dtype=bool)
            off_diag_mask[x] = False
            T_aug[x, off_diag_mask] = ((1 - alpha_x) * T[x, off_diag_mask] + 
                                       alpha_x * U[x, off_diag_mask])
            T_aug[x, x] = diag_val
        else:
            T_aug[x, :] = (1 - alpha_x) * T[x, :] + alpha_x * U[x, :]
        
        # Renormalize row to ensure it sums to 1
        row_sum = T_aug[x, :].sum()
        if row_sum > 0:
            T_aug[x, :] = T_aug[x, :] / row_sum
        else:
            # If row is all zeros, replace with uniform
            T_aug[x, :] = 1.0 / n
    
    # Prepare info dict
    info = {
        'p_star': p_star,
        'capacity': capacity,
        'q': q,
        'converged': converged,
        'row_strengths': row_strengths,
    }
    
    return T_aug, info

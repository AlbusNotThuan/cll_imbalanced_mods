import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any
import json
from scipy import linalg

# Unified function to load data from array format files
def load_data_file(filename):
    """Load data from files containing arrays in various formats:
    - CIFAR-10 format: [value1, value2, ...] (single line with brackets)
    - CIFAR-20 format: array([value]) (multiple lines)
    """
    data = []
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
            
            # Check if it's CIFAR-10 format (single line with brackets)
            if content.startswith('[') and content.endswith(']'):
                # Remove brackets and split by comma
                content = content[1:-1]  # Remove brackets
                
                # Split by comma and extract numbers
                for item in content.split(','):
                    item = item.strip()
                    if item:
                        # Try different patterns to extract numbers
                        # Pattern 1: array([number])
                        match = re.search(r'array\(\[(\d+)\]\)', item)
                        if match:
                            data.append(int(match.group(1)))
                            continue
                        
                        # Pattern 2: np.int64(number)
                        match = re.search(r'np\.int64\((\d+)\)', item)
                        if match:
                            data.append(int(match.group(1)))
                            continue
                        
                        # Pattern 3: just a number
                        match = re.search(r'(\d+)', item)
                        if match:
                            data.append(int(match.group(1)))
            else:
                # CIFAR-20 format: multiple lines with array([number]) format
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        # Pattern 1: array([number])
                        match = re.search(r'array\(\[(\d+)\]\)', line)
                        if match:
                            data.append(int(match.group(1)))
                            continue
                        
                        # Pattern 2: np.int64(number)
                        match = re.search(r'np\.int64\((\d+)\)', line)
                        if match:
                            data.append(int(match.group(1)))
                            continue
                        
                        # Pattern 3: just a number
                        match = re.search(r'(\d+)', line)
                        if match:
                            data.append(int(match.group(1)))
                        
    except FileNotFoundError:
        print(f"{filename} not found. Returning empty list.")
    except Exception as e:
        print(f"Error reading {filename}: {e}")
    
    return data

def plot_dataframe_distributions(df_combined=None, figsize_per_col=5):
    """
    Plot histograms for each column in df_combined.
    
    Args:
        df_combined (pd.DataFrame, optional): DataFrame to plot. If None, will look for 'df_combined' in globals().
        figsize_per_col (int): Width in inches per subplot column.
    
    Example usage:
        # Use with explicit dataframe
        plot_dataframe_distributions(df_combined)
        
        # Use with automatic detection from globals
        plot_dataframe_distributions()
    """
    
    # Get df_combined either from parameter or globals
    if df_combined is None:
        if 'df_combined' in globals() and isinstance(globals()['df_combined'], pd.DataFrame):
            df_combined = globals()['df_combined']
        else:
            print("❌ No df_combined found. Please provide a DataFrame or ensure 'df_combined' exists in globals().")
            return
    
    if not isinstance(df_combined, pd.DataFrame):
        print("❌ Provided data is not a pandas DataFrame.")
        return
    
    cols = df_combined.columns.tolist()
    if not cols:
        print("⚠️ DataFrame is empty, nothing to plot.")
        return
    
    print(f"� Plotting histograms for df_combined")
    print(f"   Shape: {df_combined.shape}")
    print(f"   Columns: {cols}")
    
    # Create separate plots for each column
    for col in cols:
        series = df_combined[col].dropna()
        
        # Create individual figure for this column
        plt.figure(figsize=(figsize_per_col, 4), dpi=300)
        
        if len(series) == 0:
            plt.text(0.5, 0.5, f'No data in {col}', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f"{col} (empty)")
        else:
            # Use bar plot to show all classes with counts on top
            value_counts = series.value_counts().sort_index()
            
            # Ensure all classes from 0 to max are shown (fill missing with 0)
            if len(value_counts) > 0:
                min_class = int(value_counts.index.min())
                max_class = int(value_counts.index.max())
                all_classes = list(range(min_class, max_class + 1))
                value_counts = value_counts.reindex(all_classes, fill_value=0)
            
            # Create bar plot
            bars = plt.bar(value_counts.index, value_counts.values, color='steelblue', alpha=0.7)
            
            # Add count labels on top of each bar
            for i, (class_label, count) in enumerate(value_counts.items()):
                if count > 0:  # Only show label if count > 0
                    plt.text(class_label, count + max(value_counts.values) * 0.01, 
                            str(count), ha='center', va='bottom', fontsize=9)
            
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.title(f"{col}\n({len(value_counts)} classes, {series.nunique()} with data)")
            
            # Set x-axis to show all class labels
            plt.xticks(value_counts.index)
        
        plt.tight_layout()
        plt.show()
    
    # Print summary statistics
    print(f"📈 Summary for df_combined:")
    for col in cols:
        series = df_combined[col].dropna()
        if len(series) > 0:
            print(f"   {col}: {len(series)} values, {series.nunique()} unique, range [{series.min()}-{series.max()}]")
        else:
            print(f"   {col}: No data")
        print("-" * 50)


def analyze_matrix_properties(matrix, matrix_name="Matrix", verbose=True):
    """
    Analyze mathematical properties of a matrix including determinant, eigenvalues, and invertibility.
    
    Args:
        matrix (np.ndarray or pd.DataFrame): Input matrix to analyze
        matrix_name (str): Name of the matrix for display purposes. Default: "Matrix"
        verbose (bool): Whether to print detailed analysis. Default: True
        
    Returns:
        dict: Dictionary containing matrix analysis results:
            - determinant: float - matrix determinant
            - eigenvalues: np.ndarray - eigenvalues (complex numbers)
            - is_invertible: bool - whether matrix is invertible (det != 0)
            - is_singular: bool - whether matrix is singular (det == 0)
            - condition_number: float - condition number (measures how close to singular)
            - rank: int - matrix rank
            - is_square: bool - whether matrix is square
            - is_stochastic: bool - whether matrix is row-stochastic (rows sum to 1)
            - is_doubly_stochastic: bool - whether matrix is doubly stochastic (rows and cols sum to 1)
            - spectral_radius: float - largest absolute eigenvalue
            - trace: float - sum of diagonal elements
            - frobenius_norm: float - Frobenius norm of the matrix
            - markov_entropy: float - Markov chain entropy (for stochastic matrices)
            - normalized_markov_entropy: float - normalized Markov entropy (0-1 scale)
            - mutual_information: float - mutual information between past and future states
            - entropy_rate: float - entropy rate (conditional entropy H(X_n|X_{n-1}))
            - mixing_time: float - mixing time estimate (SLEM-based)
            - spectral_gap: float - spectral gap (1 - second largest eigenvalue magnitude)
    
    Example usage:
        # Analyze a transition matrix
        results = analyze_matrix_properties(transition_matrix, "CIFAR-10 Transition Matrix")
        
        # Get analysis without printing
        results = analyze_matrix_properties(transition_matrix, verbose=False)
        
        # Check specific properties
        if results['is_invertible']:
            print("Matrix is invertible!")
    """
    
    # Convert to numpy array if it's a DataFrame
    if isinstance(matrix, pd.DataFrame):
        matrix_array = matrix.values
    else:
        matrix_array = np.array(matrix)
    
    # Initialize results dictionary
    results = {
        'matrix_name': matrix_name,
        'shape': matrix_array.shape,
        'is_square': matrix_array.shape[0] == matrix_array.shape[1],
        'determinant': None,
        'eigenvalues': None,
        'is_invertible': False,
        'is_singular': True,
        'condition_number': None,
        'rank': None,
        'is_stochastic': False,
        'is_doubly_stochastic': False,
        'spectral_radius': None,
        'trace': None,
        'frobenius_norm': None,
        'markov_entropy': None,
        'normalized_markov_entropy': None,
        'mutual_information': None,
        'entropy_rate': None,
        'mixing_time': None,
        'spectral_gap': None
    }
    
    # Basic properties that work for any matrix
    results['rank'] = np.linalg.matrix_rank(matrix_array)
    results['frobenius_norm'] = np.linalg.norm(matrix_array, 'fro')
    
    # Check if matrix is row-stochastic (each row sums to 1)
    row_sums = np.sum(matrix_array, axis=1)
    results['is_stochastic'] = np.allclose(row_sums, 1.0, atol=1e-10)
    
    # Check if matrix is doubly stochastic (rows and columns sum to 1)
    if results['is_stochastic']:
        col_sums = np.sum(matrix_array, axis=0)
        results['is_doubly_stochastic'] = np.allclose(col_sums, 1.0, atol=1e-10)
    
    # Calculate Markov entropy for stochastic matrices
    if results['is_stochastic']:
        def calculate_markov_entropy(matrix, base=2):
            """Calculate Markov chain entropy for a stochastic matrix"""
            matrix = np.array(matrix)
            epsilon = 1e-10
            matrix_safe = matrix + epsilon
            
            # Calculate entropy for each row (state)
            row_entropies = []
            for row in matrix_safe:
                # Normalize row to ensure it sums to 1
                row_normalized = row / row.sum()
                # Calculate entropy: -sum(p * log(p))
                entropy = -np.sum(row_normalized * np.log(row_normalized) / np.log(base))
                row_entropies.append(entropy)
            
            # Return average entropy across all states
            return np.mean(row_entropies)
        
        def calculate_stationary_distribution(matrix):
            """Calculate stationary distribution of the Markov chain"""
            try:
                eigenvals, eigenvects = np.linalg.eig(matrix.T)
                # Find eigenvalue closest to 1
                idx = np.argmin(np.abs(eigenvals - 1.0))
                stationary = np.real(eigenvects[:, idx])
                # Normalize to get probability distribution
                stationary = np.abs(stationary) / np.sum(np.abs(stationary))
                return stationary
            except:
                # Fallback: uniform distribution
                return np.ones(matrix.shape[0]) / matrix.shape[0]
        
        def calculate_mutual_information(matrix, stationary_dist, base=2):
            """Calculate mutual information between consecutive states"""
            matrix = np.array(matrix)
            epsilon = 1e-10
            
            mutual_info = 0.0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if matrix[i, j] > epsilon:
                        # P(X_t = i, X_{t+1} = j)
                        joint_prob = stationary_dist[i] * matrix[i, j]
                        # P(X_t = i) * P(X_{t+1} = j)
                        marginal_prob = stationary_dist[i] * stationary_dist[j]
                        
                        if marginal_prob > epsilon:
                            mutual_info += joint_prob * np.log(joint_prob / marginal_prob) / np.log(base)
            
            return mutual_info
        
        def calculate_entropy_rate(matrix, stationary_dist, base=2):
            """Calculate entropy rate H(X_n | X_{n-1})"""
            # Entropy rate = sum over states of stationary_prob * conditional_entropy
            entropy_rate = 0.0
            epsilon = 1e-10
            
            for i in range(matrix.shape[0]):
                if stationary_dist[i] > epsilon:
                    # Calculate conditional entropy H(X_{n+1} | X_n = i)
                    conditional_entropy = 0.0
                    for j in range(matrix.shape[1]):
                        if matrix[i, j] > epsilon:
                            conditional_entropy -= matrix[i, j] * np.log(matrix[i, j]) / np.log(base)
                    
                    entropy_rate += stationary_dist[i] * conditional_entropy
            
            return entropy_rate
        
        def calculate_mixing_properties(matrix):
            """Calculate mixing time and spectral gap"""
            try:
                eigenvals = np.linalg.eigvals(matrix)
                # Sort eigenvalues by magnitude (descending)
                eigenvals_sorted = np.sort(np.abs(eigenvals))[::-1]
                
                # Spectral gap = 1 - second largest eigenvalue magnitude
                if len(eigenvals_sorted) > 1:
                    spectral_gap = 1.0 - eigenvals_sorted[1]
                else:
                    spectral_gap = 1.0
                
                # Mixing time estimate using spectral gap
                # T_mix ≈ log(n) / gap where n is number of states
                if spectral_gap > 1e-10:
                    mixing_time = np.log(matrix.shape[0]) / spectral_gap
                else:
                    mixing_time = np.inf
                
                return mixing_time, spectral_gap
            except:
                return np.inf, 0.0
        
        # Calculate all Markov chain properties
        results['markov_entropy'] = calculate_markov_entropy(matrix_array)
        
        # Calculate normalized entropy (0-1 scale)
        if results['is_square']:
            n_states = matrix_array.shape[0]
            max_entropy = np.log(n_states) / np.log(2)  # Maximum entropy for uniform distribution
            results['normalized_markov_entropy'] = results['markov_entropy'] / max_entropy if max_entropy > 0 else 0
        else:
            results['normalized_markov_entropy'] = None
        
        # Calculate stationary distribution
        stationary_dist = calculate_stationary_distribution(matrix_array)
        
        # Calculate mutual information
        results['mutual_information'] = calculate_mutual_information(matrix_array, stationary_dist)
        
        # Calculate entropy rate (conditional entropy)
        results['entropy_rate'] = calculate_entropy_rate(matrix_array, stationary_dist)
        
        # Calculate mixing properties
        if results['is_square']:
            mixing_time, spectral_gap = calculate_mixing_properties(matrix_array)
            results['mixing_time'] = mixing_time
            results['spectral_gap'] = spectral_gap
        else:
            results['mixing_time'] = None
            results['spectral_gap'] = None
    
    # Square matrix specific properties
    if results['is_square']:
        try:
            # Calculate determinant
            results['determinant'] = np.linalg.det(matrix_array)
            
            # Check invertibility (non-zero determinant)
            results['is_invertible'] = abs(results['determinant']) > 0
            results['is_singular'] = not results['is_invertible']
            
            # Calculate condition number
            if results['is_invertible']:
                results['condition_number'] = np.linalg.cond(matrix_array)
            else:
                results['condition_number'] = np.inf
            
            # Calculate eigenvalues
            eigenvals = np.linalg.eigvals(matrix_array)
            results['eigenvalues'] = eigenvals
            
            # Spectral radius (largest absolute eigenvalue)
            results['spectral_radius'] = np.max(np.abs(eigenvals))
            
            # Trace (sum of diagonal elements)
            results['trace'] = np.trace(matrix_array)
            
        except np.linalg.LinAlgError as e:
            if verbose:
                print(f"⚠️ Warning: Could not compute some properties for {matrix_name}: {e}")
    
    # Print detailed analysis if requested
    if verbose:
        print(f"\n{'='*60}")
        print(f"🧮 MATRIX ANALYSIS: {matrix_name.upper()}")
        print(f"{'='*60}")
        
        # Basic properties
        print(f"📊 Shape: {results['shape']}")
        print(f"📏 Rank: {results['rank']}")
        print(f"📐 Frobenius Norm: {results['frobenius_norm']:.6f}")
        
        # Square matrix properties
        if results['is_square']:
            print(f"⬜ Square Matrix: ✅")
            print(f"🔢 Determinant: {results['determinant']:.6f}")
            print(f"🔄 Invertible: {'✅ Yes' if results['is_invertible'] else '❌ No (Singular)'}")
            
            if results['condition_number'] is not None:
                if results['condition_number'] == np.inf:
                    print(f"📏 Condition Number: ∞ (Singular)")
                else:
                    print(f"📏 Condition Number: {results['condition_number']:.6f}")
                    if results['condition_number'] > 1e12:
                        print("   ⚠️ Very ill-conditioned (near singular)")
                    elif results['condition_number'] > 1e6:
                        print("   ⚠️ Ill-conditioned")
                    else:
                        print("   ✅ Well-conditioned")
            
            print(f"🎯 Trace: {results['trace']:.6f}")
            print(f"🌟 Spectral Radius: {results['spectral_radius']:.6f}")
            
            # Eigenvalue analysis
            if results['eigenvalues'] is not None:
                real_eigenvals = results['eigenvalues'].real
                imag_eigenvals = results['eigenvalues'].imag
                
                print(f"\n🔍 Eigenvalue Analysis:")
                print(f"   Number of eigenvalues: {len(results['eigenvalues'])}")
                print(f"   Real eigenvalues range: [{real_eigenvals.min():.6f}, {real_eigenvals.max():.6f}]")
                
                # Check for complex eigenvalues
                complex_count = np.sum(np.abs(imag_eigenvals) > 1e-10)
                if complex_count > 0:
                    print(f"   Complex eigenvalues: {complex_count} found")
                    print(f"   Imaginary parts range: [{imag_eigenvals.min():.6f}, {imag_eigenvals.max():.6f}]")
                else:
                    print(f"   All eigenvalues are real ✅")
                
                # Dominant eigenvalue
                dominant_idx = np.argmax(np.abs(results['eigenvalues']))
                dominant_eigenval = results['eigenvalues'][dominant_idx]
                print(f"   Dominant eigenvalue: {dominant_eigenval:.6f}")
                
                # Check for eigenvalue = 1 (important for stochastic matrices)
                unit_eigenvals = np.sum(np.abs(np.abs(results['eigenvalues']) - 1.0) < 1e-10)
                if unit_eigenvals > 0:
                    print(f"   Unit eigenvalues (|λ| = 1): {unit_eigenvals}")
        else:
            print(f"⬜ Square Matrix: ❌ (Cannot compute determinant/eigenvalues)")
        
        # Stochastic properties
        print(f"\n🎲 Stochastic Properties:")
        print(f"   Row-stochastic: {'✅ Yes' if results['is_stochastic'] else '❌ No'}")
        if results['is_stochastic']:
            print(f"   Doubly-stochastic: {'✅ Yes' if results['is_doubly_stochastic'] else '❌ No'}")
            print("   ✅ Valid transition/probability matrix")
            
            # Print Markov entropy information
            if results['markov_entropy'] is not None:
                print(f"\n🎯 Markov Chain Entropy:")
                print(f"   Entropy: {results['markov_entropy']:.6f} bits")
                if results['normalized_markov_entropy'] is not None:
                    print(f"   Normalized entropy: {results['normalized_markov_entropy']:.6f} (0-1 scale)")
                    
                    # # Interpretation of entropy level
                    # if results['normalized_markov_entropy'] > 0.8:
                    #     print(f"   🎲 High entropy: Very random/unpredictable transitions")
                    # elif results['normalized_markov_entropy'] > 0.5:
                    #     print(f"   ⚖️ Medium entropy: Moderately predictable transitions")
                    # elif results['normalized_markov_entropy'] > 0.2:
                    #     print(f"   🎯 Low entropy: Fairly predictable transitions")
                    # else:
                    #     print(f"   🔒 Very low entropy: Highly deterministic transitions")
                
                # Print additional Markov chain properties
                if results['entropy_rate'] is not None:
                    print(f"\n📊 Advanced Markov Properties:")
                    print(f"   Entropy rate H(X_n|X_{{n-1}}): {results['entropy_rate']:.6f} bits")
                    
                    if results['mutual_information'] is not None:
                        print(f"   Mutual information I(X_n;X_{{n+1}}): {results['mutual_information']:.6f} bits")
                        
                        # # Interpretation of memory/dependence
                        # if results['mutual_information'] > 1.0:
                        #     print(f"   🧠 High memory: Strong dependence between consecutive states")
                        # elif results['mutual_information'] > 0.5:
                        #     print(f"   🧠 Medium memory: Moderate dependence between states")
                        # elif results['mutual_information'] > 0.1:
                        #     print(f"   🧠 Low memory: Weak dependence between states")
                        # else:
                        #     print(f"   🧠 Very low memory: Nearly independent states")
                    
                    if results['spectral_gap'] is not None and results['mixing_time'] is not None:
                        print(f"\n⚡ Mixing Properties:")
                        print(f"   Spectral gap: {results['spectral_gap']:.6f}")
                        
                        if results['mixing_time'] == np.inf:
                            print(f"   Mixing time: ∞ (reducible/non-ergodic chain)")
                        else:
                            print(f"   Mixing time estimate: {results['mixing_time']:.2f} steps")
                            
                            # # Interpretation of mixing speed
                            # n_states = matrix_array.shape[0]
                            # if results['mixing_time'] < n_states:
                            #     print(f"   🚀 Fast mixing: Quickly reaches equilibrium")
                            # elif results['mixing_time'] < 5 * n_states:
                            #     print(f"   ⚖️ Moderate mixing: Reasonable convergence time")
                            # else:
                            #     print(f"   🐌 Slow mixing: Takes long to reach equilibrium")
                        
                        # # Spectral gap interpretation
                        # if results['spectral_gap'] > 0.5:
                        #     print(f"   ✅ Large spectral gap: Well-connected chain")
                        # elif results['spectral_gap'] > 0.1:
                        #     print(f"   ⚠️ Medium spectral gap: Moderately connected")
                        # else:
                        #     print(f"   ❌ Small spectral gap: Poorly connected or reducible")
        else:
            row_sums = np.sum(matrix_array, axis=1)
            print(f"   Row sums range: [{row_sums.min():.6f}, {row_sums.max():.6f}]")
            print("   ⚠️ Not a valid probability matrix")
        
        print(f"{'='*60}")
    
    return results


def calculate_markov_chain_entropy(df, col1, col2, true_col='true', normalize=True, base=2, verbose=True):
    """
    Calculate the Markov Chain Entropy between two transition matrices from DataFrame columns.
    
    Markov Chain Entropy measures the uncertainty or information content in transition patterns.
    Higher entropy indicates more uncertainty/randomness in transitions, while lower entropy
    indicates more predictable transition patterns.
    
    Args:
        df (pd.DataFrame): Combined dataframe with prediction columns and true labels
        col1 (str): Name of first prediction column
        col2 (str): Name of second prediction column  
        true_col (str): Name of the column containing true/reference labels. Default: 'true'
        normalize (bool): Whether to normalize entropy by maximum possible entropy. Default: True
        base (float): Logarithm base for entropy calculation (2=bits, e=nats, 10=dits). Default: 2
        verbose (bool): Whether to print detailed analysis. Default: True
        
    Returns:
        dict: Dictionary containing entropy analysis results:
            - entropy_col1: float - entropy of first transition matrix
            - entropy_col2: float - entropy of second transition matrix
            - entropy_difference: float - absolute difference between entropies
            - entropy_ratio: float - ratio of entropies (col1/col2)
            - cross_entropy: float - cross entropy between the two matrices
            - kl_divergence_col1_to_col2: float - KL divergence from col1 to col2
            - kl_divergence_col2_to_col1: float - KL divergence from col2 to col1
            - js_divergence: float - Jensen-Shannon divergence (symmetric)
            - max_entropy: float - maximum possible entropy for this number of classes
            - normalized_entropy_col1: float - normalized entropy of col1 (if normalize=True)
            - normalized_entropy_col2: float - normalized entropy of col2 (if normalize=True)
    
    Example usage:
        # Basic entropy comparison between 'most' and 'least' columns
        entropy_results = calculate_markov_chain_entropy(df_combined, 'most', 'least')
        
        # Use different true column and entropy base
        entropy_results = calculate_markov_chain_entropy(df_combined, 'pred1', 'pred2', 
                                                       true_col='ground_truth', base=np.e)
        
        # Get results without detailed printing
        entropy_results = calculate_markov_chain_entropy(df_combined, 'most', 'least', verbose=False)
        
        # Access specific metrics
        print(f"Entropy difference: {entropy_results['entropy_difference']:.4f}")
        print(f"JS Divergence: {entropy_results['js_divergence']:.4f}")
    """
    
    # Validate inputs
    missing_cols = [col for col in [col1, col2, true_col] if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing columns {missing_cols} in dataframe!")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🎲 MARKOV CHAIN ENTROPY ANALYSIS")
        print(f"{'='*80}")
        print(f"Comparing: '{col1}' vs '{col2}' (relative to '{true_col}')")
        print(f"Entropy base: {base} ({'bits' if base==2 else 'nats' if base==np.e else 'dits' if base==10 else 'custom'})")
    
    def create_transition_matrix(df, pred_col, true_col):
        """Create normalized transition matrix from DataFrame columns"""
        # Get canonical class set
        all_true_labels = df[true_col].dropna()
        if len(all_true_labels) == 0:
            return None
        
        unique_true_labels = sorted(all_true_labels.unique())
        min_label = int(min(unique_true_labels))
        max_label = int(max(unique_true_labels))
        canonical_classes = list(range(min_label, max_label + 1))
        
        # Create crosstab and normalize
        ct_raw = pd.crosstab(df[true_col], df[pred_col], dropna=False)
        ct = ct_raw.reindex(index=canonical_classes, columns=canonical_classes, fill_value=0)
        
        # Row-normalize to get P(predicted | true)
        transition_matrix = ct.div(ct.sum(axis=1), axis=0).fillna(0)
        return transition_matrix.values
    
    def calculate_entropy(matrix, base=2):
        """Calculate entropy of a transition matrix"""
        matrix = np.array(matrix)
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        matrix_safe = matrix + epsilon
        
        # Calculate entropy for each row (true class)
        row_entropies = []
        for row in matrix_safe:
            # Normalize row to ensure it sums to 1
            row_normalized = row / row.sum()
            # Calculate entropy: -sum(p * log(p))
            entropy = -np.sum(row_normalized * np.log(row_normalized) / np.log(base))
            row_entropies.append(entropy)
        
        # Return average entropy across all true classes
        return np.mean(row_entropies)
    
    def calculate_cross_entropy(matrix1, matrix2, base=2):
        """Calculate cross entropy between two matrices"""
        matrix1 = np.array(matrix1)
        matrix2 = np.array(matrix2)
        epsilon = 1e-10
        
        # Ensure matrices are properly normalized
        matrix1_norm = matrix1 / (matrix1.sum(axis=1, keepdims=True) + epsilon)
        matrix2_norm = matrix2 / (matrix2.sum(axis=1, keepdims=True) + epsilon) + epsilon
        
        # Calculate cross entropy: -sum(p * log(q))
        cross_entropies = []
        for i in range(len(matrix1_norm)):
            cross_entropy = -np.sum(matrix1_norm[i] * np.log(matrix2_norm[i]) / np.log(base))
            cross_entropies.append(cross_entropy)
        
        return np.mean(cross_entropies)
    
    def calculate_kl_divergence(matrix1, matrix2, base=2):
        """Calculate KL divergence from matrix1 to matrix2"""
        cross_ent = calculate_cross_entropy(matrix1, matrix2, base)
        entropy1 = calculate_entropy(matrix1, base)
        return cross_ent - entropy1
    
    def calculate_js_divergence(matrix1, matrix2, base=2):
        """Calculate Jensen-Shannon divergence (symmetric)"""
        # Average of the two matrices
        matrix_avg = 0.5 * (matrix1 + matrix2)
        
        # JS divergence = 0.5 * (KL(P||M) + KL(Q||M))
        kl1 = calculate_kl_divergence(matrix1, matrix_avg, base)
        kl2 = calculate_kl_divergence(matrix2, matrix_avg, base)
        
        return 0.5 * (kl1 + kl2)
    
    # Create transition matrices
    if verbose:
        print(f"\n🔄 Creating transition matrices...")
    
    matrix1 = create_transition_matrix(df, col1, true_col)
    matrix2 = create_transition_matrix(df, col2, true_col)
    
    if matrix1 is None or matrix2 is None:
        print(f"❌ Error: Could not create transition matrices")
        return None
    
    if matrix1.shape != matrix2.shape:
        print(f"❌ Error: Matrix shapes don't match: {matrix1.shape} vs {matrix2.shape}")
        return None
    
    n_classes = matrix1.shape[0]
    if verbose:
        print(f"   Matrix shape: {matrix1.shape}")
        print(f"   Number of classes: {n_classes}")
    
    # Calculate entropies and divergences
    entropy1 = calculate_entropy(matrix1, base)
    entropy2 = calculate_entropy(matrix2, base)
    cross_entropy = calculate_cross_entropy(matrix1, matrix2, base)
    kl_div_1_to_2 = calculate_kl_divergence(matrix1, matrix2, base)
    kl_div_2_to_1 = calculate_kl_divergence(matrix2, matrix1, base)
    js_div = calculate_js_divergence(matrix1, matrix2, base)
    
    # Calculate maximum possible entropy (uniform distribution)
    max_entropy = np.log(n_classes) / np.log(base)
    
    # Normalize entropies if requested
    normalized_entropy1 = entropy1 / max_entropy if normalize else None
    normalized_entropy2 = entropy2 / max_entropy if normalize else None
    
    # Calculate differences and ratios
    entropy_diff = abs(entropy1 - entropy2)
    entropy_ratio = entropy1 / entropy2 if entropy2 > 0 else np.inf
    
    # Store results
    results = {
        'col1': col1,
        'col2': col2,
        'true_col': true_col,
        'n_classes': n_classes,
        'base': base,
        'entropy_col1': entropy1,
        'entropy_col2': entropy2,
        'entropy_difference': entropy_diff,
        'entropy_ratio': entropy_ratio,
        'cross_entropy': cross_entropy,
        'kl_divergence_col1_to_col2': kl_div_1_to_2,
        'kl_divergence_col2_to_col1': kl_div_2_to_1,
        'js_divergence': js_div,
        'max_entropy': max_entropy,
        'normalized_entropy_col1': normalized_entropy1,
        'normalized_entropy_col2': normalized_entropy2,
        'matrix1': matrix1,
        'matrix2': matrix2
    }
    
    # Print detailed analysis
    if verbose:
        print(f"\n📊 ENTROPY RESULTS:")
        print(f"   {col1} entropy: {entropy1:.6f}")
        print(f"   {col2} entropy: {entropy2:.6f}")
        print(f"   Entropy difference: {entropy_diff:.6f}")
        print(f"   Entropy ratio ({col1}/{col2}): {entropy_ratio:.6f}")
        print(f"   Maximum possible entropy: {max_entropy:.6f}")
        
        if normalize:
            print(f"\n📏 NORMALIZED ENTROPIES (0-1 scale):")
            print(f"   {col1} normalized: {normalized_entropy1:.6f}")
            print(f"   {col2} normalized: {normalized_entropy2:.6f}")
        
        print(f"\n🔀 DIVERGENCE MEASURES:")
        print(f"   Cross entropy ({col1} → {col2}): {cross_entropy:.6f}")
        print(f"   KL divergence ({col1} → {col2}): {kl_div_1_to_2:.6f}")
        print(f"   KL divergence ({col2} → {col1}): {kl_div_2_to_1:.6f}")
        print(f"   Jensen-Shannon divergence: {js_div:.6f}")
        
        # Interpretation
        print(f"\n🎯 INTERPRETATION:")
        if entropy_diff < 0.1:
            print(f"   ✅ Very similar entropy levels (diff: {entropy_diff:.6f})")
        elif entropy_diff < 0.5:
            print(f"   ⚠️ Moderately different entropy levels (diff: {entropy_diff:.6f})")
        else:
            print(f"   ❗ Significantly different entropy levels (diff: {entropy_diff:.6f})")
        
        higher_entropy_col = col1 if entropy1 > entropy2 else col2
        print(f"   📈 '{higher_entropy_col}' has higher entropy (more uncertain/random)")
        
        if js_div < 0.1:
            print(f"   ✅ Matrices are quite similar (JS divergence: {js_div:.6f})")
        elif js_div < 0.5:
            print(f"   ⚠️ Matrices are moderately different (JS divergence: {js_div:.6f})")
        else:
            print(f"   ❗ Matrices are significantly different (JS divergence: {js_div:.6f})")
        
        if normalize:
            if max(normalized_entropy1, normalized_entropy2) > 0.8:
                print(f"   🎲 High randomness: Close to uniform distribution")
            elif max(normalized_entropy1, normalized_entropy2) < 0.3:
                print(f"   🎯 Low randomness: Highly predictable transitions")
            else:
                print(f"   ⚖️ Moderate randomness: Balanced transition patterns")
        
        print(f"{'='*80}")
    
    return results


def analyze_wandb_run(run_id):
    """
    Analyze a wandb run by extracting information from metadata, summary, and output log.
    
    Args:
        run_id (str): The wandb run ID (e.g., 'run-20250827_221945-q2rhukv1')
    
    Returns:
        dict: Run information including dataset, algo, cll_type, and best_acc
    """
    base_path = f"wandb/{run_id}/files"
    
    # Check if the run directory exists
    if not os.path.exists(base_path):
        print(f"❌ Error: Run directory {base_path} not found!")
        return None
    
    try:
        # 1. Load wandb-metadata.json for run info
        metadata_path = f"{base_path}/wandb-metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Extract run information from args
            args = metadata.get('args', [])
            
            # Parse arguments to extract key information
            run_info = {}
            i = 0
            while i < len(args):
                arg = args[i]
                if arg.startswith('--'):
                    key = arg[2:]  # Remove '--' prefix
                    if i + 1 < len(args) and not args[i + 1].startswith('--'):
                        run_info[key] = args[i + 1]
                        i += 2
                    else:
                        run_info[key] = True
                        i += 1
                else:
                    i += 1
            
            dataset = run_info.get('dataset_name', 'Unknown')
            algo = run_info.get('algo', 'Unknown')
            cll_type = run_info.get('cll_type', 'Unknown')
            
        else:
            print(f"⚠️ Warning: {metadata_path} not found!")
            dataset = algo = cll_type = 'Unknown'
            run_info = {}
        
        # 2. Load wandb-summary.json for best accuracy
        summary_path = f"{base_path}/wandb-summary.json"
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            best_acc = summary.get('best_acc', 'Not available')
        else:
            print(f"⚠️ Warning: {summary_path} not found!")
            best_acc = 'Not available'
        
        # 3. Load and parse output.log for learning curves
        output_log_path = f"{base_path}/output.log"
        if os.path.exists(output_log_path):
            with open(output_log_path, 'r') as f:
                log_data = f.read()
            
            # Parse training log and create learning curve plot
            try:
                metrics = parse_training_log(log_data)
                
                # Create filename based on run info
                output_filename = f"{dataset}-{algo}-{cll_type}.png"
                
                plot_learning_curves(metrics, output_filename=output_filename)
                print(f"📊 Learning curve saved as: {output_filename}")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not generate learning curve plot: {e}")
        else:
            print(f"⚠️ Warning: {output_log_path} not found!")
        
        # Print run information
        print(f"\n{'='*60}")
        print(f"📋 WANDB RUN ANALYSIS: {run_id}")
        print(f"{'='*60}")
        print(f"🎯 Dataset: {dataset}")
        print(f"🔧 Algorithm: {algo}")
        print(f"🏷️  CLL Type: {cll_type}")
        print(f"🏆 Best Accuracy: {best_acc}")
        
        # Print additional run info if available
        if run_info:
            print(f"\n📝 Additional Run Parameters:")
            for key, value in run_info.items():
                if key not in ['dataset_name', 'algo', 'cll_type']:
                    print(f"   {key}: {value}")
        
        print(f"{'='*60}")
        
        return {
            'run_id': run_id,
            'dataset': dataset,
            'algo': algo,
            'cll_type': cll_type,
            'best_acc': best_acc,
            'run_info': run_info
        }
        
    except Exception as e:
        print(f"❌ Error analyzing run {run_id}: {e}")
        return None



def parse_training_log(log_content: str) -> Dict[str, List[Any]]:
    """
    Parses a machine learning training log to extract metrics for each epoch.

    Args:
        log_content: A string containing the entire log output.

    Returns:
        A dictionary containing lists of extracted metrics like loss and precision
        for both training and testing phases, indexed by epoch number.
    """
    # Regex to find the summary lines for training and testing results per epoch
    train_results_regex = re.compile(r"Training Results: Prec@1 ([\d.]+) Prec@5 [\d.]+ \s+ Loss ([\d.]+)")
    test_results_regex = re.compile(r"Testing Results: Prec@1 ([\d.]+) Prec@5 [\d.]+ \s+ Loss ([\d.]+)")

    # Find all matches for training and testing metrics in the log
    train_matches = train_results_regex.finditer(log_content)
    test_matches = test_results_regex.finditer(log_content)

    metrics = {
        'epochs': [],
        'train_loss': [],
        'train_prec1': [],
        'test_loss': [],
        'test_prec1': [],
    }

    # Extract training metrics
    for i, match in enumerate(train_matches):
        metrics['epochs'].append(i)
        metrics['train_prec1'].append(float(match.group(1)))
        metrics['train_loss'].append(float(match.group(2)))

    # Extract testing metrics
    for match in test_matches:
        metrics['test_prec1'].append(float(match.group(1)))
        metrics['test_loss'].append(float(match.group(2)))

    return metrics

def plot_learning_curves(metrics: Dict[str, List[Any]], output_filename: str = "learning_curves.png") -> None:
    """
    Generates and saves learning curve graphs for loss and accuracy to a file.

    Args:
        metrics: A dictionary containing the parsed training and testing metrics.
        output_filename: The name of the file to save the plot as. The plot will
                         be saved in an 'output' directory.
    """
    output_dir = "output"
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the full path for the output file
    output_path = os.path.join(output_dir, output_filename)
    
    epochs = metrics['epochs']

    # Create a figure with two subplots (one for loss, one for accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Training and Testing Loss
    ax1.plot(epochs, metrics['train_loss'], 'o-', label='Training Loss')
    ax1.plot(epochs, metrics['test_loss'], 'o-', label='Testing Loss')
    ax1.set_title('Loss Over Epochs', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # Plot Training and Testing Accuracy (Precision@1)
    # ax2.plot(epochs, metrics['train_prec1'], 'o-', label='Training Prec@1')
    ax2.plot(epochs, metrics['test_prec1'], 'o-', label='Testing Prec@1')
    ax2.set_title('Accuracy (Prec@1) Over Epochs', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Precision@1 (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    plt.suptitle('Model Learning Curves', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure to the specified path instead of showing it
    plt.savefig(output_path)
    
    # Close the figure to free up memory
    plt.close(fig)
    
    print(f"Plot successfully saved to: {output_path}")


def plot_transition_matrices(df, true_col='true', figsize=(8, 6), dpi=300, include_missing_values=True, savepath=None, analyze_matrix=True):
    """
    Plot transition matrices for all columns vs a reference 'true' column in a DataFrame.
    
    Args:
        df (pd.DataFrame): Combined dataframe with prediction columns and true labels
        true_col (str): Name of the column containing true/reference labels. Default: 'true'
        figsize (tuple): Figure size for each heatmap. Default: (8, 6)
        dpi (int): DPI for high-resolution plots. Default: 300
        include_missing_values (bool): Whether to include missing categorical values with zero counts
        savepath (str): Path to save the transition matrix as a txt file. Default: None (no saving)
                       If multiple prediction columns exist, column names will be appended to filename.
                       Example: 'matrix.txt' becomes 'matrix_most.txt', 'matrix_least.txt', etc.
        analyze_matrix (bool): Whether to perform mathematical analysis of matrix properties. Default: True
        
    Returns:
        dict: Dictionary containing transition counts, proportions, and matrix analysis for each column
        
    Example usage:
        # Basic usage with standard 'true' column
        results = plot_transition_matrices(df_combined)
        
        # Custom true column name and figure settings
        results = plot_transition_matrices(df_combined, true_col='ground_truth', figsize=(10, 8))
        
        # For CIFAR-20/100 with many classes, use larger figures
        results = plot_transition_matrices(df_combined, figsize=(12, 10))
        
        # Save transition matrix to file
        results = plot_transition_matrices(df_combined, savepath='transition_matrix_output.txt')
        
        # Save with multiple columns (creates multiple files with column names appended)
        results = plot_transition_matrices(df_combined, savepath='matrices/transition_matrix.txt')
        # Creates: matrices/transition_matrix_most.txt, matrices/transition_matrix_least.txt, etc.
        
        # Skip matrix analysis for faster processing
        results = plot_transition_matrices(df_combined, analyze_matrix=False)
    """
    
    if true_col not in df.columns:
        print(f"❌ Error: '{true_col}' column not found in dataframe!")
        print(f"Available columns: {list(df.columns)}")
        return {}
    
    # Get all prediction columns (everything except the true column)
    pred_cols = [c for c in df.columns if c != true_col]
    
    if not pred_cols:
        print(f"❌ Error: No prediction columns found! Only '{true_col}' column exists.")
        return {}
    
    print(f"🔍 Analyzing transition matrices for {len(pred_cols)} prediction columns vs '{true_col}'")
    print(f"📊 Prediction columns: {pred_cols}")
    
    transition_counts = {}
    transition_props = {}
    matrix_analyses = {}
    
    for col in pred_cols:
        print(f"\n{'='*60}")
        print(f"TRANSITION MATRIX: {col.upper()} vs {true_col.upper()}")
        print(f"{'='*60}")
        
        # Create crosstab: rows = true labels, cols = predicted labels
        if include_missing_values:
            # Get the canonical set of class labels from true labels column
            all_true_labels = df[true_col].dropna()
            
            if len(all_true_labels) > 0:
                # Get all unique true labels and create complete range (0 to max)
                unique_true_labels = sorted(all_true_labels.unique())
                min_true_label = int(min(unique_true_labels))
                max_true_label = int(max(unique_true_labels))
                
                # Create complete canonical class set (assuming 0-indexed classes)
                canonical_classes = list(range(min_true_label, max_true_label + 1))
                
                print(f"   📋 Canonical class set from true labels: {canonical_classes}")
                
                # Create crosstab with dropna=False to include NaN categories
                ct_no_margins_raw = pd.crosstab(df[true_col], df[col], dropna=False)
                
                # Force square matrix: both rows AND columns use the canonical class set
                # This ensures that even if a true class never gets predicted, it appears as a column
                ct_no_margins = ct_no_margins_raw.reindex(
                    index=canonical_classes, 
                    columns=canonical_classes,  # Same set for columns to ensure square matrix
                    fill_value=0
                )
                
                # Recreate the marginal crosstab with complete labels
                ct = ct_no_margins.copy()
                ct['All'] = ct.sum(axis=1)  # Row totals
                ct.loc['All'] = ct.sum(axis=0)  # Column totals (including 'All' total)
                
                print(f"   📊 Transition matrix shape: {ct_no_margins.shape} (square: {ct_no_margins.shape[0] == ct_no_margins.shape[1]})")
            else:
                # Fallback if no valid data
                ct = pd.crosstab(df[true_col], df[col], margins=True, dropna=False)
                ct_no_margins = pd.crosstab(df[true_col], df[col], dropna=False)
        else:
            ct = pd.crosstab(df[true_col], df[col], margins=True)
            ct_no_margins = pd.crosstab(df[true_col], df[col])
        
        transition_counts[col] = ct
        
        print(f"\nCounts ({true_col} vs {col}):")
        print(ct)
        
        # Check if we have data to work with
        if ct_no_margins.empty or ct_no_margins.shape[0] == 0 or ct_no_margins.shape[1] == 0:
            print(f"\n⚠️ No data available for transition matrix between '{true_col}' and '{col}'")
            transition_props[col] = pd.DataFrame()
            matrix_analyses[col] = None
            continue
        
        # Calculate row-normalized proportions P(predicted | true) - rows sum to 1
        transition_props[col] = ct_no_margins.div(ct_no_margins.sum(axis=1), axis=0).fillna(0)
        
        print(f"\nRow-normalized proportions P({col}|{true_col}):")
        print(transition_props[col].round(3))
        
        # Perform matrix analysis if requested
        if analyze_matrix and not transition_props[col].empty:
            matrix_name = f"{col} vs {true_col} Transition Matrix"
            matrix_analyses[col] = analyze_matrix_properties(transition_props[col], matrix_name)
        else:
            matrix_analyses[col] = None
        
        # Save transition matrix to file if savepath is provided
        if savepath is not None and not transition_props[col].empty:
            try:
                actual_savepath = f"{savepath}/{col}.txt" if len(pred_cols) >= 1 else savepath
                
                # Save as space-delimited text file (like the existing transition matrix files)
                np.savetxt(actual_savepath, transition_props[col].values, fmt='%.6f')
                print(f"💾 Transition matrix saved to: {actual_savepath}")
                print(f"   Matrix shape: {transition_props[col].shape}")
                print(f"   File format: Space-delimited text file with 6 decimal places")
            except Exception as e:
                print(f"❌ Error saving transition matrix to {actual_savepath}: {e}")
        
        # Create heatmap if we have valid data
        if not transition_props[col].empty and transition_props[col].size > 0:
            plt.figure(figsize=figsize, dpi=dpi)
            
            # Adjust annotation format based on matrix size
            n_classes = max(transition_props[col].shape)
            if n_classes > 20:
                # For large matrices (like CIFAR-100), use smaller annotations
                annot_fmt = '.1f'
                annot_fontsize = 6
            elif n_classes > 10:
                # For medium matrices (like CIFAR-20), use medium annotations  
                annot_fmt = '.2f'
                annot_fontsize = 8
            else:
                # For small matrices (like CIFAR-10), use detailed annotations
                annot_fmt = '.3f'
                annot_fontsize = 10
            
            sns.heatmap(transition_props[col], 
                       annot=True, 
                       fmt=annot_fmt, 
                       cmap='Blues',
                       cbar_kws={'label': 'Probability'},
                       square=True,
                       annot_kws={'fontsize': annot_fontsize})
            
            plt.title(f'Transition Matrix: P({col}|{true_col})', fontsize=14, pad=20)
            plt.ylabel(f'{true_col.capitalize()} Label', fontsize=12)
            plt.xlabel(f'{col.capitalize()} Prediction', fontsize=12)
            plt.tight_layout()
            plt.show()
            
            # Print accuracy (diagonal sum / total)
            if transition_props[col].shape[0] == transition_props[col].shape[1]:
                diagonal_sum = np.trace(transition_props[col].values)
                n_classes_actual = transition_props[col].shape[0]
                accuracy = diagonal_sum / n_classes_actual if n_classes_actual > 0 else 0
                print(f"📈 Average per-class accuracy for {col}: {accuracy:.3f}")
            
        else:
            print(f"⚠️ Skipping heatmap for '{col}' - no valid data to plot")
    
    print(f"\n{'='*60}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*60}")
    
    return {
        'transition_counts': transition_counts,
        'transition_props': transition_props,
        'matrix_analyses': matrix_analyses,
        'prediction_columns': pred_cols,
        'true_column': true_col
    }


def plot_transition_matrix_from_file(file_path, matrix_name="Transition Matrix", figsize=(10, 8), dpi=300, analyze_matrix=True):
    """
    Load a transition matrix from a text file and plot it as a heatmap.
    
    Args:
        file_path (str): Path to the text file containing the transition matrix
        matrix_name (str): Name/title for the matrix plot. Default: "Transition Matrix"
        figsize (tuple): Figure size for the heatmap. Default: (8, 6)
        dpi (int): DPI for high-resolution plots. Default: 300
        analyze_matrix (bool): Whether to perform mathematical analysis of matrix properties. Default: True
        
    Returns:
        dict: Dictionary containing the loaded matrix and analysis results (if requested)
        
    Example usage:
        # Load and plot a saved transition matrix
        result = plot_transition_matrix_from_file('transition_matrix_least_fwd-int_cifar20.txt', 
                                                 'CIFAR-20 FWD-INT Least')
        
        # Load with custom figure size for larger matrices
        result = plot_transition_matrix_from_file('transition_matrix_cifar100.txt', 
                                                 'CIFAR-100 Transition Matrix', 
                                                 figsize=(12, 10))
        
        # Load without matrix analysis for faster processing
        result = plot_transition_matrix_from_file('matrix.txt', analyze_matrix=False)
    """
    
    try:
        # Load the transition matrix from the text file
        transition_matrix = np.loadtxt(file_path)
        
        print(f"📁 Loaded transition matrix from: {file_path}")
        print(f"📊 Matrix shape: {transition_matrix.shape}")
        print(f"📈 Matrix range: [{transition_matrix.min():.6f}, {transition_matrix.max():.6f}]")
        
        # Convert to DataFrame for better visualization with labels
        n_classes = transition_matrix.shape[0]
        class_labels = list(range(n_classes))
        
        df_matrix = pd.DataFrame(transition_matrix, 
                               index=class_labels, 
                               columns=class_labels)
        
        # Perform matrix analysis if requested (same as plot_transition_matrices)
        matrix_analysis = None
        if analyze_matrix:
            matrix_analysis = analyze_matrix_properties(transition_matrix, matrix_name, verbose=True)
        
        # Create heatmap if we have valid data
        if not df_matrix.empty and df_matrix.size > 0:
            plt.figure(figsize=figsize, dpi=dpi)
            
            # Adjust annotation format based on matrix size (same logic as plot_transition_matrices)
            if n_classes > 20:
                # For large matrices (like CIFAR-100), use smaller annotations
                annot_fmt = '.1f'
                annot_fontsize = 6
            elif n_classes > 10:
                # For medium matrices (like CIFAR-20), use medium annotations  
                annot_fmt = '.2f'
                annot_fontsize = 8
            else:
                # For small matrices (like CIFAR-10), use detailed annotations
                annot_fmt = '.3f'
                annot_fontsize = 10
            
            sns.heatmap(df_matrix, 
                       annot=True, 
                       fmt=annot_fmt, 
                       cmap='Blues',
                       cbar_kws={'label': 'Probability'},
                       square=True,
                       annot_kws={'fontsize': annot_fontsize})
            
            plt.title(f'{matrix_name}', fontsize=14, pad=20)
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️ Skipping heatmap - no valid data to plot")
        
        # Calculate and print accuracy (diagonal sum / number of classes)
        if transition_matrix.shape[0] == transition_matrix.shape[1]:
            diagonal_sum = np.trace(transition_matrix)
            accuracy = diagonal_sum / n_classes
            print(f"📈 Average per-class accuracy: {accuracy:.3f}")
            
            # Show diagonal values (class-wise accuracies)
            diagonal_values = np.diag(transition_matrix)
            print(f"📋 Per-class accuracies:")
            for i, acc in enumerate(diagonal_values):
                if acc > 0:  # Only show classes with non-zero accuracy
                    print(f"   Class {i}: {acc:.3f}")
        
        print(f"\n{'='*60}")
        print(f"✅ MATRIX VISUALIZATION COMPLETE")
        print(f"{'='*60}")
        
        return {
            'matrix': transition_matrix,
            'dataframe': df_matrix,
            'analysis': matrix_analysis,
            'file_path': file_path,
            'matrix_name': matrix_name
        }
        
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading transition matrix from '{file_path}': {e}")
        return None

def verify_cifar_data_indexing(dataset_name, num_samples=3, data_path_override=None):
    """
    Verify that indexing between raw CIFAR data and generated label files is consistent.
    
    Args:
        dataset_name (str): One of 'cifar10', 'cifar20', or 'cifar100'
        num_samples (int): Number of random samples to test (default: 3)
        data_path_override (str): Custom path to data directory (optional)
    
    Returns:
        bool: True if all verifications pass, False otherwise
    """
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    
    # Dataset configurations
    configs = {
        'cifar10': {
            'num_classes': 10,
            'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck'],
            'data_subdir': 'cifar10/cifar-10-batches-py',
            'batch_files': ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'],
            'meta_file': 'batches.meta'
        },
        'cifar20': {
            'num_classes': 20,
            'class_names': [f'class_{i}' for i in range(20)],  # Generic names for CIFAR-20
            'data_subdir': 'cifar20/cifar-100-python',  # Based on actual folder structure
            'batch_files': ['train'],  # CIFAR-20 uses single train file
            'meta_file': 'meta'
        },
        'cifar100': {
            'num_classes': 100,
            'class_names': [f'class_{i}' for i in range(100)],  # Generic names for CIFAR-100
            'data_subdir': 'cifar100/cifar-100-python',  # Based on actual folder structure
            'batch_files': ['train'],  # CIFAR-100 uses single train file
            'meta_file': 'meta'
        }
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Must be one of {list(configs.keys())}")
    
    config = configs[dataset_name]
    
    def load_cifar_data():
        """Load CIFAR data from the preprocessed files"""
        if data_path_override:
            data_path = f'{data_path_override}/{config["data_subdir"]}'
        else:
            data_path = f'data/{config["data_subdir"]}'
        
        # Load training batches
        images = []
        labels = []
        
        for batch_file in config['batch_files']:
            try:
                with open(f'{data_path}/{batch_file}', 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    
                    # Handle different data structures
                    if dataset_name == 'cifar10':
                        # CIFAR-10 structure: b'data' and b'labels'
                        images.append(batch[b'data'])
                        labels.extend(batch[b'labels'])
                    else:
                        # CIFAR-20/100 structure: b'data' and b'fine_labels' (or b'coarse_labels')
                        images.append(batch[b'data'])
                        # Use fine_labels for CIFAR-100, try both for CIFAR-20
                        if b'fine_labels' in batch:
                            labels.extend(batch[b'fine_labels'])
                        elif b'coarse_labels' in batch:
                            labels.extend(batch[b'coarse_labels'])
                        elif b'labels' in batch:
                            labels.extend(batch[b'labels'])
                        else:
                            print(f"⚠️ Warning: No recognized label field in {batch_file}")
                            print(f"Available keys: {list(batch.keys())}")
                            continue
                            
            except FileNotFoundError:
                print(f"⚠️ Warning: {batch_file} not found, skipping...")
                continue
        
        if not images:
            raise FileNotFoundError(f"No batch files found for {dataset_name}")
        
        # Concatenate all batches
        images = np.concatenate(images, axis=0)
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(labels)
        
        # Load meta information (optional)
        try:
            with open(f'{data_path}/{config["meta_file"]}', 'rb') as f:
                meta = pickle.load(f, encoding='bytes')
        except FileNotFoundError:
            print(f"⚠️ Warning: {config['meta_file']} not found, continuing without metadata...")
            meta = None
        
        return images, labels, meta
    
    # Load the dataset
    print(f"Loading {dataset_name.upper()} dataset...")
    try:
        images, true_labels_from_data, meta = load_cifar_data()
    except Exception as e:
        print(f"❌ Failed to load {dataset_name} data: {e}")
        return False
    
    # Load the generated label files
    file_path = f'generated_labels/{dataset_name}/'
    
    try:
        true_data = load_data_file(f'{file_path}true.txt')
        most_data = load_data_file(f'{file_path}most.txt')
        least_data = load_data_file(f'{file_path}least.txt')
    except Exception as e:
        print(f"❌ Failed to load generated label files: {e}")
        return False
    
    print(f"Dataset loaded:")
    print(f"  Images shape: {images.shape}")
    print(f"  True labels from data: {len(true_labels_from_data)} samples")
    print(f"  True labels from file: {len(true_data)} samples")
    print(f"  Most labels from file: {len(most_data)} samples")
    print(f"  Least labels from file: {len(least_data)} samples")
    
    # Verify lengths match
    try:
        assert len(true_labels_from_data) == len(true_data), "True labels length mismatch!"
        assert len(true_data) == len(most_data), "Most labels length mismatch!"
        assert len(true_data) == len(least_data), "Least labels length mismatch!"
        print("✅ All label files have matching lengths!")
    except AssertionError as e:
        print(f"❌ Length verification failed: {e}")
        return False
    
    # Test with random samples
    print(f"\n{'='*80}")
    print(f"TESTING {num_samples} RANDOM SAMPLES TO VERIFY INDEXING - {dataset_name.upper()}")
    print(f"{'='*80}")
    
    verification_passed = True
    
    for test_num in range(num_samples):
        # Pick a random index
        random_idx = random.randint(0, len(images) - 1)
        
        # Get the image and labels
        image = images[random_idx]
        true_from_data = true_labels_from_data[random_idx]
        true_from_file = true_data[random_idx]
        most_label = most_data[random_idx]
        least_label = least_data[random_idx]
        
        print(f"\n📋 Sample {test_num + 1} (Index: {random_idx})")
        print(f"   True label from dataset: {true_from_data} ({config['class_names'][true_from_data]})")
        print(f"   True label from file:    {true_from_file} ({config['class_names'][true_from_file]})")
        print(f"   Most complementary:      {most_label} ({config['class_names'][most_label]})")
        print(f"   Least complementary:     {least_label} ({config['class_names'][least_label]})")
        
        # Verify true labels match
        if true_from_data == true_from_file:
            print("   ✅ True labels match!")
        else:
            print("   ❌ True labels DON'T match!")
            verification_passed = False
        
        # Display the image
        plt.figure(figsize=(8, 3))
        
        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.title(f'Sample {test_num + 1}\nIndex: {random_idx}')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.text(0.1, 0.7, f'TRUE:\n{true_from_file}\n{config["class_names"][true_from_file]}', 
                 fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')
        plt.axis('off')
        plt.title('True Label')
        
        plt.subplot(1, 4, 3)
        plt.text(0.1, 0.7, f'MOST:\n{most_label}\n{config["class_names"][most_label]}', 
                 fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')
        plt.axis('off')
        plt.title('Most Complementary')
        
        plt.subplot(1, 4, 4)
        plt.text(0.1, 0.7, f'LEAST:\n{least_label}\n{config["class_names"][least_label]}', 
                 fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')
        plt.axis('off')
        plt.title('Least Complementary')
        
        plt.tight_layout()
        plt.show()
    
    print(f"\n{'='*80}")
    print(f"INDEX VERIFICATION COMPLETE - {dataset_name.upper()}")
    print(f"{'='*80}")
    
    if verification_passed:
        print(f"✅ All verifications passed for {dataset_name.upper()}!")
    else:
        print(f"❌ Some verifications failed for {dataset_name.upper()}!")
    
    return verification_passed

def normalize_and_save_transition_matrix(filepath, normalize_type='row'):
    """
    Load a transition matrix, normalize it, and save back to the same file.
    
    Args:
        filepath: Path to the transition matrix file
        normalize_type: 'row' (default) - rows sum to 1, P(predicted|true)
                       'col' - columns sum to 1, P(true|predicted)
    """
    import numpy as np
    
    # Load the transition matrix
    print(f"Loading transition matrix from: {filepath}")
    matrix = np.loadtxt(filepath)
    
    print(f"Original matrix shape: {matrix.shape}")
    print(f"Original matrix:\n{matrix}")
    
    # Normalize the matrix
    if normalize_type == 'row':
        # Row normalization: each row sums to 1
        row_sums = matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        normalized_matrix = matrix / row_sums
        print("\n✅ Row-normalized (rows sum to 1) - P(predicted|true)")
        
    elif normalize_type == 'col':
        # Column normalization: each column sums to 1
        col_sums = matrix.sum(axis=0, keepdims=True)
        # Avoid division by zero
        col_sums[col_sums == 0] = 1
        normalized_matrix = matrix / col_sums
        print("\n✅ Column-normalized (columns sum to 1) - P(true|predicted)")
        
    else:
        raise ValueError(f"Unknown normalize_type: {normalize_type}. Use 'row' or 'col'.")
    
    print(f"Normalized matrix:\n{normalized_matrix}")
    
    # Verify normalization
    if normalize_type == 'row':
        row_sums_after = normalized_matrix.sum(axis=1)
        print(f"\nRow sums after normalization: {row_sums_after}")
        print(f"All rows sum to ~1.0: {np.allclose(row_sums_after, 1.0)}")
    else:
        col_sums_after = normalized_matrix.sum(axis=0)
        print(f"\nColumn sums after normalization: {col_sums_after}")
        print(f"All columns sum to ~1.0: {np.allclose(col_sums_after, 1.0)}")
    
    # Save back to the same file
    np.savetxt(filepath, normalized_matrix, fmt='%.6f')
    print(f"\n💾 Normalized matrix saved to: {filepath}")
    
    return normalized_matrix

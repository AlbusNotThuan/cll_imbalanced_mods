import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any
import json

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

def plot_dataframe_distributions(df_dict=None, candidate_names=None, max_unique_for_bars=50, figsize_per_col=5):
    """
    Plot histograms/count plots for multiple DataFrames and their columns.
    
    Args:
        df_dict (dict, optional): Dictionary of {name: dataframe} to plot. 
                                 If None, will search globals() for candidate_names.
        candidate_names (list, optional): List of DataFrame variable names to search for in globals().
                                       Default: ['df_random', 'df_most', 'df_least', 'df_most_no_noise', 'df_combined']
        max_unique_for_bars (int): Maximum number of unique values to use bar plots instead of histograms.
        figsize_per_col (int): Width in inches per subplot column.
    
    Example usage:
        # Use with explicit dataframes
        plot_dataframe_distributions({'my_data': df1, 'other_data': df2})
        
        # Use with automatic detection from globals
        plot_dataframe_distributions()
        
        # Use with custom candidate names
        plot_dataframe_distributions(candidate_names=['df_cifar10', 'df_cifar20', 'df_cifar100'])
    """
    
    # Default candidate names if not provided
    if candidate_names is None:
        candidate_names = [
            'df_random', 'df_most', 'df_least', 'df_most_no_noise', 'df_combined',
            'df_random_100', 'df_most_100', 'df_least_100', 'df_combined_100'
        ]
    
    # Get dataframes either from provided dict or by searching globals
    if df_dict is None:
        available_dfs = {
            name: globals()[name] 
            for name in candidate_names 
            if name in globals() and isinstance(globals()[name], pd.DataFrame)
        }
    else:
        # Validate that all provided objects are DataFrames
        available_dfs = {
            name: df 
            for name, df in df_dict.items() 
            if isinstance(df, pd.DataFrame)
        }
    
    if not available_dfs:
        print("No target DataFrames found.")
        return
    
    print(f"📊 Plotting distributions for {len(available_dfs)} DataFrames: {list(available_dfs.keys())}")
    
    for name, df in available_dfs.items():
        cols = df.columns.tolist()
        if not cols:
            print(f"⚠️ {name} is empty, skipping.")
            continue
        
        print(f"\n🔍 Analyzing DataFrame: {name}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {cols}")
        
        # Create subplots (one plot per dataframe column)
        n_cols = len(cols)
        # Use high DPI for publication-quality / high-resolution images
        fig, axes = plt.subplots(1, n_cols, figsize=(figsize_per_col * n_cols, 4), dpi=300, constrained_layout=True)
        if n_cols == 1:
            axes = [axes]
        
        for ax, col in zip(axes, cols):
            series = df[col].dropna()
            nunique = series.nunique()
            
            if len(series) == 0:
                ax.text(0.5, 0.5, f'No data\nin {col}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{col} (empty)")
                continue
            
            # Choose plot type based on number of unique values
            if nunique <= max_unique_for_bars:
                # Categorical/discrete: use bar plot of value counts
                vc = series.value_counts().sort_index()
                bars = sns.barplot(x=[str(x) for x in vc.index], y=vc.values, ax=ax, palette='pastel')
                ax.set_xlabel("Value")
                ax.set_ylabel("Count")
                ax.set_title(f"{col}\n({nunique} unique values)")
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars if not too many
                if len(vc) <= 20:
                    for i, (idx, val) in enumerate(vc.items()):
                        ax.text(i, val + max(vc) * 0.01, str(val), ha='center', va='bottom', fontsize=9)
            else:
                # Continuous-like: use histogram
                sns.histplot(series, bins=min(50, nunique//2), kde=False, ax=ax, color='steelblue', alpha=0.7)
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.set_title(f"{col}\n({nunique} unique values)")
        
        plt.show()
        
        # Print summary statistics
        print(f"   📈 Summary for {name}:")
        for col in cols:
            series = df[col].dropna()
            if len(series) > 0:
                print(f"      {col}: {len(series)} values, {series.nunique()} unique, range [{series.min()}-{series.max()}]")
            else:
                print(f"      {col}: No data")
        print("-" * 50)


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

# --- Main execution ---
if __name__ == "__main__":
    # You can paste your log data directly into this string
    log_data = """
    => Weighting per class: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    Alpha value for generating Lambda with Dirichlet(alpha, alpha, alpha) distribution: 1.0
    ===========================================
    Epoch: [0][0/97], lr: 0.00100	Loss 1.7878 (1.7878)	Prec@1 2.148 (2.148)	Prec@5 23.047 (23.047)
    Epoch: [0][90/97], lr: 0.00100	Loss 1.0272 (1.2338)	Prec@1 0.781 (2.026)	Prec@5 13.281 (24.521)
    Training Results: Prec@1 1.9531 Prec@5 23.8080         Loss 1.224455
    Testing Results: Prec@1 44.8499 Prec@5 88.2093         Loss 1.615152
    Best Prec@1: 44.850

    => Weighting per class: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    Alpha value for generating Lambda with Dirichlet(alpha, alpha, alpha) distribution: 1.0
    ===========================================
    Epoch: [1][0/97], lr: 0.00100	Loss 1.0694 (1.0694)	Prec@1 0.195 (0.195)	Prec@5 16.992 (16.992)
    Epoch: [1][90/97], lr: 0.00100	Loss 1.0248 (1.0850)	Prec@1 0.781 (0.989)	Prec@5 9.180 (11.388)
    Training Results: Prec@1 0.9786 Prec@5 11.2637         Loss 1.082576
    Testing Results: Prec@1 52.3849 Prec@5 95.6517         Loss 1.254187
    Best Prec@1: 52.385

    => Weighting per class: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    Alpha value for generating Lambda with Dirichlet(alpha, alpha, alpha) distribution: 1.0
    ===========================================
    Epoch: [2][0/97], lr: 0.00100	Loss 0.9539 (0.9539)	Prec@1 0.391 (0.391)	Prec@5 7.031 (7.031)
    Epoch: [2][90/97], lr: 0.00100	Loss 0.9736 (1.0037)	Prec@1 0.586 (0.642)	Prec@5 11.719 (10.433)
    Training Results: Prec@1 0.6403 Prec@5 10.3254         Loss 1.002666
    Testing Results: Prec@1 51.9120 Prec@5 93.0201         Loss 1.346574
    Best Prec@1: 52.385
    """

    # To use a log file instead of the string, comment out the `log_data`
    # variable and uncomment the following lines.
    #
    # log_file_path = "your_training_log.log"
    # try:
    #     with open(log_file_path, "r") as f:
    #         log_data = f.read()
    # except FileNotFoundError:
    #     print(f"Error: Log file not found at '{log_file_path}'.")
    #     exit()

    # Parse the log data to extract metrics
    extracted_metrics = parse_training_log(log_data)

    # Check if any data was extracted before plotting
    if not extracted_metrics['epochs']:
        print("Could not find any training or testing results in the log data.")
    else:
        # Generate and save the plots to the 'output' folder.
        # You can change the filename for each run.
        plot_learning_curves(extracted_metrics, output_filename="first_training_run.png")

def plot_transition_matrices(df, true_col='true', figsize=(8, 6), dpi=300, include_missing_values=True):
    """
    Plot transition matrices for all columns vs a reference 'true' column in a DataFrame.
    
    Args:
        df (pd.DataFrame): Combined dataframe with prediction columns and true labels
        true_col (str): Name of the column containing true/reference labels. Default: 'true'
        figsize (tuple): Figure size for each heatmap. Default: (8, 6)
        dpi (int): DPI for high-resolution plots. Default: 300
        include_missing_values (bool): Whether to include missing categorical values with zero counts
        
    Returns:
        dict: Dictionary containing transition counts and proportions for each column
        
    Example usage:
        # Basic usage with standard 'true' column
        results = plot_transition_matrices(df_combined)
        
        # Custom true column name and figure settings
        results = plot_transition_matrices(df_combined, true_col='ground_truth', figsize=(10, 8))
        
        # For CIFAR-20/100 with many classes, use larger figures
        results = plot_transition_matrices(df_combined, figsize=(12, 10))
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
            continue
        
        # Calculate row-normalized proportions P(predicted | true) - rows sum to 1
        transition_props[col] = ct_no_margins.div(ct_no_margins.sum(axis=1), axis=0).fillna(0)
        
        print(f"\nRow-normalized proportions P({col}|{true_col}):")
        print(transition_props[col].round(3))
        
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
        'prediction_columns': pred_cols,
        'true_column': true_col
    }


def plot_transition_matrix_from_file(file_path, matrix_name="Transition Matrix", figsize=(8, 6), dpi=300):
    """
    Load a transition matrix from a text file and plot it as a heatmap.
    
    Args:
        file_path (str): Path to the text file containing the transition matrix
        matrix_name (str): Name/title for the matrix plot. Default: "Transition Matrix"
        figsize (tuple): Figure size for the heatmap. Default: (8, 6)
        dpi (int): DPI for high-resolution plots. Default: 300
        
    Returns:
        np.ndarray: The loaded transition matrix as a numpy array
        
    Example usage:
        # Load and plot a saved transition matrix
        matrix = plot_transition_matrix_from_file('transition_matrix_least_fwd-int_cifar20.txt', 
                                                 'CIFAR-20 FWD-INT Least')
        
        # Load with custom figure size for larger matrices
        matrix = plot_transition_matrix_from_file('transition_matrix_cifar100.txt', 
                                                 'CIFAR-100 Transition Matrix', 
                                                 figsize=(12, 10))
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
        
        return transition_matrix
        
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
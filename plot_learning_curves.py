import re
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Any

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
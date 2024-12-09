import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import savgol_filter

# Set the style
plt.style.use('seaborn')
sns.set_palette("husl")

def smooth_curve(y, window=7):
    """Apply Savitzky-Golay filter to smooth the curve"""
    if len(y) < window:
        return y
    return savgol_filter(y, window, 3)

def add_min_max_annotations(ax, x, y, label):
    """Add min and max annotations to plot"""
    min_idx = np.argmin(y)
    max_idx = np.argmax(y)
    
    ax.annotate(f'Min {label}: {y[min_idx]:.4f}',
                xy=(x[min_idx], y[min_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, alpha=0.7)
    
    ax.annotate(f'Max {label}: {y[max_idx]:.4f}',
                xy=(x[max_idx], y[max_idx]),
                xytext=(10, -10), textcoords='offset points',
                fontsize=8, alpha=0.7)

def calculate_statistics(df):
    """Calculate training statistics"""
    stats = {
        'Loss Improvement': (df['loss'].iloc[0] - df['loss'].iloc[-1]) / df['loss'].iloc[0] * 100,
        'Val Loss Improvement': (df['val_loss'].iloc[0] - df['val_loss'].iloc[-1]) / df['val_loss'].iloc[0] * 100,
        'Final Train-Val Loss Gap': df['loss'].iloc[-1] - df['val_loss'].iloc[-1],
        'Best Epoch (Val AUC)': df.loc[df['val_auc'].idxmax(), 'epoch'],
        'Best Epoch (Val Loss)': df.loc[df['val_loss'].idxmin(), 'epoch'],
        'Convergence Epoch': df.index[df['loss'].diff().abs() < 0.001][0] if any(df['loss'].diff().abs() < 0.001) else len(df),
    }
    return stats

def load_and_plot_metrics(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Calculate statistics
    stats = calculate_statistics(df)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Loss Plot
    ax1 = plt.subplot(2, 2, 1)
    train_loss_smooth = smooth_curve(df['loss'])
    val_loss_smooth = smooth_curve(df['val_loss'])
    
    ax1.plot(df['epoch'], df['loss'], 'b-', alpha=0.3, label='Training Loss (Raw)')
    ax1.plot(df['epoch'], df['val_loss'], 'r-', alpha=0.3, label='Validation Loss (Raw)')
    ax1.plot(df['epoch'], train_loss_smooth, 'b-', label='Training Loss (Smoothed)', linewidth=2)
    ax1.plot(df['epoch'], val_loss_smooth, 'r--', label='Validation Loss (Smoothed)', linewidth=2)
    
    add_min_max_annotations(ax1, df['epoch'], df['loss'], 'Train Loss')
    add_min_max_annotations(ax1, df['epoch'], df['val_loss'], 'Val Loss')
    
    ax1.set_title('Loss Over Epochs', fontsize=14, pad=15)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy Plot
    ax2 = plt.subplot(2, 2, 2)
    train_acc_smooth = smooth_curve(df['accuracy'])
    val_acc_smooth = smooth_curve(df['val_accuracy'])
    
    ax2.plot(df['epoch'], df['accuracy'], 'b-', alpha=0.3, label='Training Accuracy (Raw)')
    ax2.plot(df['epoch'], df['val_accuracy'], 'r-', alpha=0.3, label='Validation Accuracy (Raw)')
    ax2.plot(df['epoch'], train_acc_smooth, 'b-', label='Training Accuracy (Smoothed)', linewidth=2)
    ax2.plot(df['epoch'], val_acc_smooth, 'r--', label='Validation Accuracy (Smoothed)', linewidth=2)
    
    add_min_max_annotations(ax2, df['epoch'], df['accuracy'], 'Train Acc')
    add_min_max_annotations(ax2, df['epoch'], df['val_accuracy'], 'Val Acc')
    
    ax2.set_title('Accuracy Over Epochs', fontsize=14, pad=15)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. AUC Plot
    ax3 = plt.subplot(2, 2, 3)
    train_auc_smooth = smooth_curve(df['auc'])
    val_auc_smooth = smooth_curve(df['val_auc'])
    
    ax3.plot(df['epoch'], df['auc'], 'b-', alpha=0.3, label='Training AUC (Raw)')
    ax3.plot(df['epoch'], df['val_auc'], 'r-', alpha=0.3, label='Validation AUC (Raw)')
    ax3.plot(df['epoch'], train_auc_smooth, 'b-', label='Training AUC (Smoothed)', linewidth=2)
    ax3.plot(df['epoch'], val_auc_smooth, 'r--', label='Validation AUC (Smoothed)', linewidth=2)
    
    add_min_max_annotations(ax3, df['epoch'], df['auc'], 'Train AUC')
    add_min_max_annotations(ax3, df['epoch'], df['val_auc'], 'Val AUC')
    
    ax3.set_title('AUC Over Epochs', fontsize=14, pad=15)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('AUC', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. AUC Difference Plot
    ax4 = plt.subplot(2, 2, 4)
    auc_diff_smooth = smooth_curve(df['auc_diff'])
    
    ax4.plot(df['epoch'], df['auc_diff'], 'g-', alpha=0.3, label='AUC Difference (Raw)')
    ax4.plot(df['epoch'], auc_diff_smooth, 'g-', label='AUC Difference (Smoothed)', linewidth=2)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    add_min_max_annotations(ax4, df['epoch'], df['auc_diff'], 'AUC Diff')
    
    ax4.set_title('AUC Difference (Training - Validation)', fontsize=14, pad=15)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('AUC Difference', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Add a main title
    plt.suptitle('Training Metrics Visualization - ViT Base Model', fontsize=16, y=0.95)
    
    # Add statistics text box
    stats_text = (
        f"Training Statistics:\n"
        f"Loss Improvement: {stats['Loss Improvement']:.1f}%\n"
        f"Val Loss Improvement: {stats['Val Loss Improvement']:.1f}%\n"
        f"Final Train-Val Loss Gap: {stats['Final Train-Val Loss Gap']:.4f}\n"
        f"Best Epoch (Val AUC): {stats['Best Epoch (Val AUC)']}\n"
        f"Best Epoch (Val Loss): {stats['Best Epoch (Val Loss)']}\n"
        f"Convergence Epoch: {stats['Convergence Epoch']}"
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the plot
    plt.savefig('training_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'training_visualization.png'")

    # Print detailed metrics summary
    print("\nTraining Summary:")
    print(f"Final Training Loss: {df['loss'].iloc[-1]:.4f}")
    print(f"Final Validation Loss: {df['val_loss'].iloc[-1]:.4f}")
    print(f"Best Training Accuracy: {df['accuracy'].max():.4f}")
    print(f"Best Validation Accuracy: {df['val_accuracy'].max():.4f}")
    print(f"Best Training AUC: {df['auc'].max():.4f}")
    print(f"Best Validation AUC: {df['val_auc'].max():.4f}")
    print("\nTraining Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    csv_path = "history/runs/run_vit_base_20241207_113554_ep28/metrics.csv"
    load_and_plot_metrics(csv_path)

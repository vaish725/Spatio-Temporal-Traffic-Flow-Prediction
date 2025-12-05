import json
import matplotlib.pyplot as plt
import numpy as np

# Load training history
with open('checkpoints_colab/training_history_optimized.json', 'r') as f:
    history = json.load(f)

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('DCRNN Training Progress (10K Optimal Model)', fontsize=16, fontweight='bold')

# Plot 1: Training and Validation Loss
epochs = range(1, len(history['train_loss']) + 1)
ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss (MAE)', fontsize=12, fontweight='bold')
ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# Plot 2: Validation MAE over epochs
ax2.plot(epochs, history['val_mae'], 'g-', linewidth=2, label='Validation MAE', marker='d', markersize=4)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('MAE (mph)', fontsize=12, fontweight='bold')
ax2.set_title('Validation MAE Progress', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

# Add text annotation for best epoch
best_epoch = np.argmin(history['val_mae']) + 1
best_mae = min(history['val_mae'])
ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best: Epoch {best_epoch}')
ax2.text(best_epoch, best_mae, f'  Best: {best_mae:.3f} mph', 
         verticalalignment='bottom', fontsize=10, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('presentation_training_curves.png', dpi=300, bbox_inches='tight')
print("✅ Saved: presentation_training_curves.png")
plt.show()

# -------------------------------------------------------------------
# Plot 3: Performance Comparison Bar Chart
# -------------------------------------------------------------------
fig2, ax3 = plt.subplots(figsize=(10, 6))

# Using the values from your README (1.930 test MAE)
metrics = ['MAE\n(mph)', 'RMSE\n(mph)', 'MAPE\n(%)']
validation = [2.097, 4.258, 4.76]
test = [1.930, 3.946, 4.48]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax3.bar(x - width/2, validation, width, label='Validation', color='#FF6B6B', alpha=0.8)
bars2 = ax3.bar(x + width/2, test, width, label='Test', color='#4ECDC4', alpha=0.8)

ax3.set_ylabel('Value', fontsize=13, fontweight='bold')
ax3.set_title('Model Performance: Validation vs Test (10K Optimal Model)', fontsize=15, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics, fontsize=12, fontweight='bold')
ax3.legend(fontsize=12)
ax3.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(bars1)
autolabel(bars2)

plt.tight_layout()
plt.savefig('presentation_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: presentation_performance_comparison.png")
plt.show()

# -------------------------------------------------------------------
# Plot 4: Multi-Horizon Performance
# -------------------------------------------------------------------
fig3, ax4 = plt.subplots(figsize=(10, 6))

horizons = ['15 min\n(Step 3)', '30 min\n(Step 6)', '45 min\n(Step 9)', '60 min\n(Step 12)']
mae_values = [1.64, 1.93, 2.08, 2.18]
rmse_values = [3.23, 3.89, 4.26, 4.53]

x = np.arange(len(horizons))
width = 0.35

bars1 = ax4.bar(x - width/2, mae_values, width, label='MAE', color='#95E1D3', alpha=0.8)
bars2 = ax4.bar(x + width/2, rmse_values, width, label='RMSE', color='#F38181', alpha=0.8)

ax4.set_ylabel('Error (mph)', fontsize=13, fontweight='bold')
ax4.set_xlabel('Prediction Horizon', fontsize=13, fontweight='bold')
ax4.set_title('Performance Across Prediction Horizons', fontsize=15, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(horizons, fontsize=11, fontweight='bold')
ax4.legend(fontsize=12)
ax4.grid(True, axis='y', alpha=0.3)

# Add value labels
def autolabel2(bars):
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel2(bars1)
autolabel2(bars2)

plt.tight_layout()
plt.savefig('presentation_horizon_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: presentation_horizon_analysis.png")
plt.show()

print("\n🎉 All presentation plots generated successfully!")
print("\nGenerated files:")
print("1. presentation_training_curves.png - Training/validation curves")
print("2. presentation_performance_comparison.png - Val vs Test comparison")
print("3. presentation_horizon_analysis.png - Multi-horizon performance")
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_curves(history, title=None, out_path=None):
    """Plot training and validation loss/accuracy curves."""
    epochs = len(history['train_loss'])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    # Loss
    axs[0].plot(range(1, epochs+1), history['train_loss'], label='Train Loss')
    axs[0].plot(range(1, epochs+1), history['val_loss'], label='Val Loss')
    axs[0].set_xlabel('Epoch'); axs[0].set_ylabel('Loss')
    axs[0].set_title('Training vs Validation Loss'); axs[0].legend()
    # Accuracy
    axs[1].plot(range(1, epochs+1), history['train_acc'], label='Train Acc')
    axs[1].plot(range(1, epochs+1), history['val_acc'], label='Val Acc')
    axs[1].set_xlabel('Epoch'); axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Training vs Validation Accuracy'); axs[1].legend()
    if title: fig.suptitle(title)
    fig.tight_layout()
    if out_path: plt.savefig(out_path)
    plt.show()

def plot_confusion_matrix(cm, class_names, normalize=False, title=None, out_path=None):
    """Plot a confusion matrix heatmap."""
    cm_disp = cm.astype(float)
    if normalize:
        cm_disp = cm_disp / cm_disp.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_disp, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Class'); plt.ylabel('True Class')
    if title: plt.title(title)
    if out_path: plt.savefig(out_path)
    plt.show()

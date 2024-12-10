import matplotlib.pyplot as plt

def plot_metrics(metrics_data, model_name='Model'):
    # Extracting data for plotting
    epochs = metrics_data['epoch']
    train_loss = metrics_data['loss']
    val_loss = metrics_data['val_loss']
    train_accuracy = metrics_data['accuracy']
    val_accuracy = metrics_data['val_accuracy']
    train_auc = metrics_data['auc']
    val_auc = metrics_data['val_auc']

    # Plotting Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.title(model_name + ': Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    # Plotting Training and Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o')
    plt.title(model_name + ': Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    # Plotting Training and Validation AUC
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_auc, label='Training AUC', marker='o')
    plt.plot(epochs, val_auc, label='Validation AUC', marker='o')
    plt.title(model_name + ': Training and Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid()
    plt.show()

import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from payload.model import RESNETBaseline
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

def load_model(model_path):
    model = RESNETBaseline()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    image = Image.open(image_path).convert('L')
    return transform(image).unsqueeze(0), image

def get_prediction(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
    return prob

def add_prediction_text(image, prob, true_label):
    draw = ImageDraw.Draw(image)
    text = f"Cancer Prob: {prob:.2%}\nTrue Label: {true_label}"
    # Position text at top-left corner with padding
    draw.text((20, 20), text, fill='white', stroke_width=2, stroke_fill='black')
    return image

def get_balanced_samples(csv_path, base_path, n_samples=10):
    """Get n_samples each from positive and negative cases"""
    df = pd.read_csv(csv_path)
    df['img_path'] = df.apply(lambda row: os.path.join(base_path, str(row.patient_id), f"{str(row.image_id)}.png"), axis=1)
    
    # Sample n_samples from each class
    pos_samples = df[df['cancer'] == 1].sample(n=n_samples)
    neg_samples = df[df['cancer'] == 0].sample(n=n_samples)
    
    # Combine and return samples
    return pd.concat([pos_samples, neg_samples])

def main():
    # Load model
    model_path = "models/run_resnet50_20241208_231720_ep6_best_auc_model.pth"
    model = load_model(model_path)
    
    # Get balanced samples from train.csv
    csv_path = 'deploy/train.csv'
    base_path = 'data/png/images_as_pngs_512/train_images_processed_512'
    samples_df = get_balanced_samples(csv_path, base_path)
    
    # Create output directory
    os.makedirs('predictions', exist_ok=True)
    
    # Process each image
    for _, row in samples_df.iterrows():
        try:
            image_path = row['img_path']
            true_label = row['cancer']
            
            # Preprocess and get prediction
            image_tensor, original_image = preprocess_image(image_path)
            prob = get_prediction(model, image_tensor)
            
            # Add prediction text
            annotated_image = add_prediction_text(original_image, prob, true_label)
            
            # Extract filename from path
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            
            # Save annotated image
            output_path = f'predictions/{base_name}_pred{prob:.2f}_true{true_label}.png'
            annotated_image.save(output_path)
            print(f"Processed {output_path}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    main()

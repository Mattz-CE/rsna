import os
import torch
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
from model import get_model  # Assuming this is available
from torch.nn import Sigmoid
import matplotlib.pyplot as plt
from collections import defaultdict

class ModelEvaluator:
    def __init__(self, model_configs, device='cuda'):
        self.device = device
        self.models = {}
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.sigmoid = Sigmoid()
        
        # Initialize all models
        for name, config in model_configs.items():
            model = get_model(
                config['architecture'],
                img_size=512,
                patch_size=16 if 'vit' in config['architecture'] else None
            )
            model.load_state_dict(torch.load(config['path'], map_location=device))
            model.eval()
            model.to(device)
            self.models[name] = model

    def predict_image(self, image_path):
        image = Image.open(image_path).convert('L')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        predictions = {}
        with torch.no_grad():
            for name, model in self.models.items():
                output = model(tensor)
                prob = self.sigmoid(output).item()
                predictions[name] = prob
        
        return predictions

    def visualize_prediction(self, image_path, predictions, true_label, save_path=None):
        # Open and resize image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512))
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Draw true label at the top
        label_text = f"True: {'Positive' if true_label == 1 else 'Negative'}"
        draw.text((10, 10), label_text, fill='white', font=font)
        
        # Draw predictions
        y_position = 40
        for model_name, prob in predictions.items():
            prediction = "Positive" if prob >= 0.5 else "Negative"
            confidence = prob if prob >= 0.5 else 1 - prob
            text = f"{model_name}: {prediction} ({confidence:.2f})"
            
            # Color based on correctness
            color = 'green' if (prob >= 0.5) == true_label else 'red'
            draw.text((10, y_position), text, fill=color, font=font)
            y_position += 25

        if save_path:
            image.save(save_path)
        return image

def main():
    # Model configurations
    model_configs = {
        'EfficientNet': {
            'architecture': 'efficientnet',
            'path': 'models/run_efficientnet_20241205_114438_ep7_best_auc_model.pth'
        },
        'ResNet50': {
            'architecture': 'resnet50',
            'path': 'models/run_resnet50_20241208_231720_ep6_best_auc_model.pth'
        },
        'ResNet101': {
            'architecture': 'resnet101',
            'path': 'models/run_resnet101_20241204_201138_ep14.pth'
        },
        'ViT-Base': {
            'architecture': 'vit_base',
            'path': 'models/run_vit_base_20241207_113554_ep28_best_auc_model.pth'
        },
        'ViT-Medium': {
            'architecture': 'vit_medium',
            'path': 'models/run_vit_mediumd_20241209_002034_ep26_best_auc_model.pth'
        }
    }

    # Initialize evaluator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = ModelEvaluator(model_configs, device)

    # Load dataset
    df = pd.read_csv('train.csv')
    base_path = '/kaggle/input/rsna-512/train_images_processed_512'  # Adjust path as needed

    # Add image paths
    df['img_path'] = df.apply(
        lambda row: os.path.join(base_path, str(row.patient_id), f"{str(row.image_id)}.png"),
        axis=1
    )

    # Sample 10 images from each class
    pos_samples = df[df['cancer'] == 1].sample(10)
    neg_samples = df[df['cancer'] == 0].sample(10)
    samples = pd.concat([pos_samples, neg_samples])

    # Create output directory
    os.makedirs('predictions', exist_ok=True)

    # Process each image
    for idx, row in samples.iterrows():
        predictions = evaluator.predict_image(row['img_path'])
        save_path = f"predictions/pred_{row['patient_id']}_{row['image_id']}.png"
        
        evaluator.visualize_prediction(
            row['img_path'],
            predictions,
            row['cancer'],
            save_path
        )
        print(f"Processed image {row['patient_id']}_{row['image_id']}")

    print("Completed! Check the 'predictions' directory for results.")

if __name__ == '__main__':
    main()
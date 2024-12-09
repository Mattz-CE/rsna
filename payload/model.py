import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchinfo import summary
import timm
    
###### RESNET ######

class RESNETBaseline(nn.Module):
    def __init__(self, img_size=None, patch_size=None, **kwargs):
        super(RESNETBaseline, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            self.resnet.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        
        # Modify final fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Freeze all layers by default
        self._freeze_layers()

    def _freeze_layers(self):
        """Freeze all layers except the final classifier"""
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Unfreeze the final fully connected layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def unfreeze_last_n_layers(self, n=3):
        """Unfreeze the last n Bottleneck modules of ResNet"""
        # First freeze everything
        self._freeze_layers()
        
        # Get the last n Bottleneck modules from layer4
        bottlenecks = list(self.resnet.layer4)[-n:]
        
        # Unfreeze only these Bottleneck modules
        for bottleneck in bottlenecks:
            for param in bottleneck.parameters():
                param.requires_grad = True
        
        # Always ensure fc layer is unfrozen
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x)

###### EfficientNetV2 ######
class RSNAEfficientNetV2(nn.Module):
    def __init__(self, img_size=None, patch_size=None, pretrained_weights=None, **kwargs):
        super(RSNAEfficientNetV2, self).__init__()
        self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        
        # Modify first conv layer for single channel input
        original_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Average the weights of the first conv layer
        with torch.no_grad():
            self.model.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                
        # Modify classifier for binary output
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_ftrs, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Freeze all layers by default
        self._freeze_layers()

    def _freeze_layers(self):
        """Freeze all layers except the classifier"""
        for param in self.model.features.parameters():
            param.requires_grad = False
        # Unfreeze the classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def unfreeze_last_n_layers(self, n=3):
        """Unfreeze the last n MBConv blocks"""
        # First freeze everything
        self._freeze_layers()
        
        # Find the last n MBConv blocks
        mbconv_blocks = []
        for module in self.model.modules():
            if isinstance(module, (models.efficientnet.MBConv, models.efficientnet.FusedMBConv)):
                mbconv_blocks.append(module)
        
        # Unfreeze only the last n MBConv blocks
        for block in mbconv_blocks[-n:]:
            for param in block.parameters():
                if not isinstance(param, nn.BatchNorm2d):
                    param.requires_grad = True
        
        # Always unfreeze classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)
    
###### RSNAViTBase ######
class RSNAViTBase(nn.Module):
    def __init__(self, img_size=512, patch_size=16, **kwargs):
        super(RSNAViTBase, self).__init__()
        # Ensure img_size is compatible with patch_size
        assert img_size % patch_size == 0, f"Image size {img_size} must be divisible by patch size {patch_size}"
        
        # Load pretrained ViT
        self.model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=1,
            img_size=img_size,
            patch_size=patch_size
        )
        
        # Modify patch embedding for single channel input
        original_patch_embed = self.model.patch_embed.proj
        self.model.patch_embed.proj = nn.Conv2d(
            1, original_patch_embed.out_channels,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Initialize new patch embedding with average of pretrained weights
        with torch.no_grad():
            self.model.patch_embed.proj.weight.data = original_patch_embed.weight.data.mean(
                dim=1, keepdim=True
            )
        
        # Add sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()
        
        # Freeze all layers by default
        self._freeze_layers()
    
    def _freeze_layers(self):
        """Freeze all layers except the head"""
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the classification head
        for param in self.model.head.parameters():
            param.requires_grad = True
    
    def unfreeze_last_n_layers(self, n=3):
        """Unfreeze the last n transformer blocks and the head"""
        # First freeze everything
        self._freeze_layers()
        
        # Unfreeze the last n transformer blocks
        for block in self.model.blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        
        # Always unfreeze head
        for param in self.model.head.parameters():
            param.requires_grad = True
        
        # Unfreeze layer norm layers
        for param in self.model.norm.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

###### RSNAViTMediumD ######
class RSNAViTMediumD(nn.Module):
    def __init__(self, img_size=384, patch_size=16, **kwargs):
        super(RSNAViTMediumD, self).__init__()
        # Ensure img_size is compatible with patch_size
        assert img_size % patch_size == 0, f"Image size {img_size} must be divisible by patch size {patch_size}"
        
        # Load pretrained ViT MediumD with registers and global average pooling
        self.model = timm.create_model(
            'vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k',
            pretrained=True,
            num_classes=1,
            img_size=img_size,
            patch_size=patch_size
        )
        
        # Modify patch embedding for single channel input
        original_patch_embed = self.model.patch_embed.proj
        self.model.patch_embed.proj = nn.Conv2d(
            1, original_patch_embed.out_channels,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Initialize new patch embedding with average of pretrained weights
        with torch.no_grad():
            self.model.patch_embed.proj.weight.data = original_patch_embed.weight.data.mean(
                dim=1, keepdim=True
            )
        
        # Add sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()
        
        # Freeze all layers by default
        self._freeze_layers()
    
    def _freeze_layers(self):
        """Freeze all layers except the head"""
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the classification head
        for param in self.model.head.parameters():
            param.requires_grad = True
    
    def unfreeze_last_n_layers(self, n=3):
        """Unfreeze the last n transformer blocks and the head"""
        # First freeze everything
        self._freeze_layers()
        
        # Unfreeze the last n transformer blocks
        for block in self.model.blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        
        # Always unfreeze head
        for param in self.model.head.parameters():
            param.requires_grad = True
        
        # Unfreeze layer norm layers
        for param in self.model.norm.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

###### FETCH #######

def get_model(model_name, **kwargs):
    """Factory function to get the specified model"""
    models = {
        'resnet': RESNETBaseline,
        'effinet': RSNAEfficientNetV2,
        'vit_base': RSNAViTBase,
        'vit_mediumd': RSNAViTMediumD
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    
    return models[model_name](**kwargs)

def print_model_summary(model, input_size=(1, 1, 512, 512), batch_size=1, device="cpu"):
    """Helper function to print model summary with consistent formatting"""
    print(f"\n{'='*80}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*80}")
    
    # Print trainable parameters info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {(trainable_params/total_params)*100:.2f}%\n")
    
    try:
        summary(model, input_size=(batch_size, *input_size[1:]), device=device, verbose=1)
    except Exception as e:
        print(f"Note: Detailed model summary unavailable for {model.__class__.__name__}. This is expected for some architectures.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define standard input size for all models
    INPUT_SIZE = (1, 1, 512, 512)  # (batch_size, channels, height, width)
    
    # Initialize all models
    models_to_test = {
        'resnet': get_model('resnet'),
        'effinet': get_model('effinet'),
        'vit_base': get_model('vit_base'),
        'vit_mediumd': get_model('vit_mediumd')
    }
    
    # Print summary for each model
    for name, model in models_to_test.items():
        try:
            model = model.to(device)
            print(f"\nTesting {name}:")
            print("-" * 50)
            
            # Unfreeze last few layers and show summary again
            model.unfreeze_last_n_layers(n=3)
            print("\nModel summary after unfreezing last few layers:")
            print_model_summary(model, input_size=INPUT_SIZE, device=device)
            
            # Test forward pass
            dummy_input = torch.randn(INPUT_SIZE).to(device)
            output = model(dummy_input)
            print(f"Output shape: {output.shape}\n")
            
        except Exception as e:
            print(f"Error testing {name}: {str(e)}\n")
            continue

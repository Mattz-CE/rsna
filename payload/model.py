import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
    
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

    def unfreeze_last_n_layers(self, n=2):
        """Unfreeze the last n blocks of the ResNet"""
        # First freeze everything
        self._freeze_layers()
        
        # Unfreeze the last n blocks (each block has multiple layers)
        layers_to_unfreeze = list(self.resnet.layer4[-n:])  # Last n blocks of layer4
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Always unfreeze FC layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x)


###### ViT ######

class PatchEmbed(nn.Module):
    """Split image into patches and embed them"""
    def __init__(self, img_size=512, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, E, H', W')
        x = x.flatten(2)  # (B, E, H'*W')
        x = x.transpose(1, 2)  # (B, H'*W', E)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            dropout=dropout
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_channels=3,
        num_classes=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        dropout=0.1,
        embed_dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(embed_dropout)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

class RSNAViT(nn.Module):
    def __init__(self, img_size=512, patch_size=16):
        super().__init__()
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,
            num_classes=1,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            dropout=0.1,
            embed_dropout=0.1
        )
        self.sigmoid = nn.Sigmoid()
        
        # Freeze all layers by default
        self._freeze_layers()

    def _freeze_layers(self):
        """Freeze all layers except the final classifier"""
        for param in self.vit.parameters():
            param.requires_grad = False
        # Unfreeze the head
        for param in self.vit.head.parameters():
            param.requires_grad = True

    def unfreeze_last_n_layers(self, n=2):
        """Unfreeze the last n transformer blocks"""
        # First freeze everything
        self._freeze_layers()
        
        # Unfreeze the last n transformer blocks
        for block in self.vit.blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        
        # Always unfreeze head
        for param in self.vit.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.vit(x)
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

###### FETCH #######

def get_model(model_name, **kwargs):
    """Factory function to get the specified model"""
    models = {
        'resnet50': RESNETBaseline,
        'vit': RSNAViT,
        'efficientnet': RSNAEfficientNetV2
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
    
    summary(model, input_size=(batch_size, *input_size[1:]), device=device)
    print(f"{'='*80}\n")

if __name__ == "__main__":
    from torchinfo import summary
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define standard input size for all models
    INPUT_SIZE = (1, 1, 512, 512)  # (batch_size, channels, height, width)
    
    # Initialize all models
    models_to_test = {
        'ResNet50': get_model('resnet50'),
        'Vision Transformer': get_model('vit', img_size=512, patch_size=16),
        'EfficientNetV2': get_model('efficientnet')
    }
    
    # Print summary for each model
    for name, model in models_to_test.items():
        try:
            model = model.to(device)
            print(f"\nTesting {name}:")
            print("-" * 50)
            
            # Show summary with frozen layers
            print("\nModel summary with frozen layers:")
            print_model_summary(model, input_size=INPUT_SIZE, device=device)
            
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
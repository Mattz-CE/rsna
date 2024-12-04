#!/bin/bash

# Check if symlink already exists
if [ -L "/root/.cache" ]; then
    echo "Symlink already exists, skipping setup"
    ls -la /root/.cache
    # jump directly to pip
    pip install scikit-learn pandas kagglehub torchsummary tensorboard
    python train.py
    exit 0
fi

# Create workspace cache directory if it doesn't exist
mkdir -p /workspace/.cache

# Check if /root/.cache exists and is not a symlink
if [ -d "/root/.cache" ] && [ ! -L "/root/.cache" ]; then
    echo "Moving existing cache contents to workspace..."
    # Move contents preserving structure
    mv /root/.cache/* /workspace/.cache/ 2>/dev/null || true
    rm -rf /root/.cache
fi

# Create symlink
ln -sf /workspace/.cache /root/.cache

echo "Cache symlink created. Using /workspace/.cache for storage"

# Show the result
ls -la /root/.cache
df -h

pip install scikit-learn pandas kagglehub torchsummary tensorboard
python train.py
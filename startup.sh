# 
git clone https://github.com/Mattz-CE/rsna

# Update Packages
apt update
apt install screen

# Monitors GPU
screen -dmS gpu-monitor bash -c 'while true; do date "+%Y-%m-%d %H:%M:%S" >> gpu.log; nvidia-smi >> gpu.log; sleep 0.2; done'


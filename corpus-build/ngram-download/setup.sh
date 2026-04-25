#!/bin/bash
# Setup script for fresh EC2 instance
# Instance: m5.xlarge on-demand, us-east-1, Amazon Linux 2023
# EBS: 800GB gp3 mounted at /home/ec2-user/gdelt

set -e

echo "=== Setting up GDELT NGrams Pipeline ==="

# Update and install
sudo yum update -y
sudo yum install -y python3 python3-pip tmux htop

# Install Python deps
pip3 install --user requests boto3 tqdm

# Create directories
mkdir -p /home/ec2-user/gdelt/{raw,reconstructed,filtered,logs}

# Copy scripts to working directory
cp /home/ec2-user/aws/*.py /home/ec2-user/gdelt/
cp /home/ec2-user/aws/run_all.sh /home/ec2-user/gdelt/

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start the pipeline:"
echo "  cd /home/ec2-user/gdelt"
echo "  tmux new -s gdelt"
echo "  bash run_all.sh"
echo ""
echo "To configure S3 sync:"
echo "  aws configure"
echo "  echo 'your-bucket-name' > /home/ec2-user/gdelt/s3_bucket.txt"

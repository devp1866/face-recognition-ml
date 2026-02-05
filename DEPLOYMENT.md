# ☁️ AWS Deployment Guide for Smart Face

This guide helps you deploy the Smart Face application to an **AWS EC2 Instance** (Free Tier eligible) using **Docker**.

---

## 🏗️ Architecture

We use **Docker** to package the app. This ensures it runs exactly the same on AWS as it does on your machine, eliminating "it works on my machine" issues.

- **App**: Flask + InsightFace (Python)
- **Server**: Gunicorn (Production WSGI)
- **Container**: Docker (Debian/Python 3.10)

---

## 📋 Prerequisites

1.  **AWS Account** (Create at aws.amazon.com)
2.  **Terminal/Command Prompt** on your local PC.

---

## 🚀 Step 1: Launch EC2 Instance

1.  Login to **AWS Console** > **EC2** > **Launch Instance**.
2.  **Name**: `SmartFace-Server`
3.  **OS Image**: `Ubuntu Server 24.04 LTS` (Free Tier).
4.  **Instance Type**: `t2.micro` (Free Tier) or `t3.medium` (Recommended for AI speed).
5.  **Key Pair**: Create new > Name: `smartface-key` > Download `.pem` file.
6.  **Network Settings**:
    - Allow SSH traffic from `My IP`.
    - Allow HTTP traffic from the internet.
    - Allow HTTPS traffic from the internet.
7.  **Launch Instance**.

---

## 🛡️ Step 2: Configure Security Group (Firewall)

1.  Go to your Instance Summary > **Security** > Click the **Security Group**.
2.  **Edit Inbound Rules** > **Add Rule**:
    - **Type**: Custom TCP
    - **Port**: `5000`
    - **Source**: `0.0.0.0/0` (Anywhere)
3.  Save Rules.

---

## 💻 Step 3: Connect to Server

Open your terminal (where you saved the `.pem` key):

```bash
# Set permissions (Linux/Mac only, skip on Windows)
chmod 400 smartface-key.pem

# SSH into the server (Replace 1.2.3.4 with your EC2 Public IP)
ssh -i "smartface-key.pem" ubuntu@1.2.3.4
```

---

## 🐳 Step 4: Install Docker on EC2

Run these commands inside the EC2 terminal:

```bash
# Update System
sudo apt-get update
sudo apt-get install -y docker.io git

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to Docker group (avoids typing sudo every time)
sudo usermod -aG docker $USER
```

_exit and reconnect via SSH for changes to take effect._

---

## 📦 Step 5: Deploy App

### Option A: Clone from GitHub (Easiest)

_(Make sure your repo is public, or check into private repo setup)_

```bash
# 1. Clone Repo
git clone https://github.com/devp1866/face-recognition-ml.git
cd face-recognition-ml

# 2. Setup SWAP Memory (Crucial for Free Tier t2.micro RAM)
# InsightFace needs RAM. Free Tier only has 1GB. We add 2GB virtual RAM.
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 3. Download Anti-Spoofing Model (Manually required as it's not in git)
# Create directories
mkdir -p resources/models
# Download MiniFASNet
wget -O resources/models/minifasv2.onnx https://github.com/suriAI/face-antispoof-onnx/blob/main/models/best/98.20/best_model.onnx

# 4. Build Docker Image (Takes ~5-10 mins)
docker build -t smartface .

# 4. Run Container
# -d: Run in background
# -p: Map port 5000
# --restart: Auto-restart if crashes
docker run -d -p 5000:5000 --name smartface_app --restart always smartface
```

---

## 🌐 Step 6: Access Your App

Go to your browser and type:
`http://YOUR_EC2_PUBLIC_IP:5000`

Result: **Smart Face Live Dashboard** 🚀

---

## 🔧 Maintenance

- **View Logs**: `docker logs -f smartface_app`
- **Stop App**: `docker stop smartface_app`
- **Update App**:
  ```bash
  git pull
  docker build -t smartface .
  docker stop smartface_app
  docker rm smartface_app
  docker run -d -p 5000:5000 --name smartface_app --restart always smartface
  ```

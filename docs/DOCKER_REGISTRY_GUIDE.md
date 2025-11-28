# Docker Registry Setup Guide

How to use Docker registries with Enroot on HPC clusters.

---

## 🎯 Overview

Enroot supports pulling container images directly from Docker registries, enabling:
- ✅ Automatic GPU mounting (no manual configuration)
- ✅ Easy version control and updates
- ✅ Team collaboration
- ✅ No local .tar/.sqsh file management

---

## 🔐 Authentication

### Setup Credentials (One-Time)

Create `~/.config/enroot/.credentials` on the cluster:

```bash
mkdir -p ~/.config/enroot

cat >> ~/.config/enroot/.credentials << 'EOF'
machine nvcr.io login $oauthtoken password YOUR_NGC_API_KEY
machine docker.io login YOUR_DOCKERHUB_USERNAME password YOUR_DOCKERHUB_TOKEN
machine ghcr.io login YOUR_GITHUB_USERNAME password YOUR_GITHUB_TOKEN
EOF

chmod 600 ~/.config/enroot/.credentials
```

**Important**: Replace placeholders with your actual credentials.

---

## 📦 Supported Registries

### 1. NVIDIA NGC (nvcr.io)

**Purpose**: Official NVIDIA containers (PyTorch, TensorFlow, etc.)

**Get API Key**:
1. Create account at: https://ngc.nvidia.com/
2. Navigate to: Setup → Generate API Key
3. Copy key

**Usage** (Public images don't need credentials):
```bash
srun --container-image=nvcr.io/nvidia/pytorch:24.10-py3 --gpus=1 --pty /bin/bash
```

### 2. Docker Hub (docker.io)

**Purpose**: Public and private Docker images

**Get Token**:
1. Login at: https://hub.docker.com/
2. Account Settings → Security → New Access Token
3. Copy token

**Usage**:
```bash
# Public image
srun --container-image=docker.io/python:3.10 --gpus=1 --pty /bin/bash

# Private image (requires .credentials)
srun --container-image=docker.io/myuser/myproject:v1.0 --gpus=1 --pty /bin/bash
```

### 3. GitHub Container Registry (ghcr.io)

**Purpose**: GitHub-hosted container images

**Get Token**:
1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select scope: `read:packages` (or `write:packages` for push)
4. Copy token

**Usage**:
```bash
# Public image
srun --container-image=ghcr.io/username/project:latest --gpus=1 --pty /bin/bash

# Private image (requires .credentials)
srun --container-image=ghcr.io/org/private-project:v1.0 --gpus=1 --pty /bin/bash
```

### 4. Private/Harbor Registry

**Purpose**: Enterprise or self-hosted registries

**Setup**:
```bash
# Add to .credentials
cat >> ~/.config/enroot/.credentials << 'EOF'
machine my-registry.com login myuser password mypassword
EOF
```

**Usage**:
```bash
srun --container-image=my-registry.com/project:v1.0 --gpus=1 --pty /bin/bash
```

---

## 🚀 Complete Workflow Example

### Scenario: Deploy Custom PyTorch Project

#### Step 1: Build on Local Machine

**Create Dockerfile** (use provided template):
```dockerfile
FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

CMD ["/bin/bash"]
```

**Build & Push**:
```bash
# Build
docker build -t ghcr.io/myuser/myproject:v1.0 .

# Test locally
docker run --gpus all -it ghcr.io/myuser/myproject:v1.0 python3 -c "import torch; print(torch.cuda.is_available())"

# Login & Push
echo $GITHUB_TOKEN | docker login ghcr.io -u myuser --password-stdin
docker push ghcr.io/myuser/myproject:v1.0
```

#### Step 2: Use on Cluster

**Configure credentials** (one-time):
```bash
mkdir -p ~/.config/enroot
cat >> ~/.config/enroot/.credentials << EOF
machine ghcr.io login myuser password $GITHUB_TOKEN
EOF
chmod 600 ~/.config/enroot/.credentials
```

**Launch**:
```bash
srun --account=your-account \
     --partition=interactive \
     --gpus=1 \
     --container-image=ghcr.io/myuser/myproject:v1.0 \
     --container-mounts=/path/to/project:/workspace \
     --pty /bin/bash
```

#### Step 3: Inside Container

```bash
# Verify environment
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# Run your code
python3 train.py
```

---

## 🔄 Update Workflow

When you update your code:

```bash
# On local machine
docker build -t ghcr.io/myuser/myproject:v1.1 .
docker push ghcr.io/myuser/myproject:v1.1

# On cluster - just change version
srun --container-image=ghcr.io/myuser/myproject:v1.1 ...
```

Enroot automatically pulls the new version.

---

## 🛡️ Security Best Practices

### 1. Protect Credentials

```bash
# Correct permissions
chmod 600 ~/.config/enroot/.credentials

# Verify
ls -la ~/.config/enroot/.credentials
# Should show: -rw------- (owner read/write only)
```

### 2. Use Tokens, Not Passwords

- ✅ Use API tokens (can be revoked)
- ❌ Avoid using account passwords
- ✅ Set token expiration dates
- ✅ Use minimal required permissions

### 3. Public vs Private Images

**Public images**: Don't require credentials
```bash
# No .credentials needed
srun --container-image=nvcr.io/nvidia/pytorch:24.10-py3 ...
```

**Private images**: Require credentials
```bash
# Needs .credentials with proper auth
srun --container-image=ghcr.io/myorg/private:v1.0 ...
```

---

## 🐛 Troubleshooting

### Issue: Authentication Failed

```bash
[ERROR] Could not authenticate to registry
```

**Solutions**:
1. Check `.credentials` file exists: `cat ~/.config/enroot/.credentials`
2. Verify permissions: `ls -la ~/.config/enroot/.credentials`
3. Test token:
   ```bash
   # Docker Hub
   echo $TOKEN | docker login docker.io -u username --password-stdin
   
   # GitHub
   echo $TOKEN | docker login ghcr.io -u username --password-stdin
   ```
4. Regenerate token if expired

### Issue: Image Not Found

```bash
[ERROR] Image not found: registry.com/project:v1.0
```

**Solutions**:
1. Verify image exists in registry
2. Check image name spelling (case-sensitive)
3. For private images, ensure credentials are set
4. Try pulling with Docker first to verify:
   ```bash
   docker pull registry.com/project:v1.0
   ```

### Issue: Slow Pull Speed

**Tip**: Enroot caches images in `~/.local/share/enroot/`

```bash
# Check cache
ls -lh ~/.local/share/enroot/

# Clear cache if needed
rm -rf ~/.local/share/enroot/*
```

---

## 📊 Registry Comparison

| Registry | Best For | Auth Required | GPU Auto-Mount |
|----------|----------|---------------|----------------|
| **NGC** | NVIDIA containers | Public: No, Private: Yes | ✅ Yes |
| **Docker Hub** | General containers | Public: No, Private: Yes | ⚠️ Depends on image |
| **GitHub CR** | Project collaboration | Public: No, Private: Yes | ⚠️ Depends on image |
| **Harbor** | Enterprise | Usually Yes | ⚠️ Depends on image |

**Recommendation**: Use NGC-based images for guaranteed GPU support.

---

## 💡 Tips

### Tip 1: Tag Versions Properly

```bash
# Good
docker tag image:latest registry.com/project:v1.0
docker tag image:latest registry.com/project:latest

# Avoid
docker tag image:latest registry.com/project:test123
```

### Tip 2: Layer Caching

Order Dockerfile commands for better caching:
```dockerfile
# 1. Base image (rarely changes)
FROM nvcr.io/nvidia/pytorch:24.10-py3

# 2. System packages (rarely change)
RUN apt-get update && apt-get install -y ...

# 3. Python dependencies (change occasionally)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 4. Code (changes frequently)
COPY . .
```

### Tip 3: Multi-Stage Builds

For smaller images:
```dockerfile
# Build stage
FROM nvcr.io/nvidia/pytorch:24.10-py3 AS builder
COPY . /build
RUN python setup.py bdist_wheel

# Runtime stage
FROM nvcr.io/nvidia/pytorch:24.10-py3
COPY --from=builder /build/dist/*.whl .
RUN pip install *.whl
```

---

## 📚 References

- **Enroot Documentation**: https://github.com/NVIDIA/enroot
- **NGC Catalog**: https://catalog.ngc.nvidia.com/
- **Docker Hub**: https://hub.docker.com/
- **GitHub Packages**: https://docs.github.com/packages

---

**Last Updated**: 2025-11-28

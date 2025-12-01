# Milestone 1: Data Collection & Baseline Calibration - Execution Plan

## Project Goal
Train a Gaussian Splat model using COLMAP-generated data and render the results for validation.

## Current Status
- ✅ Repository cloned locally
- ✅ Documentation reviewed and comprehensive guide created
- ⏳ GCP VM setup pending
- ⏳ Data collection pending
- ⏳ Training execution pending

---

## Execution Checklist

### Phase 1: GCP VM Setup (Est. Time: 30-45 minutes)

#### 1.1 Verify GCP Project Access
```powershell
# Check your GCP projects
gcloud projects list

# Set active project (replace with your project ID)
gcloud config set project YOUR_PROJECT_ID
```

#### 1.2 Check GPU Quota
Before creating the VM, verify you have GPU quota:
```powershell
gcloud compute project-info describe --project=YOUR_PROJECT_ID
```

If quota is 0, request increase:
1. Go to: https://console.cloud.google.com/iam-admin/quotas
2. Filter: `gpus_all_regions`
3. Request quota increase to 1+ GPUs
4. Justification: "Training 3D Gaussian Splatting models for computer vision research"

#### 1.3 Create VM Instance (RECOMMENDED: Use Console for First Time)

**Option A: Via GCP Console (Recommended for First Time)**
1. Navigate to: https://console.cloud.google.com/compute/instances
2. Click "CREATE INSTANCE"
3. Configure as per COMPLETE_GCP_TRAINING_GUIDE.md Section "VM Instance Setup"
   - Name: `gaussian-splatting-vm`
   - Region: `us-west1-b` (Oregon)
   - Machine: `g2-standard-4` (includes L4 GPU)
   - Boot Disk: Deep Learning VM with CUDA 11.8, 100GB SSD
   - Firewall: Allow HTTP/HTTPS traffic

**Option B: Via gcloud CLI (Advanced)**
```powershell
gcloud compute instances create gaussian-splatting-vm `
  --zone=us-west1-b `
  --machine-type=g2-standard-4 `
  --accelerator="type=nvidia-l4,count=1" `
  --image-family=common-cu118 `
  --image-project=deeplearning-platform-release `
  --boot-disk-size=100GB `
  --boot-disk-type=pd-ssd `
  --metadata="install-nvidia-driver=True"
```

#### 1.4 Connect to VM
```powershell
gcloud compute ssh gaussian-splatting-vm --zone=us-west1-b
```

**✅ Verification Step**: Once connected, run:
```bash
nvidia-smi  # Should show NVIDIA L4 GPU
```

---

### Phase 2: Environment Setup on VM (Est. Time: 15-20 minutes)

#### 2.1 Install System Dependencies
```bash
# Update package manager
sudo apt-get update

# Install COLMAP for Structure-from-Motion
sudo apt-get install -y colmap

# Install FFmpeg for video processing
sudo apt-get install -y ffmpeg

# Install git
sudo apt-get install -y git

# Verify installations
colmap help
ffmpeg -version
```

#### 2.2 Clone Repository
```bash
cd ~
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
```

#### 2.3 Setup Conda Environment
```bash
# Verify conda is available
which conda

# If not found, add to PATH
export PATH=/opt/conda/bin:$PATH
conda init
source ~/.bashrc

# Create environment with Python 3.8
conda create -n gaussian_splatting python=3.8 -y
conda activate gaussian_splatting

# Install PyTorch 2.0 with CUDA 11.8 (CRITICAL for L4 GPU)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install plyfile tqdm opencv-python joblib

# Install CUDA extensions
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim
```

#### 2.4 Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Expected Output**:
```
PyTorch: 2.0.1
CUDA available: True
GPU: NVIDIA L4
```

---

### Phase 3: Data Collection & Preparation (Est. Time: Variable)

You have three options depending on your data source:

#### Option A: Test with Sample Dataset (RECOMMENDED FIRST)
```bash
# On VM
cd ~/gaussian-splatting
mkdir -p data
cd data

# Download sample Tanks & Temples dataset (650 MB)
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip tandt_db.zip

# Verify structure
ls tandt/train/
# Should see: images/, sparse/, etc.
```

**Why start here?**: Validates your entire pipeline before processing custom data.

#### Option B: Custom Images (Your Photos)

**On Local Machine** (prepare your images):
1. Collect 50-200 images of your scene
2. Requirements:
   - High resolution (1920x1080 or higher)
   - 70-80% overlap between consecutive images
   - Consistent lighting
   - Good coverage from multiple angles
3. Organize in a folder: `e:\Dev\gaussian-splatting\local-data\your-scene\`

**Upload to VM**:
```powershell
# On local machine (PowerShell)
gcloud compute scp --recurse e:\Dev\gaussian-splatting\local-data\your-scene\ gaussian-splatting-vm:~/gaussian-splatting/data/your-scene-images/ --zone=us-west1-b
```

**Process on VM**:
```bash
# On VM
cd ~/gaussian-splatting
mkdir -p data/your-scene/input
mv data/your-scene-images/* data/your-scene/input/

# Run COLMAP to generate sparse reconstruction
conda activate gaussian_splatting
python convert.py -s data/your-scene

# This takes 15-60 minutes depending on number of images
# Generates: data/your-scene/sparse/0/ directory
```

#### Option C: Video to 3D Model

**Upload Video**:
```powershell
# On local machine
gcloud compute scp e:\path\to\your-video.mp4 gaussian-splatting-vm:~/gaussian-splatting/ --zone=us-west1-b
```

**Extract Frames on VM**:
```bash
# On VM
cd ~/gaussian-splatting
mkdir -p data/video-scene/input

# Extract 2 frames per second (adjust based on video)
ffmpeg -i your-video.mp4 -qscale:v 1 -qmin 1 -vf "fps=2" data/video-scene/input/%04d.jpg

# Check frame count
ls data/video-scene/input/ | wc -l
# Aim for 100-300 frames

# Run COLMAP
python convert.py -s data/video-scene
```

**✅ Verification Step**: After COLMAP completes, verify sparse reconstruction:
```bash
ls data/your-scene/sparse/0/
# Should contain: cameras.bin, images.bin, points3D.bin
```

**Transfer dataset to vm**
```
gcloud compute scp --recurse data\silverlake zackt@gaussian-splatting-vm:gaussian-splatting/data/ --zone=us-west1-a
```
---

### Phase 4: Baseline Training (Est. Time: 20-40 minutes)

#### 4.1 Start Training
```bash
cd ~/gaussian-splatting
conda activate gaussian_splatting

# For sample data:
python train.py -s data/tandt/train --eval

# For custom data:
python train.py -s data/your-scene --eval

# Training parameters explained:
# -s : source data path
# --eval : creates train/test split for metrics
# --iterations 30000 : default, full training (optional to specify)
# -r 2 : reduce resolution by 50% for faster training (optional)
```

#### 4.2 Monitor Training Progress

**Option 1: Watch Log File**
Open a second SSH session:
```powershell
# In new local PowerShell window
gcloud compute ssh gaussian-splatting-vm --zone=us-west1-b
```

Then monitor:
```bash
# On VM (second session)
cd ~/gaussian-splatting
watch -n 5 'ls -lh output/*/point_cloud/iteration_*/  | tail -20'
```

**Option 2: TensorBoard (Advanced)**
```bash
# On VM
cd ~/gaussian-splatting/output/<your-model-uuid>
tensorboard --logdir=. --host=0.0.0.0 --port=6006
```

Then on local machine:
```powershell
gcloud compute ssh gaussian-splatting-vm --zone=us-west1-b -- -L 6006:localhost:6006
```
Open browser: http://localhost:6006

#### 4.3 Training Progress Indicators

You'll see output like:
```
Iteration 500/30000
L1 Loss: 0.045 | SSIM Loss: 0.123 | Total Loss: 0.168
Number of Gaussians: 145,234

Iteration 7000/30000  [CHECKPOINT SAVED]
L1 Loss: 0.012 | SSIM Loss: 0.034 | Total Loss: 0.046
Number of Gaussians: 523,891
```

**What to expect**:
- Iterations 0-500: Initialization
- Iterations 500-15,000: Active densification (Gaussian count increases)
- Iterations 15,000-30,000: Refinement (loss decreases)
- Total time on L4 GPU: ~20-30 minutes

---

### Phase 5: Rendering & Validation (Est. Time: 5-10 minutes)

#### 5.1 Find Your Model
```bash
cd ~/gaussian-splatting
ls output/
# You'll see a UUID folder like: 7a3b9c2d-1e4f-5a6b-8c9d-0e1f2a3b4c5d
```

#### 5.2 Render Images
```bash
# Replace <uuid> with your actual folder name
python render.py -m output/<uuid>

# This generates:
# output/<uuid>/train/ours_30000/renders/  - rendered images
# output/<uuid>/train/ours_30000/gt/       - ground truth images
# output/<uuid>/test/ours_30000/renders/   - test set renders
# output/<uuid>/test/ours_30000/gt/        - test ground truth
```

#### 5.3 Evaluate Metrics
```bash
python metrics.py -m output/<uuid>
```

**Expected Output**:
```
Train Set Results:
PSNR: 28.45 dB
SSIM: 0.912
LPIPS: 0.087

Test Set Results:
PSNR: 26.32 dB
SSIM: 0.891
LPIPS: 0.102
```

**Quality Benchmarks**:
- ✅ Good: PSNR > 25, SSIM > 0.85, LPIPS < 0.15
- ⚠️ Acceptable: PSNR > 22, SSIM > 0.80, LPIPS < 0.20
- ❌ Poor: PSNR < 20, SSIM < 0.75, LPIPS > 0.25

---

### Phase 6: Download Results for Validation (Est. Time: 5-15 minutes)

#### 6.1 Download Rendered Images
```powershell
# On local machine (PowerShell)
# Create local results folder
mkdir e:\Dev\gaussian-splatting\local-results

# Download all rendered images
gcloud compute scp --recurse gaussian-splatting-vm:~/gaussian-splatting/output/<uuid>/train/ours_30000/renders/ e:\Dev\gaussian-splatting\local-results\renders\ --zone=us-west1-b

# Download ground truth for comparison
gcloud compute scp --recurse gaussian-splatting-vm:~/gaussian-splatting/output/<uuid>/train/ours_30000/gt/ e:\Dev\gaussian-splatting\local-results\gt\ --zone=us-west1-b
```

#### 6.2 Download Complete Model (For Viewer)
```powershell
# Download entire model folder
gcloud compute scp --recurse gaussian-splatting-vm:~/gaussian-splatting/output/<uuid>/ e:\Dev\gaussian-splatting\local-models\<uuid>\ --zone=us-west1-b
```

#### 6.3 Visual Validation

**Quick Check**: Open rendered images in Windows Explorer
```powershell
explorer e:\Dev\gaussian-splatting\local-results\renders\
```

**Compare**: Look at renders vs. ground truth side-by-side
- Check for blurriness (indicates underfitting)
- Check for artifacts (indicates overfitting or data issues)
- Verify colors and lighting match

---

## Milestone Completion Criteria

### ✅ Success Indicators

1. **Training Completed Successfully**
   - [ ] Training reached 30,000 iterations without errors
   - [ ] Checkpoints saved at iterations 7000 and 30000
   - [ ] Loss decreased steadily (no NaN values)

2. **Quantitative Validation**
   - [ ] PSNR > 25 dB on test set
   - [ ] SSIM > 0.85 on test set
   - [ ] LPIPS < 0.15 on test set

3. **Qualitative Validation**
   - [ ] Rendered images are sharp and clear
   - [ ] Colors match ground truth
   - [ ] No major artifacts or holes in rendering
   - [ ] Test views (unseen during training) render correctly

4. **Deliverables**
   - [ ] Trained model saved in `output/<uuid>/`
   - [ ] Rendered images downloaded locally
   - [ ] Metrics calculated and documented
   - [ ] Baseline results ready for comparison with future improvements

---

## Troubleshooting Guide

### Issue: COLMAP fails with "insufficient matches"
**Solution**: 
- Ensure images have good overlap (70%+)
- Try different camera model: `python convert.py -s data/scene --camera SIMPLE_RADIAL`
- Check image quality with: `identify data/scene/input/*.jpg`

### Issue: Out of memory during training
**Solution**:
```bash
# Reduce resolution
python train.py -s data/scene --eval -r 4  # 25% resolution

# Or reduce image count (remove some from input/ folder)
```

### Issue: Training loss becomes NaN
**Solution**:
- Verify COLMAP reconstruction: `ls data/scene/sparse/0/`
- Check for corrupted images
- Reduce learning rate: `python train.py -s data/scene --position_lr_init 0.00008`

### Issue: Poor quality results (PSNR < 20)
**Possible causes**:
1. Insufficient training: Try `--iterations 50000`
2. Poor COLMAP reconstruction: Re-process with more/better images
3. Resolution too low: Train at full resolution (remove `-r` flag)
4. Moving objects in scene: Remove frames with motion

---

## Next Steps After Milestone 1

Once baseline training is complete and validated:

1. **Document Results**
   - Screenshot best/worst renders
   - Record training time, GPU usage, metrics
   - Note any issues or observations

2. **Prepare for Future Milestones**
   - Identify areas for improvement
   - Plan data augmentation strategies
   - Consider quality optimization techniques

3. **Cost Management**
   ```bash
   # Stop VM when not in use
   gcloud compute instances stop gaussian-splatting-vm --zone=us-west1-b
   
   # Restart when needed
   gcloud compute instances start gaussian-splatting-vm --zone=us-west1-b
   ```

---

## Quick Command Reference

```bash
# === VM Connection ===
gcloud compute ssh gaussian-splatting-vm --zone=us-west1-b

# === Activate Environment ===
conda activate gaussian_splatting

# === Training ===
python train.py -s data/your-scene --eval

# === Rendering ===
python render.py -m output/<uuid>

# === Metrics ===
python metrics.py -m output/<uuid>

# === File Transfer (from local machine) ===
# Upload
gcloud compute scp --recurse local-path/ gaussian-splatting-vm:~/destination/ --zone=us-west1-b

# Download
gcloud compute scp --recurse gaussian-splatting-vm:~/source/ local-path/ --zone=us-west1-b
```

---

## Estimated Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1 | GCP VM Setup | 30-45 min |
| 2 | Environment Configuration | 15-20 min |
| 3 | Data Collection & Preparation | 15-60 min |
| 4 | Baseline Training | 20-40 min |
| 5 | Rendering & Validation | 5-10 min |
| 6 | Download Results | 5-15 min |
| **Total** | **End-to-End** | **1.5-3 hours** |

**Note**: First-time setup takes longer. Subsequent training runs: ~30-45 minutes.

---

## Ready to Begin?

Start with Phase 1 and work through systematically. Each phase has verification steps - don't skip them!

**First Step**: Run this command to check your GCP setup:
```powershell
gcloud projects list
```

Then proceed with VM creation. Good luck! 🚀

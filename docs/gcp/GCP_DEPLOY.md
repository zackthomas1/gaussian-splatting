# Deploying 3D Gaussian Splatting on Google Cloud Platform (GCP)

This guide outlines the steps to set up a Virtual Machine (VM) on GCP to train and render Gaussian Splats.

## 1. Prerequisites
- A Google Cloud Platform account.
- A GCP Project with billing enabled.
- The `gcloud` CLI installed on your local machine (optional, but recommended).

## 2. Create a VM Instance

The project recommends **24 GB VRAM** for full quality training.
- **Recommended GPU**: NVIDIA L4 (24GB) or NVIDIA A10G (24GB).
- **Budget GPU**: NVIDIA T4 (16GB) - *Note: You may need to reduce batch size or scene size.*

### Steps:
1. Go to the **Compute Engine** > **VM Instances** page in the GCP Console.
2. Click **Create Instance**.
3. **Name**: `gaussian-splatting-vm` (or similar).
4. **Region**: Select `us-west1` (Oregon).
5. **Machine Configuration**:
   - **Series**: `G2` (for L4 GPUs).
   - **Machine type**: `g2-standard-4` (4 vCPUs, 16GB RAM).
6. **GPU**:
   - The G2 series includes the NVIDIA L4 GPU by default.
7. **Boot Disk**:
   - Scroll down to the **Boot disk** section and click **Change**.
   - In the pop-up window:
     - **Operating System**: Click the dropdown and select `Deep Learning on Linux`.
     - **Version**: Look for a version that says **CUDA 11.8**.
       - Example: `Deep Learning VM with CUDA 11.8 M115` (or similar).
       - *Critical*: Do **not** select a CUDA 12 image. The project dependencies are built for CUDA 11.
     - **Boot disk type**: Select `Balanced persistent disk` or `SSD persistent disk`.
     - **Size (GB)**: Change to **100** or more.
   - Click **Select** at the bottom of the window.
8. **Firewall**: Check "Allow HTTP traffic" and "Allow HTTPS traffic" (useful if you run a web viewer later, though mostly we use SSH).
9. Click **Create**.

### Troubleshooting: GPU Quota Limits
If you receive an error like `The GPUS-ALL-REGIONS-per-project quota maximum has been exceeded`, it means your project has a default GPU quota of 0 (common for new accounts).

**To fix this:**
1. Go to **IAM & Admin** > **Quotas** in the GCP Console.
2. In the filter box, type `gpus_all_regions` and select the metric `compute.googleapis.com/gpus_all_regions`.
3. Select the row for "GPUs (all regions)" and click **Edit Quotas**.
4. Enter the new limit (e.g., `1`) and a justification (e.g., "Training 3D Gaussian Splatting models").
5. Click **Submit Request**.
   - *Note: Approval is usually automatic but can take up to 24-48 hours for new accounts.*

## 3. Connect to the VM

Once the VM is running, click the **SSH** button in the console, or use the gcloud command:
```bash
gcloud compute ssh zackt@gaussian-splatting-vm
```

## 4. Setup the Environment

On the VM terminal, follow these steps:

### 4.1. Clone the Repository
First, ensure git is installed (it may be missing on some minimal VM images):
```bash
sudo apt-get update
sudo apt-get install git -y
```

Then clone the repository:
```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
```

### 4.2. Configure Conda Environment
The Deep Learning VM comes with Conda. We will create the environment defined in the repo.

**Important for G2 (L4 GPU) Users**: The default `environment.yml` uses PyTorch 1.12 and CUDA 11.6, which does **not** support the L4 GPU (Ada Lovelace architecture). You must use PyTorch 2.0+ and CUDA 11.8.

You have two options: manually creating the environment (Recommended) or modifying the `environment.yml` file.

#### Option 1: Manual Setup (Recommended for G2/L4)
This is the most reliable method. We create an empty environment and install the exact compatible versions using `pip`.

1. Create a new environment with Python 3.8:
   ```bash
   conda create -n gaussian_splatting python=3.8 -y
   conda activate gaussian_splatting
   ```
2. Install PyTorch 2.0 with CUDA 11.8 support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Install other dependencies:
   ```bash
   pip install plyfile tqdm opencv-python joblib
   ```
4. Install the submodules (ensure you are in the `gaussian-splatting` root directory):
   ```bash
   pip install ./submodules/diff-gaussian-rasterization
   pip install ./submodules/simple-knn
   pip install ./submodules/fused-ssim
   ```

#### Option 2: Modifying `environment.yml`
Yes, you can modify the file directly in the shell using a text editor like `nano`.

1. Open the file:
   ```bash
   nano environment.yml
   ```
2. Make the following changes:
   - Change `python=3.7.13` to `python=3.8`
   - Change `cudatoolkit=11.6` to `pytorch-cuda=11.8`
   - Change `pytorch=1.12.1` to `pytorch=2.0.0`
   - Change `torchvision=0.13.1` to `torchvision`
3. Save (Ctrl+O, Enter) and Exit (Ctrl+X).
4. Create the environment:
   ```bash
   conda env create --file environment.yml
   ```
*Note: This method can sometimes lead to long "Solving environment" times or conflicts. Option 1 is usually faster.*

### Troubleshooting: "conda: command not found"
If you see this error, it means Conda is not in your PATH or you might have selected a standard OS image instead of the Deep Learning VM.

**Fix 1: Add Conda to PATH (Deep Learning VM)**
On Google Deep Learning VMs, Conda is installed but sometimes not in the user's path. Run:
```bash
export PATH=/opt/conda/bin:$PATH
conda init
source ~/.bashrc
```

**Fix 2: Install Miniconda (If Fix 1 fails)**
If `/opt/conda` does not exist, you likely created a standard Ubuntu VM. Install Miniconda manually:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
export PATH=~/miniconda3/bin:$PATH
conda init
source ~/.bashrc
```

## 5. Uploading Data

You need to get your COLMAP or NeRF Synthetic dataset onto the VM.

### Option A: Upload from Local Machine (using gcloud)
```bash
# Run this on your LOCAL machine
gcloud compute scp --recurse ./data/<input dataset> zackt@gaussian-splatting-vm:~/gaussian-splatting/data/
```

### Option B: Download directly to VM
If your data is hosted online (e.g., the sample datasets):
```bash
# On the VM
mkdir data
cd data
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip tandt_db.zip
```

## 6. Running Training

Now you can run the training script.

```bash
# Assuming you are in ~/gaussian-splatting and data is in ./data/tandt/train
python train.py -s data/tandt/train
```

The output models will be saved in the `output/` directory.

## 7. Rendering

After training is complete, you can generate renderings (images) from your trained model.

```bash
# 1. List the output directory to find your model folder (it has a random name)
ls output/

# 2. Run the render script
# Replace <YOUR_MODEL_DIR> with the actual folder name from the previous step
python render.py -m output/<YOUR_MODEL_DIR>
```

This will create `train` and `test` directories inside your model folder containing the rendered images.

## 8. Visualization & Results

Since the VM is headless (no monitor), you cannot run the Real-Time Viewer (`SIBR_gaussianViewer_app`) directly on the VM GUI.

### Option A: Download Trained Model (Recommended)
Download the trained model folder to your local machine and view it locally.

```bash
# Run on LOCAL machine
gcloud compute scp --recurse zackt@gaussian-splatting-vm:gaussian-splatting/output/your-model-id ./local-models/
```
  
### Option B: Remote Viewer
You can use the Network Viewer to monitor training.
1. On VM, start training with a known IP/Port (or default).
2. On Local machine, use SSH tunneling to forward the port (default 6009).
   ```bash
   gcloud compute ssh zackt@gaussian-splatting-vm -- -L 6009:localhost:6009
   ```
3. Run `SIBR_remoteGaussian_app` locally.

### Option C: Download Rendered Images
If you ran `render.py` and just want to view the resulting image frames:

```bash
# Replace <YOUR_MODEL_DIR> with your actual model folder name
gcloud compute scp --recurse zackt@gaussian-splatting-vm:~/gaussian-splatting/output/<YOUR_MODEL_DIR>/train ./local-renders/
```
You can then open the `./local-renders/` folder on your computer to view the standard image files (e.g., .png or .jpg).

## 9. Clean Up
**Don't forget to stop or delete your VM when you are done to avoid high costs!**
- **Stop**: Pauses billing for compute, but you still pay for disk storage.
- **Delete**: Removes everything.

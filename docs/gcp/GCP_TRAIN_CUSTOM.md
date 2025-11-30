# Training on Custom Data (Video/Images) on GCP

This guide outlines the process of training a Gaussian Splatting model using your own custom video or image dataset on a Google Cloud Platform (GCP) VM.

## 1. Connect to your VM

Open your local terminal (PowerShell or Command Prompt) and connect to your GCP VM:

```bash
gcloud compute ssh gaussian-splatting-vm
```

## 2. Install Dependencies on the VM

Once inside the VM, you need to install **COLMAP** (for Structure-from-Motion) and **FFmpeg** (for video processing). Run these commands:

```bash
sudo apt-get update
sudo apt-get install -y colmap ffmpeg
```
1. Check FFmpeg
Run this command to see the version information:
```bash
ffmpeg -version
```
You should see output starting with ffmpeg version ... followed by configuration details.


2. Check COLMAP
Run this command to see the help menu (which confirms it's executable):
```bash
colmap help
```
You should see a list
of available COLMAP commands (like gui, feature_extractor, mapper, etc.).

## 3. Prepare and Upload Your Data

You need to transfer your video or images from your local machine to the VM.

### Option A: If you have a Video file
1.  On your **local machine**, navigate to the folder containing your video.
2.  Upload it to the VM (replace `my_video.mp4` with your actual file name):
    ```bash
    gcloud compute scp my_video.mp4 gaussian-splatting-vm:~/gaussian-splatting/
    ```

### Option B: If you have a sequence of Images
1.  Ensure your images are in a folder on your local machine.
2.  Upload the folder:
    ```bash
    gcloud compute scp --recurse my_images_folder gaussian-splatting-vm:~/gaussian-splatting/
    ```

## 4. Process the Data on the VM

Back in your **VM terminal**, follow these steps to prepare the dataset for training.

### 4.1. Create the dataset structure
Create a folder for your new scene (e.g., `myscene`) and an `input` subfolder where the images must reside.

```bash
cd ~/gaussian-splatting
mkdir -p data/myscene/input
```

### 4.2. Convert Video to Images (Skip if you uploaded images)
If you uploaded a video, extract frames into the `input` folder.
*   `-fps 2`: Extracts 2 frames per second. Adjust this depending on your video speed and length (aim for 100-300 images total for a good balance).
*   `-qscale:v 1`: Maintains high JPEG quality.

```bash
# Replace 'my_video.mp4' with your uploaded filename
ffmpeg -i my_video.mp4 -qscale:v 1 -qmin 1 -vf "fps=2" data/myscene/input/%04d.jpg
```

*If you uploaded a folder of images instead, move or copy them into `data/myscene/input/`.*

### 4.3. Run COLMAP (SfM)
Use the provided `convert.py` script to analyze the images, calculate camera positions, and generate the sparse point cloud required for training.

```bash
python convert.py -s data/myscene
```
*Note: This process can take minutes to hours depending on the number of images.*

**Troubleshooting:** If you encounter errors related to "display" or "qt", try running it with `xvfb` (virtual display):
```bash
sudo apt-get install -y xvfb
xvfb-run -a python convert.py -s data/myscene
```

## 5. Train the Model

Once `convert.py` finishes successfully, you will see a new `sparse` folder inside `data/myscene`. You are now ready to train.

```bash
python train.py -s data/myscene
```

## 6. Visualize Results

After training finishes:

1.  **Check the output folder:**
    ```bash
    ls output/
    ```
    Find the new random folder name corresponding to your custom training.

2.  **Download the model to your local machine:**
    (Run this on your **local** terminal, not the VM)
    ```bash
    # Replace <YOUR_NEW_MODEL_FOLDER> with the actual folder name
    gcloud compute scp --recurse gaussian-splatting-vm:~/gaussian-splatting/output/<YOUR_NEW_MODEL_FOLDER> ./local-models/
    ```

3.  **Run the viewer locally:**
    ```bash
    ./SIBR_gaussianViewer_app -m ./local-models/<YOUR_NEW_MODEL_FOLDER>
    ```

@echo off
REM Quick setup and test script for OpenCV calibration pipeline (Windows)
REM Run this after cloning/updating the repository

echo ========================================
echo OpenCV Calibration Pipeline - Setup
echo ========================================
echo.

REM Step 1: Install dependencies
echo Step 1: Installing dependencies...
python setup_opencv_pipeline.py

if %errorlevel% neq 0 (
    echo Setup failed! Please check errors above.
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo.
echo 1. Test on silverlake dataset:
echo    python opencv\opencv_convert.py -s data\silverlake
echo.
echo 2. Or prepare your own dataset:
echo    mkdir data\my_scene\input
echo    REM Copy images to data\my_scene\input\
echo    python opencv\opencv_convert.py -s data\my_scene
echo.
echo 3. Train Gaussian Splatting:
echo    python train.py -s data\my_scene -m output\my_scene
echo.
echo For more info, see:
echo   * OPENCV_IMPLEMENTATION.md (start here!)
echo   * docs\OPENCV_QUICKSTART.md
echo   * docs\OPENCV_CALIBRATION_README.md
echo.

# Sentinel-2 Image Compression with Principal Component Analysis (PCA)

This repository contains a Python implementation of an image compression project focused on Sentinel-2 satellite images. The project utilizes Principal Component Analysis (PCA) to reduce the dimensionality of multi-band satellite imagery while preserving critical information.

## Features:
- **Data Loading:** Efficient loading of Sentinel-2 bands using the rasterio library.
- **Interactive Crop Selection:** User-friendly interface for interactively selecting a crop area from the satellite image.
- **PCA Implementation:** Implementation of PCA to perform image compression and reduce dimensionality.
- **Thresholding:** Application of a threshold to the PCA results for further compression and noise reduction.
- **Reconstruction:** Reconstruction of the compressed image to evaluate the effectiveness of the compression technique.
- **Error Analysis:** Analysis of reconstruction errors for different numbers of principal components.
- **Information Loss Visualization:** Visualization of information loss against the number of principal components.

## How to Use:
1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Run the `main.py` script to execute the image compression and analysis.

Feel free to explore and modify the code to suit your specific needs. Contributions and suggestions are welcome!

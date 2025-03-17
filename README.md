# Image Segmentation: K-means vs. Ratio-Cut Clustering

This repository implements and compares two image segmentation techniques:
1. K-means clustering
2. Ratio-Cut based clustering

The implementation allows you to segment images using both techniques and compare their performance visually.

## Features

- Image segmentation using K-means clustering
- Image segmentation using Ratio-Cut clustering
- Comparison of both techniques with different cluster numbers (3 and 6)
- Modular code design for easy use with different images
- Visualization of segmentation results

## Requirements

- Python 3.6+
- NumPy
- OpenCV
- Matplotlib
- scikit-learn

Install all requirements using:
```
git clone https://github.com/mitesh-kr/Image-Segmentation-K-means-vs.-Ratio-Cut-Clustering.git

cd Image-Segmentation-K-means-vs.-Ratio-Cut-Clustering

pip install -r requirements.txt
```

## Dataset

The original assignment uses images from: https://bit.ly/cvasg2img

You'll need to download these images and place them in an input folder before running the script.

## Usage

Basic usage:
```
python image_segmentation.py --input_folder ./images
```

Advanced usage with custom parameters:
```
python image_segmentation.py --input_folder ./images --output_folder ./results --k_values 3 6 --sigma 1.0 --lambda 0.001
```

### Parameters

- `--input_folder`: Path to folder containing input images (required)
- `--output_folder`: Path to folder for output images (default: 'output')
- `--k_values`: Number of clusters to use (default: [3, 6])
- `--sigma`: Sigma parameter for Ratio-Cut (default: 1.0)
- `--lambda`: Lambda parameter for Ratio-Cut (default: 0.001)

## Implementation Details

### Image Loading and Preprocessing
- Images are resized to 64x64 pixels to reduce computational complexity
- Images are converted from BGR to RGB color space
- Pixel values are normalized to the range [0, 1]

### K-means Clustering
A basic clustering algorithm that groups pixel values based on color similarity:
1. Reshape the image to a 2D array of pixels
2. Apply K-means clustering on the pixel values
3. Reshape the cluster labels back to the image dimensions

### Ratio-Cut Clustering
A spectral clustering method that considers both color similarity and spatial proximity:
1. Compute pairwise distances between pixels (both color and spatial)
2. Create an adjacency matrix using a Gaussian similarity function
3. Compute the Laplacian matrix
4. Find the eigenvectors corresponding to the smallest eigenvalues
5. Apply K-means clustering on these eigenvectors
6. Reshape the cluster labels back to the image dimensions

## Results

The script generates comparison images showing:
- Original image
- K-means segmentation result
- Ratio-Cut segmentation result

These images are saved in the specified output folder.

## Notes on Performance

- The Ratio-Cut algorithm is computationally intensive for larger images due to the pairwise distance calculations
- Using 64x64 resolution allows for reasonable computation time
- The lambda parameter controls the influence of spatial distance
- Sigma controls the scaling of the similarity function

## License

This project is for educational purposes and is part of an academic assignment.

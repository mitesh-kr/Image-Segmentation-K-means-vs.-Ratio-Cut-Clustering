import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse

def load_images(folder_path):
    """
    Load and preprocess images from the specified folder.
    Resize images to 64x64 and convert to RGB.
    """
    image_files = os.listdir(folder_path)
    images = []
    filenames = []
    
    for file in image_files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (64, 64))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype(np.float32) / 255.0
                images.append(image)
                filenames.append(file)
    
    return images, filenames

def kmeans_clustering(image, num_clusters):
    """
    Perform K-means clustering on the input image.
    
    Args:
        image: Input image (H, W, 3)
        num_clusters: Number of clusters for segmentation
        
    Returns:
        Segmented image with cluster labels
    """
    height, width = image.shape[:2]
    flatten_image = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=19)
    labels = kmeans.fit_predict(flatten_image)
    segmented_image = labels.reshape((height, width))
    return segmented_image

def ratio_cut_clustering(image, num_clusters, sigma, lamda):
    """
    Perform Ratio-Cut based clustering on the input image.
    
    Args:
        image: Input image (H, W, 3)
        num_clusters: Number of clusters for segmentation
        sigma: Parameter for Gaussian similarity function
        lamda: Weight for spatial distance
        
    Returns:
        Segmented image with cluster labels
    """
    height, width = image.shape[:2]
    flattened_image = image.reshape((-1, 3))
    
    # Calculate pairwise intensity distances
    print("Computing intensity distances...")
    intensity_distance = np.zeros((height * width, height * width))
    for i in range(height * width):
        for j in range(i + 1, height * width):
            distance = np.sqrt(np.sum((flattened_image[i] - flattened_image[j]) ** 2))
            intensity_distance[i, j] = distance
            intensity_distance[j, i] = distance
    
    # Calculate pairwise spatial distances
    print("Computing spatial distances...")
    spatial_distance = np.zeros((height * width, height * width))
    for i in range(height * width):
        y_i, x_i = i // width, i % width
        for j in range(i + 1, height * width):
            y_j, x_j = j // width, j % width
            distance = np.sqrt((y_i - y_j)**2 + (x_i - x_j)**2)
            spatial_distance[i, j] = distance
            spatial_distance[j, i] = distance
    
    # Compute total pairwise distances
    total_pairwise_distance = intensity_distance + lamda * spatial_distance
    
    # Compute adjacency matrix using Gaussian similarity
    print("Computing adjacency matrix...")
    adjacency_matrix = np.exp(-total_pairwise_distance / (2 * sigma ** 2))
    
    # Normalize adjacency matrix
    adjacency_matrix = adjacency_matrix / np.sum(adjacency_matrix, axis=1, keepdims=True)
    
    # Compute degree matrix
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    
    # Compute Laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix
    
    # Compute eigenvalues and eigenvectors
    print("Computing eigenvalues and eigenvectors...")
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    
    # Sort eigenvectors by eigenvalues
    sorted_indices = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, sorted_indices[:num_clusters]]
    
    # Apply K-means on eigenvectors
    print("Applying K-means on eigenvectors...")
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=19)
    labels = kmeans.fit_predict(eigenvectors)
    segmented_image = labels.reshape((height, width))
    
    return segmented_image

def display_side_by_side(original_image, image1, image2, k, output_folder=None, filename=None,
                        title_original='Original Image', title1='K-means Segmentation', title2='Ratio Cut Segmentation'):
    """
    Display or save comparison images side by side.
    
    Args:
        original_image: Original input image
        image1: First segmented image (K-means)
        image2: Second segmented image (Ratio-Cut)
        k: Number of clusters used
        output_folder: If specified, save image to this folder
        filename: Base filename to use when saving
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title(title_original)
    axes[0].axis('off')
    
    axes[1].imshow(image1, cmap='viridis')
    axes[1].set_title(f'{title1} (k={k})')
    axes[1].axis('off')
    
    axes[2].imshow(image2, cmap='viridis')
    axes[2].set_title(f'{title2} (k={k})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_folder and filename:
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"{filename}_k{k}.png")
        plt.savefig(output_path)
        print(f"Saved comparison to {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Image Segmentation using K-means and Ratio-Cut Clustering')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to folder containing input images')
    parser.add_argument('--output_folder', type=str, default='output', help='Path to folder for output images')
    parser.add_argument('--k_values', type=int, nargs='+', default=[3, 6], help='Number of clusters to use')
    parser.add_argument('--sigma', type=float, default=1.0, help='Sigma parameter for Ratio-Cut')
    parser.add_argument('--lambda', type=float, dest='lamda', default=0.001, help='Lambda parameter for Ratio-Cut')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Load images
    images, filenames = load_images(args.input_folder)
    
    if not images:
        print("No valid images found in the specified folder.")
        return
    
    # Process each image
    for i, (image, filename) in enumerate(zip(images, filenames)):
        print(f"Processing image {i+1}/{len(images)}: {filename}")
        
        for k in args.k_values:
            print(f"  Applying clustering with k={k}")
            
            # Apply K-means clustering
            image_kmeans = kmeans_clustering(image=image, num_clusters=k)
            
            # Apply Ratio-Cut clustering
            print(f"  Applying Ratio-Cut clustering (this may take a while)...")
            image_ratio_cut = ratio_cut_clustering(
                image=image, 
                num_clusters=k, 
                sigma=args.sigma, 
                lamda=args.lamda if k == 3 else 0  # Use lambda for k=3, 0 for k=6 (as in original code)
            )
            
            # Display and save results
            base_filename = os.path.splitext(filename)[0]
            display_side_by_side(
                original_image=image,
                image1=image_kmeans,
                image2=image_ratio_cut,
                k=k,
                output_folder=args.output_folder,
                filename=base_filename
            )
            
            print(f"  Completed segmentation with k={k}")
        
        print(f"Completed processing for {filename}")

if __name__ == "__main__":
    main()

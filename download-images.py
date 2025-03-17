import os
import requests
import shutil
from urllib.parse import urlparse

def download_file(url, output_folder):
    """Download a file from a URL to the specified folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get the filename from the URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    if not filename:
        filename = 'image.jpg'  # Default name if unable to extract from URL
    
    output_path = os.path.join(output_folder, filename)
    
    # Download the file
    print(f"Downloading {url} to {output_path}...")
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
        print(f"Download complete: {filename}")
        return output_path
    else:
        print(f"Failed to download {url}, status code: {response.status_code}")
        return None

def main():
    # Create images directory
    image_folder = os.path.join(os.getcwd(), 'images')
    os.makedirs(image_folder, exist_ok=True)
    
    # URL from the assignment
    base_url = "https://bit.ly/cvasg2img"
    
    print("This script will attempt to download the images from the provided URL.")
    print("Note: If the URL is a shortlink to a page rather than direct images,")
    print("you may need to manually download the images from the website.")
    
    try:
        # Try to download directly from the URL
        download_file(base_url, image_folder)
        print("\nIf the downloaded file isn't an image, please visit the URL manually:")
        print(base_url)
        print("Download the images and place them in the 'images' folder.")
    except Exception as e:
        print(f"\nError downloading: {e}")
        print("\nPlease manually visit the URL and download the images:")
        print(base_url)
        print("Place the downloaded images in the 'images' folder.")

if __name__ == "__main__":
    main()

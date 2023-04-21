import os
import random
import requests
import base64
import io
from PIL import Image


def upload_image(image_path: str, api_key: str, project_id: str, split: str, batch_name: str):
    """
    Upload a single image to the Roboflow Upload API with the given parameters.
​
    Args:
        image_path (str): Path to the image file.
        api_key (str): Roboflow API key.
        project_id (str): Roboflow project ID.
        split (str): Dataset split, can be 'train', 'valid', or 'test'.
        batch_name (str): Batch name for the uploaded images.
    Returns:
        dict: JSON response from the Roboflow API.
    """
    image = Image.open(image_path).convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, quality=90, format="JPEG")

    img_str = base64.b64encode(buffered.getvalue())
    img_str = img_str.decode("ascii")

    upload_url = "".join([
        f"https://api.roboflow.com/dataset/{project_id}/upload",
        f"?api_key={api_key}",
        f"&name={os.path.basename(image_path)}",
        f"&split={split}",
        f"&batch={batch_name}"
    ])

    r = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    })

    return r.json()

def get_image_paths(directory: str):
    """
    Get a list of image file paths from a directory.

    Args:
        directory (str): Path to the directory containing images.
    Returns:
        list: A list of image file paths.
    """
    image_extensions = {'.jpeg', '.jpg', '.png'}
    image_paths = []

    for file in os.listdir(directory):
        file_extension = os.path.splitext(file)[1].lower()
        if file_extension in image_extensions:
            image_paths.append(os.path.join(directory, file))

    return image_paths

def upload_images(directory: str, api_key: str, project_id: str, split: str, percentage: int, batch_name: str):
    """
    Upload a specified percentage of images from a directory to a given dataset split.

    Args:
        directory (str): Path to the directory containing images.
        api_key (str): Roboflow API key.
        project_id (str): Roboflow project ID.
        split (str): Dataset split, can be 'train', 'valid', or 'test'.
        percentage (int): The percentage of images to upload (1-100).
        batch_name (str): Batch name for the uploaded images.
    """
    image_paths = get_image_paths(directory)
    num_images_to_upload = int(len(image_paths) * percentage / 100)
    print(f"Uploading {num_images_to_upload} images to the {split} split...")
    sampled_image_paths = random.sample(image_paths, num_images_to_upload)

    for image_path in sampled_image_paths:
        result = upload_image(image_path, api_key, project_id, split, batch_name)
        print(result)

    if __name__ == '__main__':
        # Example usage:
        image_directory = 'path/to/your/image/directory'
        api_key = 'YOUR_API_KEY'
        project_id = 'YOUR_PROJECT_ID'
        split = 'train'  # can be 'train', 'valid', or 'test'
        percentage = 50  # value between 1 and 100
        batch_name = 'YOUR_BATCH_NAME'
    
    print("Uploading images to Roboflow...This may take a few moments.\n")
    print(f"Uploading from directory: {image_directory} | Project ID: {project_id} | Dataset Split for Upload: {split}")
    print(f"Percent of images in the directory to be uploaded to the {split} split: {percentage} | Upload Batch Name: {batch_name}")
    upload_images(image_directory, api_key, project_id, split, percentage, batch_name)

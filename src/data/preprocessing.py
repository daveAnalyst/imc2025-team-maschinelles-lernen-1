from PIL import Image, ExifTags # ExifTags for potential orientation correction
import os
import numpy as np # If you plan to add functions returning NumPy arrays

def load_image_pil(image_path, ensure_rgb=True):
    """
    Loads an image using Pillow.
    Optionally ensures the image is converted to RGB.

    Args:
        image_path (str): Path to the image file.
        ensure_rgb (bool): If True, converts image to RGB format.

    Returns:
        PIL.Image.Image or None: The loaded image object, or None if loading fails.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    try:
        img = Image.open(image_path)
        
        # Optional: Correct for EXIF orientation tag
        # This can be important as some images might be saved with rotation flags
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = img._getexif()
            if exif is not None and orientation in exif:
                if exif[orientation] == 3:
                    img = img.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    img = img.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    img = img.rotate(90, expand=True)
        except Exception as e:
            # print(f"Warning: Could not process EXIF orientation for {image_path}: {e}")
            pass # Continue even if EXIF processing fails

        if ensure_rgb:
            if img.mode == 'RGBA':
                # Create a white background image
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3]) # Paste using alpha channel as mask
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def resize_image_maintain_aspect_ratio(image_pil, target_max_dimension):
    """
    Resizes a PIL image so its longest side matches target_max_dimension,
    maintaining aspect ratio.

    Args:
        image_pil (PIL.Image.Image): The input PIL image.
        target_max_dimension (int): The target size for the longest dimension.

    Returns:
        PIL.Image.Image: The resized PIL image.
    """
    if image_pil is None:
        return None
        
    original_width, original_height = image_pil.size

    if original_width == 0 or original_height == 0:
        print(f"Warning: Image has zero dimension: {original_width}x{original_height}")
        return image_pil # Or handle error appropriately

    if max(original_width, original_height) <= target_max_dimension:
        return image_pil # No need to resize if already smaller or equal

    if original_width > original_height:
        # Width is the longest side
        new_width = target_max_dimension
        new_height = int(original_height * (new_width / original_width))
    else:
        # Height is the longest side (or they are equal)
        new_height = target_max_dimension
        new_width = int(original_width * (new_height / original_height))
    
    try:
        resized_img = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS) # High quality downsampling
        return resized_img
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image_pil # Return original on error

# Example of a padding function (you might or might not need this directly
# if model transforms handle it, but good to have the concept)
def pad_image_to_square(image_pil, target_size, padding_color=(128, 128, 128)):
    """
    Pads a PIL image to a target square size.
    The image is centered, and padding is added to the shorter dimension.
    """
    if image_pil is None:
        return None

    original_width, original_height = image_pil.size
    
    # Create a new image with the padding color
    padded_img = Image.new("RGB", (target_size, target_size), padding_color)
    
    # Calculate position to paste the original image (centered)
    paste_x = (target_size - original_width) // 2
    paste_y = (target_size - original_height) // 2
    
    padded_img.paste(image_pil, (paste_x, paste_y))
    return padded_img


# --- Test block ---
if __name__ == '__main__':
    # NOTE: Adjust this path to point to an actual image in your project for testing
    # This path assumes your script is in src/data/ and data is ../../data/
    # If running from project root: python src/data/preprocessing.py
    # then sample_img_path = './data/image-matching-challenge-2025/train/SOME_DATASET/SOME_IMAGE.png'
    
    # Construct a path relative to this script file for testing, assuming a certain structure
    # This is just for the __main__ test block. When imported, paths will be absolute.
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..')) # Goes up two levels from src/data/
    
    # *** Replace with a valid dataset and image name from your data ***
    # This sample path is a placeholder.
    sample_dataset = "amy_gardens" # Example, pick one of yours
    sample_image_name = "peach_0000.png"    # Example, pick one of yours
    
    sample_img_path = os.path.join(project_root, 'data', 'image-matching-challenge-2025', 'train', sample_dataset, sample_image_name)
    print(f"Test image path: {sample_img_path}")

    if os.path.exists(sample_img_path):
        print(f"\n--- Testing load_image_pil ---")
        pil_img = load_image_pil(sample_img_path)
        if pil_img:
            print(f"Loaded image. Original size: {pil_img.size}, Mode: {pil_img.mode}")
            # pil_img.show() # Opens in default image viewer

        print(f"\n--- Testing resize_image_maintain_aspect_ratio ---")
        if pil_img:
            target_dim = 512
            resized_img = resize_image_maintain_aspect_ratio(pil_img, target_max_dimension=target_dim)
            if resized_img:
                print(f"Resized image (max_dim {target_dim}): {resized_img.size}")
                # resized_img.show()

                print(f"\n--- Testing pad_image_to_square (example) ---")
                # Example: if a model needed 512x512 input after resizing longest side to 512
                # This isn't always the case, ViT transforms often do their own padding/cropping
                if resized_img.size[0] != target_dim or resized_img.size[1] != target_dim:
                    padded_square_img = pad_image_to_square(resized_img, target_size=target_dim)
                    if padded_square_img:
                        print(f"Padded to square ({target_dim}x{target_dim}): {padded_square_img.size}")
                        # padded_square_img.show()
                else:
                    print(f"Image was already square after resize, or target_dim matched both sides.")

    else:
        print(f"Test image not found at {sample_img_path}. Please update the path in the __main__ block of preprocessing.py for testing.")
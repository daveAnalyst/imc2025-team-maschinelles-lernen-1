import torch
import timm # Keep for transforms if desired, or use DINOv2's own if specific
from PIL import Image
from torchvision import transforms # For standard transforms
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
import argparse # For command-line arguments

class DinoV2EmbeddingExtractor:
    def __init__(self, model_size='s', device=None): # e.g., 's' for vits14, 'b' for vitb14
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device} for DINOv2 Extractor")

        # Construct model name for PyTorch Hub
        dinov2_model_hub_name = f'dinov2_vit{model_size}14' # e.g., dinov2_vits14

        try:
            # Using torch.hub.load to get official DINOv2 models
            self.model = torch.hub.load('facebookresearch/dinov2', dinov2_model_hub_name).to(self.device)
            self.model.eval()
            print(f"Successfully loaded {dinov2_model_hub_name} from PyTorch Hub.")
        except Exception as e:
            print(f"Error loading {dinov2_model_hub_name} from PyTorch Hub: {e}")
            raise RuntimeError(f"Failed to load DINOv2 model: {dinov2_model_hub_name}. Check internet or model name.")

        # Standard DINOv2/ViT transforms - important for consistency
        # These are typical values used with ViTs pre-trained on ImageNet, also suitable for DINOv2
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), # DINOv2 often uses 224x224 or patch-compatible sizes like 518x518
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # print(f"Image transform: {self.transform}") # Can be verbose

    def get_embedding(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device) # Add batch dimension
            
            with torch.no_grad():
                embedding = self.model(img_tensor) # For FB DINOv2 models, this is usually the CLS token embedding
            return embedding.squeeze().cpu().numpy() # Remove batch dim, move to CPU, convert to numpy
        except Exception as e:
            # For a script, print full error for the specific image, but don't stop the whole batch
            # print(f"Error processing image {os.path.basename(image_path)}: {e}") 
            return None

def get_full_image_path(dataset_name, image_filename, base_image_dir):
    """Constructs the full path to an image from CSV row data."""
    if pd.notna(dataset_name) and pd.notna(image_filename):
        # Ensure components are strings for os.path.join
        return os.path.join(base_image_dir, str(dataset_name), str(image_filename))
    return None

def extract_features_and_save(image_list_csv_path, output_npz_path, 
                              base_image_dir, dataset_col='dataset', 
                              image_col='image', model_size='s'):
    """
    Main function to extract DINOv2 features for all images listed in a CSV 
    and save them to a single NPZ file.
    """
    print(f"Loading image list from: {image_list_csv_path}")
    try:
        df = pd.read_csv(image_list_csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {image_list_csv_path}")
        return

    print(f"Initializing DINOv2 extractor (model size: {model_size})...")
    try:
        extractor = DinoV2EmbeddingExtractor(model_size=model_size)
    except Exception as e:
        print(f"Failed to initialize DINOv2 extractor: {e}")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(output_npz_path)
    if output_dir and not os.path.exists(output_dir): # Create only if output_dir is not empty (i.e., not current dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")


    all_embeddings = {}
    processed_count = 0
    error_count = 0
    skipped_path_count = 0

    print(f"Starting DINOv2 embedding extraction for {len(df)} entries...")
    # Iterating through the DataFrame rows
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting Features"):
        dataset_name = row.get(dataset_col) # Use .get() for safety if col might be missing
        image_filename = row.get(image_col)
        
        img_path = get_full_image_path(dataset_name, image_filename, base_image_dir)

        if img_path is None or not os.path.exists(img_path):
            # print(f"Skipping invalid or non-existent path for row {index}: dataset='{dataset_name}', image='{image_filename}'")
            skipped_path_count +=1
            continue # Skip to the next row

        embedding = extractor.get_embedding(img_path)
        
        # Create a unique key for the dictionary
        unique_image_id = f"{dataset_name}__{image_filename}" 

        if embedding is not None:
            all_embeddings[unique_image_id] = embedding
            processed_count += 1
        else:
            print(f"Failed to extract embedding for {unique_image_id} (path: {img_path})") # Log failure with path
            error_count += 1
            
    print(f"\n--- Feature Extraction Summary ---")
    print(f"Successfully extracted embeddings for {processed_count} images.")
    if skipped_path_count > 0:
        print(f"Skipped {skipped_path_count} entries due to invalid/missing paths.")
    if error_count > 0:
        print(f"Encountered errors for {error_count} images during extraction (see logs above for details).")

    if all_embeddings:
        print(f"Saving {len(all_embeddings)} embeddings to {output_npz_path}...")
        try:
            np.savez_compressed(output_npz_path, **all_embeddings)
            print("Embeddings saved successfully.")
        except Exception as e:
            print(f"Error saving embeddings to {output_npz_path}: {e}")
    else:
        print("No embeddings were extracted to save.")

# This block allows the script to be run from the command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract DINOv2 global embeddings for images.")
    parser.add_argument("image_list_csv", 
                        help="Path to CSV file (e.g., train_labels.csv) containing image information.")
    parser.add_argument("output_npz", 
                        help="Path to save the output NPZ file with embeddings (e.g., ./data/features/train_embeddings.npz).")
    parser.add_argument("--base_image_dir", default="./data/train", 
                        help="Base directory where dataset subfolders with images are located (default: ./data/train).")
    parser.add_argument("--dataset_col", default="dataset", 
                        help="Column name for dataset identifier in the CSV (default: dataset).")
    parser.add_argument("--image_col", default="image", 
                        help="Column name for image filename in the CSV (default: image).")
    parser.add_argument("--model_size", default="s", choices=['s', 'b', 'l', 'g'],
                        help="Size of the DINOv2 ViT model to use: s (small), b (base), l (large), g (giant) (default: s). Corresponds to vits14, vitb14, etc.")
    
    args = parser.parse_args()

    extract_features_and_save(
        image_list_csv_path=args.image_list_csv,
        output_npz_path=args.output_npz,
        base_image_dir=args.base_image_dir,
        dataset_col=args.dataset_col,
        image_col=args.image_col,
        model_size=args.model_size
    )
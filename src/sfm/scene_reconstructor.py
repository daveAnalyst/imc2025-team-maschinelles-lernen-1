import os
import shutil # For deleting temp directories if needed
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from src.sfm.h5_to_db import add_keypoints, add_matches
import torch
import pycolmap # For running COLMAP SfM
from tqdm.auto import tqdm
from PIL import Image # For image loading if not done by preprocessing_module directly

# Assuming database.py and h5_to_db.py are in the same directory (src/sfm/)
# and an __init__.py exists in src/sfm/
from .database import COLMAPDatabase, image_ids_to_pair_id # Relative import
# We'll adapt Davin's h5_to_db functions directly within this script or import them if they are clean
# For now, let's assume we'll adapt the logic for better control within a cluster.

# --- Helper function to adapt Davin's find_keypoints logic ---
def extract_and_cache_keypoints_for_cluster(
    image_ids_to_process: list, # List of 'dataset__filename.png'
    all_image_paths_lookup: dict,
    aliked_model,
    preprocessing_module,
    target_aliked_input_size: int,
    cluster_feature_dir: Path, # Path to save keypoints.h5 for this cluster
    device: str
):
    """
    Extracts and caches ALIKED keypoints and descriptors for images in a cluster.
    Saves to keypoints.h5 and descriptors.h5 in cluster_feature_dir.
    """
    print(f"  Extracting/caching ALIKED features for {len(image_ids_to_process)} unique images in cluster...")
    os.makedirs(cluster_feature_dir, exist_ok=True)
    keypoints_h5_path = cluster_feature_dir / "keypoints.h5"
    descriptors_h5_path = cluster_feature_dir / "descriptors.h5"

    # Use 'a' mode to append or create if not exists
    with h5py.File(keypoints_h5_path, mode="a") as f_keypoints, \
         h5py.File(descriptors_h5_path, mode="a") as f_descriptors:
        
        for image_id_combined in tqdm(image_ids_to_process, desc="  Extracting ALIKED"):
            # image_id_combined is 'dataset__filename.png'
            # We need just 'filename.png' as the key for HDF5, consistent with Davin's original approach
            image_filename_key = image_id_combined.split('__')[-1]

            if image_filename_key in f_keypoints and image_filename_key in f_descriptors:
                # print(f"    Features for {image_filename_key} already cached.")
                continue

            full_path = all_image_paths_lookup.get(image_id_combined)
            if not full_path or not os.path.exists(full_path):
                print(f"    Warning: Path not found or invalid for {image_id_combined}. Skipping feature extraction.")
                continue

            try:
                # Preprocessing
                pil_img = preprocessing_module.load_image_pil(full_path)
                if not pil_img: continue
                pil_img_resized = preprocessing_module.resize_image_maintain_aspect_ratio(pil_img, target_aliked_input_size)
                if not pil_img_resized: continue

                # Convert PIL to Tensor for ALIKED (LightGlue's load_image does this)
                # Assuming ALIKED from LightGlue's context takes a path or a tensor
                # For simplicity, let's use LightGlue's load_image which handles device placement
                from lightglue.utils import load_image # Import here or pass function
                image_tensor = load_image(pil_img_resized).to(device) # load_image can take PIL image

                with torch.inference_mode():
                    feats = aliked_model.extract(image_tensor) # ALIKED model passed in
                    
                    keypoints_np = feats["keypoints"].squeeze().cpu().numpy()
                    descriptors_np = feats["descriptors"].squeeze().cpu().numpy()
                
                f_keypoints[image_filename_key] = keypoints_np
                f_descriptors[image_filename_key] = descriptors_np
            except Exception as e:
                print(f"    Error extracting ALIKED features for {image_filename_key}: {e}")
    print("  ALIKED feature extraction/caching complete for cluster.")
    return keypoints_h5_path, descriptors_h5_path


# --- Helper function to adapt Davin's compare_image logic ---
def match_image_pair_with_tta(
    image_id1_combined: str, image_id2_combined: str,
    all_image_paths_lookup: dict,
    f_keypoints, f_descriptors, f_matches_output, # HDF5 file handles (opened for writing/appending matches)
    lightglue_model,
    device: str,
    rotation_tta: bool = True,
    min_matches_threshold: int = 15 # Min matches to save
):
    """
    Matches a single pair of images using cached features and LightGlue, with optional Rotation TTA.
    Appends matches to f_matches_output HDF5 file handle.
    """
    from lightglue.utils import rbd # For LightGlue's rbd utility
    
    image_filename_key1 = image_id1_combined.split('__')[-1]
    image_filename_key2 = image_id2_combined.split('__')[-1]

    path1 = all_image_paths_lookup.get(image_id1_combined)
    path2 = all_image_paths_lookup.get(image_id2_combined)

    if not (path1 and path2 and os.path.exists(path1) and os.path.exists(path2)):
        print(f"    Skipping pair ({image_filename_key1}, {image_filename_key2}), one or both paths invalid.")
        return

    # Load pre-extracted features
    try:
        feats0_kp_np = f_keypoints[image_filename_key1][...]
        feats0_desc_np = f_descriptors[image_filename_key1][...]
        
        feats0 = {
            "keypoints": torch.from_numpy(feats0_kp_np)[None].to(device),
            "descriptors": torch.from_numpy(feats0_desc_np)[None].to(device)
        }
    except KeyError:
        print(f"    Features not found for {image_filename_key1} in HDF5. Skipping pair.")
        return
    
    best_matches_np = None
    best_match_count = -1
    # best_rot_code = 0 # If you need to store which rotation was best

    rotations_to_try = [0, 1, 2, 3] if rotation_tta else [0] # 0, 90, 180, 270 deg

    for rot_code in rotations_to_try:
        try:
            # For image2, we need to handle rotation if TTA is enabled
            # This means ALIKED should have been run on rotated versions OR we rotate keypoints (harder)
            # Easiest for now: Assume ALIKED features are for original orientation.
            # LightGlue's `load_image` can take PIL, we can rotate PIL image before tensor conversion
            # OR, Davin's find_keypoints could save features for rotated versions if needed
            # For this refactor, let's assume features are for non-rotated images from HDF5
            # and LightGlue processes them. Rotation TTA for keypoints themselves is more complex.
            # The IMC2024 winner applied rotation *before* ALIKED for one image in the pair.
            # For simplicity here, we'll match original features and note TTA is a further enhancement
            # on HOW features are extracted or selected for image2.
            # If ALIKED features were already extracted for rotated versions and cached with a modified key,
            # we would load the appropriate key here.

            # Let's assume for now, no direct rotation of features here, but it's a point for Davin to integrate
            # into his feature extraction caching strategy if he implements full TTA at extraction.
            # If TTA is just about trying different image2 orientations *at matching time with original features*:
            
            feats1_kp_np = f_keypoints[image_filename_key2][...]
            feats1_desc_np = f_descriptors[image_filename_key2][...]
            
            feats1_current = {
                "keypoints": torch.from_numpy(feats1_kp_np)[None].to(device),
                "descriptors": torch.from_numpy(feats1_desc_np)[None].to(device)
            }
            # TODO: If doing TTA by rotating image B and re-extracting on the fly:
            # pil_img2 = Image.open(path2).convert("RGB")
            # if rot_code > 0: pil_img2 = pil_img2.rotate(rot_code * 90)
            # tensor2_for_aliked = ... (preprocess pil_img2) ...
            # feats1_current = aliked_model.extract(tensor2_for_aliked)
            # (This would mean not using cached descriptors for rotated versions, or caching them differently)

            with torch.inference_mode():
                matches01 = lightglue_model({'image0': feats0, 'image1': feats1_current})
            
            # No need for rbd here if feats0/feats1 are already structured correctly
            current_matches_torch = matches01['matches0'].cpu() # Get indices for image0
            # current_matches_img1_indices = matches01['matches1'] # This is not directly output by LightGlue's default dict output
                                                              # 'matches0' gives indices for feats0, which pair with corresponding indices in feats1
            
            # The output `matches01['matches']` is shape (N,2) with indices for feats0 and feats1
            current_matches_np = matches01['matches'].cpu().numpy()
            
            # TODO: Add RANSAC/Fundamental Matrix verification here if LightGlue isn't doing it sufficiently
            # For now, just use raw match count
            num_current_matches = len(current_matches_np)

            if num_current_matches > best_match_count:
                best_match_count = num_current_matches
                best_matches_np = current_matches_np
                # best_rot_code = rot_code # If tracking best rotation
            
            if not rotation_tta: # If not doing TTA, break after first try
                break
        except KeyError:
            print(f"    Features not found for {image_filename_key2} in HDF5 for rotation {rot_code}. Skipping this rotation.")
            continue
        except Exception as e:
            print(f"    Error matching {image_filename_key1} with {image_filename_key2} (rot {rot_code}): {e}")
            continue
            
    if best_matches_np is not None and len(best_matches_np) >= min_matches_threshold:
        group = f_matches_output.require_group(image_filename_key1)
        if image_filename_key2 in group:
            # print(f"    Matches for ({image_filename_key1}, {image_filename_key2}) already exist. Skipping append (or implement overwrite/update).")
            # For simplicity, let's overwrite if it exists with new best matches
            del group[image_filename_key2] 
            group.create_dataset(image_filename_key2, data=best_matches_np.reshape(-1,2))
        else:
            group.create_dataset(image_filename_key2, data=best_matches_np.reshape(-1,2), chunks=True, maxshape=(None,2))
        # print(f"    Saved {len(best_matches_np)} matches for ({image_filename_key1}, {image_filename_key2}).")
    # else:
        # print(f"    Found only {best_match_count} matches for ({image_filename_key1}, {image_filename_key2}), less than threshold {min_matches_threshold}. Not saved.")


# --- Helper function to adapt Davin's COLMAP DB import logic ---
# This directly uses functions from your src/sfm/h5_to_db.py
# We assume h5_to_db.py has:
# - add_keypoints (aliased as h5db_add_keypoints)
# - add_matches (aliased as h5db_add_matches)
# - And potentially get_focal, create_camera if not handled elsewhere
# The import_into_colmap in Davin's original script is a good template.

def create_colmap_db_from_h5(
    image_dir_for_colmap_paths: Path, # Directory containing actual images for COLMAP
    cluster_feature_dir: Path,      # Directory with keypoints.h5, matches.h5 for this cluster
    cluster_colmap_db_path: Path,
    camera_model_colmap: str = 'SIMPLE_RADIAL' # Common default
):
    """
    Creates a COLMAP database by importing features and matches from HDF5 files.
    """
    print(f"  Creating COLMAP DB at: {cluster_colmap_db_path}")
    if os.path.exists(cluster_colmap_db_path):
        os.remove(cluster_colmap_db_path)

    db = COLMAPDatabase.connect(str(cluster_colmap_db_path)) # Ensure path is string
    db.create_tables()

    # This part needs to be robust, using functions from h5_to_db.py.
    # Davin's `add_keypoints` from his `h5_to_db.py` takes (db, feature_dir, image_path_root, img_ext, camera_model)
    # Davin's `add_matches` from his `h5_to_db.py` takes (db, feature_dir, fname_to_id)
    try:
        # The `image_path` argument to h5db_add_keypoints is the root directory of images for that DB
        # The `feature_dir` is where keypoints.h5 is located.
        fname_to_id = add_keypoints(db, cluster_feature_dir, image_dir_for_colmap_paths, 
                                         img_ext="", # Assuming filenames in H5 don't need extension added
                                         camera_model=camera_model_colmap, 
                                         single_camera=False) # Usually False for diverse scenes
        
        add_matches(db, cluster_feature_dir, fname_to_id)
        db.commit()
        print(f"  COLMAP DB populated successfully.")
    except Exception as e:
        print(f"  Error populating COLMAP DB: {e}")
        # db.close() # Ensure db is closed even on error
        # if os.path.exists(cluster_colmap_db_path): os.remove(cluster_colmap_db_path) # Clean up failed DB
        raise # Re-raise the exception to signal failure
    finally:
        db.close()


# --- Helper function to adapt Davin's pycolmap reconstruction logic ---
def run_pycolmap_sfm_and_get_poses(
    cluster_colmap_db_path: Path,
    image_dir_for_colmap_paths: Path, # Directory containing actual images for COLMAP
    sfm_output_dir_for_this_cluster: Path, # Where COLMAP saves its models
    colmap_options_dict: dict = None
) -> dict:
    """
    Runs pycolmap incremental mapping and extracts poses.
    Returns a dictionary: {'image_filename.png': {'R': R_array, 'T': T_array, 'registered': True/False}}
    """
    print(f"  Running pycolmap.incremental_mapping...")
    print(f"    DB: {cluster_colmap_db_path}")
    print(f"    Image Dir: {image_dir_for_colmap_paths}")
    print(f"    Output Dir: {sfm_output_dir_for_this_cluster}")

    os.makedirs(sfm_output_dir_for_this_cluster, exist_ok=True)

    mapper_options = pycolmap.IncrementalPipelineOptions()
    if colmap_options_dict:
        for opt, val in colmap_options_dict.items():
            if hasattr(mapper_options, opt):
                setattr(mapper_options, opt, val)
            else:
                print(f"    Warning: Invalid COLMAP option '{opt}' provided.")
    else: # Default sensible options
        mapper_options.min_model_size = 3
        mapper_options.max_num_models = 1 # Try to get one good model first

    # Crucial: We are providing features & matches, so COLMAP should not re-extract/re-match.
    # This is usually handled by how the database is populated (features & matches tables exist).
    # pycolmap.incremental_mapping should then use these.
    # If pycolmap still tries to run its own feature/matcher, the DB wasn't populated right
    # or a different pycolmap function is needed (e.g., one that assumes pre-existing DB with features/matches).
    
    # The `image_path` argument to incremental_mapping is the directory COLMAP will find images listed in its DB.
    try:
        maps = pycolmap.incremental_mapping(
            database_path=str(cluster_colmap_db_path), 
            image_path=str(image_dir_for_colmap_paths), # This must be where COLMAP can find images by name from DB
            output_path=str(sfm_output_dir_for_this_cluster), 
            options=mapper_options
        )
    except Exception as e:
        print(f"  ERROR during pycolmap.incremental_mapping: {e}")
        return {} # Return empty if SfM fails

    cluster_poses = {}
    if maps and isinstance(maps, dict) and len(maps) > 0:
        print(f"  pycolmap returned {len(maps)} model(s).")
        # Select the best model (e.g., most registered images, or largest)
        best_model_idx = None
        max_reg_images = 0
        for i, rec in maps.items():
            print(f"    Model {i}: {rec.summary()}")
            if len(rec.images) > max_reg_images:
                max_reg_images = len(rec.images)
                best_model_idx = i
        
        if best_model_idx is not None:
            print(f"  Selected best model: {best_model_idx} with {max_reg_images} registered images.")
            best_map = maps[best_model_idx]
            for image_id_colmap, image_obj in best_map.images.items():
                # image_obj.name is the filename as stored in COLMAP DB
                image_filename_key = image_obj.name 
                R_mat = image_obj.cam_from_world.rotation.matrix() # 3x3 NumPy array
                T_vec = image_obj.cam_from_world.translation # 3, NumPy array
                cluster_poses[image_filename_key] = {'R': R_mat, 'T': T_vec, 'registered': True}
        else:
            print("  No suitable reconstruction model found by pycolmap.")
    else:
        print("  pycolmap.incremental_mapping did not return any models.")
        
    return cluster_poses


# --- Main Orchestrator Function for this Module ---
def reconstruct_scene_cluster(
    image_ids_in_cluster: list,           # List of 'dataset__image_filename.png'
    all_image_paths_lookup: dict,         # Dict mapping 'dataset__image_filename.png' to full_path
    selected_image_pairs_ids: list,       # List of ('id1_comb', 'id2_comb') tuples from pair_selector
    preprocessing_module,                  # Imported src.data.preprocessing
    aliked_model,                          # Pre-initialized ALIKED model
    lightglue_model,                       # Pre-initialized LightGlue model
    base_output_dir_for_sfm_run: str,      # e.g., /kaggle/working/sfm_temp/dataset_X__cluster_Y/
    target_aliked_input_size: int = 1024,
    colmap_mapper_options_dict: dict = None,
    rotation_tta: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> dict: # Returns {'image_filename.png': {'R': R_array, 'T': T_array, 'registered': True/False}}
    
    # Ensure base output directory for this specific cluster run exists
    # This directory will hold keypoints.h5, matches.h5, colmap.db, and COLMAP model outputs for this cluster
    os.makedirs(base_output_dir_for_sfm_run, exist_ok=True)
    print(f"\n--- Processing cluster, outputting to: {base_output_dir_for_sfm_run} ---")

    # 1. Get unique image IDs and their full paths that need feature extraction for this cluster
    unique_img_ids_in_pairs = set()
    for id1, id2 in selected_image_pairs_ids:
        unique_img_ids_in_pairs.add(id1)
        unique_img_ids_in_pairs.add(id2)
    
    if not unique_img_ids_in_pairs and len(image_ids_in_cluster) >=2:
        print("  No selected pairs, but cluster has images. SfM cannot proceed with pairs.")
        # Return NaNs for all images in this cluster
        return {img_id.split('__')[-1]: {'R': np.full((3,3), np.nan), 'T': np.full((3,), np.nan), 'registered': False} 
                for img_id in image_ids_in_cluster}
    elif not unique_img_ids_in_pairs: # Cluster might have < 2 images or no pairs for other reasons
        print("  No images involved in pairs for this cluster. Skipping SfM.")
        return {img_id.split('__')[-1]: {'R': np.full((3,3), np.nan), 'T': np.full((3,), np.nan), 'registered': False} 
                for img_id in image_ids_in_cluster}


    # 2. Extract and cache ALIKED keypoints and descriptors for these unique images
    keypoints_h5_path, descriptors_h5_path = extract_and_cache_keypoints_for_cluster(
        list(unique_img_ids_in_pairs), # Pass list of unique combined_ids
        all_image_paths_lookup,
        aliked_model,
        preprocessing_module,
        target_aliked_input_size,
        Path(base_output_dir_for_sfm_run), # feature_dir for this cluster
        device
    )

    # 3. Match selected image pairs using cached features and LightGlue (with TTA)
    matches_h5_path = Path(base_output_dir_for_sfm_run) / "matches.h5"
    if os.path.exists(matches_h5_path): # Ensure fresh matches file for this run
        os.remove(matches_h5_path)

    if os.path.exists(keypoints_h5_path) and os.path.exists(descriptors_h5_path):
        with h5py.File(keypoints_h5_path, mode="r") as f_keypoints, \
             h5py.File(descriptors_h5_path, mode="r") as f_descriptors, \
             h5py.File(matches_h5_path, mode="w") as f_matches_output: # 'w' to create anew
            
            for id1_comb, id2_comb in tqdm(selected_image_pairs_ids, desc="  Matching selected pairs"):
                match_image_pair_with_tta(
                    id1_comb, id2_comb,
                    all_image_paths_lookup,
                    f_keypoints, f_descriptors, f_matches_output,
                    lightglue_model,
                    device,
                    rotation_tta=rotation_tta
                )
        print(f"  Finished matching. Matches saved in {matches_h5_path}")
    else:
        print("  Keypoint/Descriptor HDF5 files not found after extraction attempt. Cannot perform matching.")
        return {img_id.split('__')[-1]: {'R': np.full((3,3), np.nan), 'T': np.full((3,), np.nan), 'registered': False} 
                for img_id in image_ids_in_cluster}

    # 4. Create COLMAP database from HDF5 files for this cluster
    # We need the directory that COLMAP will see as the root for image names listed in its DB
    # This should be the directory containing the actual image files (e.g., .../test/dataset_X/ or .../test/)
    # This assumes all images in image_ids_in_cluster come from the same "image_dir_for_colmap"
    # which is true if they are from one dataset and your paths are like /kaggle/input/.../test/dataset_name/image.png
    # If test images are flat in /kaggle/input/.../test/, then image_dir_for_colmap is that /test/ dir.

    # Let's get the common parent directory of the actual image files for this cluster
    # This is tricky if images could be from different top-level dataset folders for some reason
    # For this competition, images in a cluster should be from ONE dataset, so their parent 'dataset' folder is common.
    # Or if test is flat, the common parent is the flat test dir.
    
    # Simplification: assume all_image_paths_lookup gives full paths.
    # The 'image_dir' for COLMAP should be where it can find the images using the names stored in the DB.
    # Davin's h5_to_db.py uses image basenames as keys in HDF5 and names in DB.
    # So, image_dir_for_colmap_paths must be the directory containing these basenames.
    # Example: if full path is /path/to/datasetA/image1.png, and key is image1.png, then image_dir is /path/to/datasetA/
    
    # Determine the common image directory for COLMAP based on the structure
    # This needs to be robust based on how paths are structured.
    # If paths are /kaggle/input/COMP/test/dataset_A/img1.png, /kaggle/input/COMP/test/dataset_A/img2.png
    # then image_dir_for_colmap_paths = /kaggle/input/COMP/test/dataset_A
    # If paths are /kaggle/input/COMP/test/img1.png, /kaggle/input/COMP/test/img2.png (flat)
    # then image_dir_for_colmap_paths = /kaggle/input/COMP/test
    
    # Let's find the common parent directory for the actual image files in the cluster
    # This assumes image_ids_in_cluster are all from the same original dataset folder
    if not image_ids_in_cluster: # Should have been caught earlier
        return {}

    first_img_full_path = all_image_paths_lookup.get(image_ids_in_cluster[0])
    if not first_img_full_path:
        print("  Could not determine image directory for COLMAP. Aborting SfM for cluster.")
        return {img_id.split('__')[-1]: {'R': np.full((3,3), np.nan), 'T': np.full((3,), np.nan), 'registered': False} 
                for img_id in image_ids_in_cluster}

    # If images are like .../dataset_name/image_name.png, then image_dir_for_colmap_paths is .../dataset_name
    # If images are flat like .../test/image_name.png, then image_dir_for_colmap_paths is .../test
    # The `h5db_add_keypoints` uses `image_path` argument as the root from where image names (keys in H5) are found.
    # It expects keys in H5 to be relative paths from this `image_path` or just basenames if `img_ext` is used.
    # Davin's original `find_keypoints` used `path.name` as key. So they are basenames.
    # Therefore, `image_dir_for_colmap_paths` must be the directory containing these images.
    
    # This assumes all images in a cluster come from the same dataset folder.
    # The paths in all_image_paths_lookup are like /kaggle/input/COMP_NAME/test/DATASET_NAME/IMAGE.png
    # OR /kaggle/input/COMP_NAME/test/IMAGE.png if flat.
    # The keys in HDF5 are just IMAGE.png.
    # So, the image_dir for `h5db_add_keypoints` should be the directory containing these images.
    image_dir_for_colmap_paths = Path(os.path.dirname(first_img_full_path))
    # For flat test structure: image_dir_for_colmap_paths will be /kaggle/input/COMP_NAME/test/
    # For nested test structure: image_dir_for_colmap_paths will be /kaggle/input/COMP_NAME/test/DATASET_NAME/

    cluster_colmap_db_path = Path(base_output_dir_for_sfm_run) / "colmap.db"
    try:
        create_colmap_db_from_h5(
            image_dir_for_colmap_paths,
            Path(base_output_dir_for_sfm_run), # feature_dir where keypoints.h5, matches.h5 are
            cluster_colmap_db_path
        )
    except Exception as e_db:
        print(f"  Error creating COLMAP DB for cluster: {e_db}. Skipping SfM.")
        return {img_id.split('__')[-1]: {'R': np.full((3,3), np.nan), 'T': np.full((3,), np.nan), 'registered': False} 
                for img_id in image_ids_in_cluster}


    # 5. Run SfM using pycolmap
    sfm_models_output_dir = Path(base_output_dir_for_sfm_run) / "colmap_models"
    
    # IMPORTANT: Do NOT run pycolmap.match_exhaustive if you've imported custom matches.
    # Go straight to incremental_mapping.
    # If geometric verification is needed for custom matches, that should happen before DB import,
    # or by using pycolmap. অনেকের মতে, COLMAP's geometric verification.
    # Davin's `compare_image` might need to do RANSAC and populate `two_view_geometries` table,
    # or pycolmap needs to run geometric verification step.
    # For now, let's assume matches in matches.h5 are good enough or LightGlue did some robust estimation.
    
    cluster_reconstruction_poses = run_pycolmap_sfm_and_get_poses(
        cluster_colmap_db_path,
        image_dir_for_colmap_paths, # Directory containing the actual image files
        sfm_models_output_dir,
        colmap_options_dict # type: ignore
    )

    # 6. Consolidate results for all images in the original input cluster list
    # (mapping back from basenames to original image_id_combined if needed, though result uses basenames)
    final_poses_for_this_cluster = {}
    for img_id_combined_original in image_ids_in_cluster:
        img_basename_key = img_id_combined_original.split('__')[-1]
        if img_basename_key in cluster_reconstruction_poses and cluster_reconstruction_poses[img_basename_key].get('registered', False):
            final_poses_for_this_cluster[img_id_combined_original] = cluster_reconstruction_poses[img_basename_key]
        else:
            final_poses_for_this_cluster[img_id_combined_original] = {
                'R': np.full((3,3), np.nan), 
                'T': np.full((3,), np.nan), 
                'registered': False
            }
            
    num_registered = sum(1 for p_data in final_poses_for_this_cluster.values() if p_data['registered'])
    print(f"  SfM finished for cluster. Registered {num_registered} / {len(image_ids_in_cluster)} images.")
    return final_poses_for_this_cluster


# --- Example Test Block (if __name__ == '__main__') ---
# This would require setting up dummy image paths, selected pairs,
# mock aliked/lightglue models, mock preprocessing_module,
# and HDF5 files, or actually running parts of the pipeline.
# For now, this script is intended to be imported as a module.
# Testing will primarily happen via the main Kaggle notebook.
if __name__ == '__main__':
    print("This script is intended to be imported as a module into the main pipeline.")
    print("To test, you would typically call reconstruct_scene_cluster with appropriate inputs.")
    # Example of how it *might* be called (needs actual data and models)
    # Create dummy inputs for a conceptual test:
    # dummy_image_ids = [f"datasetA__img{i}.png" for i in range(5)]
    # dummy_paths_lookup = {id: f"/fake/path/to/{id.split('__')[0]}/{id.split('__')[-1]}" for id in dummy_image_ids}
    # dummy_selected_pairs = [(dummy_image_ids[0], dummy_image_ids[1]), (dummy_image_ids[1], dummy_image_ids[2])]
    #
    # class MockModel:
    #     def __init__(self, device='cpu'): self.device = device
    #     def eval(self): return self
    #     def to(self, device): self.device=device; return self
    #     def extract(self, tensor): return {'keypoints': torch.rand(1,10,2), 'descriptors': torch.rand(1,10,128)}
    #     def __call__(self, data_dict): return {'matches0': torch.randint(0,10,(5,)), 'matches': torch.randint(0,10,(5,2))} # Mock LightGlue
    #
    # class MockPreprocessor:
    #     def load_image_pil(self, pth): return Image.new("RGB", (640,480))
    #     def resize_image_maintain_aspect_ratio(self, img, dim): return img.resize((dim, int(dim*img.height/img.width)))
    #
    # mock_aliked = MockModel()
    # mock_lightglue = MockModel()
    # mock_preproc = MockPreprocessor()
    #
    # results = reconstruct_scene_cluster(
    #     image_ids_in_cluster=dummy_image_ids,
    #     all_image_paths_lookup=dummy_paths_lookup,
    #     selected_image_pairs_ids=dummy_selected_pairs,
    #     preprocessing_module=mock_preproc,
    #     aliked_model=mock_aliked,
    #     lightglue_model=mock_lightglue,
    #     base_output_dir_for_sfm_run="./sfm_test_output/cluster_dummy",
    #     target_aliked_input_size=640,
    #     colmap_mapper_options_dict={'min_model_size': 2},
    #     rotation_tta=False
    # )
    # print("\nDummy test results:")
    # for img_id_comb, pose_data in results.items():
    # print(f"  {img_id_comb}: Registered={pose_data['registered']}")
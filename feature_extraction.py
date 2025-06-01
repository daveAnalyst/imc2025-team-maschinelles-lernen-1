#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import os


# In[2]:


from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
from lightglue import match_pair
from itertools import combinations
import h5py
import numpy as np
import re
import subprocess
import pycolmap
from database import COLMAPDatabase
from h5_to_db import add_keypoints, add_matches
from tqdm import tqdm
import kornia.feature as KF
import open3d as o3d
import global_dino_extractor as gdino
import pair_selector


# In[3]:


def read_csv(path): 
    file_path = os.path.join(os.getcwd(),"image-matching-challenge-2025", path)
    return pd.read_csv(file_path)


# In[4]:


def open_image(img_paths, scenes, n_cols=7, figsize=(15, 8)): 
    n_rows = len(img_paths) // 7 + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, img_path in enumerate(img_paths):
        #In case we have more images than subplots
        if i >= len(axes): 
            break
        img = cv2.imread(img_path)
        if img is None: 
            axes[i].set_title("No Image found")
            axes[i].axis("off")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].axis("off")
        axes[i].set_title(f"{scenes[i]}")
        
    for j in range(i+1, len(axes)): 
        axes[j].axis("off")
    plt.show()


# In[5]:


def get_pair_index(paths): 
    #Obtains all index pairs of small list
    return list(combinations(range(len(paths)), 2))

def find_keypoints(extractor, paths, feature_dir): 
    with h5py.File(feature_dir/ "keypoints.h5", mode="a") as f_keypoints, h5py.File(feature_dir / "descriptors.h5", mode="a") as f_descriptors: 
        for path in tqdm(paths, desc="Computing and saving keypoints..."): 
            #If path is a string and not Path(..) object
            if(isinstance(path, str)): 
                path = Path(path)
            key = path.name
            if key in f_keypoints and key in f_descriptors: 
                print(f"This key {key} is already here")
                continue
                
            #Using inference_mode to save memory and efficienter
            with torch.inference_mode(): 
                image = load_image(path)
                feats = extractor.extract(image)
                keypoints = feats["keypoints"]
                descriptors = feats["descriptors"]
                if not isinstance(keypoints, np.ndarray): 
                    keypoints = keypoints.squeeze().cpu().numpy()
                if not isinstance(descriptors, np.ndarray): 
                    descriptors = descriptors.squeeze().cpu().numpy() 
                f_keypoints[key] = keypoints
                f_descriptors[key] = descriptors
                


# In[6]:


def compare_image(f_keypoints, f_descriptors, f_matches, device, key1_path, key2_path, matcher,  show_comparison = False, show_points=False):
    min_matches=20
    image0 = load_image(key1_path)
    image1 = load_image(key2_path)
    key1 = Path(key1_path).name
    key2 = Path(key2_path).name

    feats0 = {
        "keypoints": torch.from_numpy(f_keypoints[key1][...])[None].to(device), 
        "descriptors": torch.from_numpy(f_descriptors[key1][...])[None].to(device)
    }
    feats1 = {
        "keypoints": torch.from_numpy(f_keypoints[key2][...])[None].to(device), 
        "descriptors": torch.from_numpy(f_descriptors[key2][...])[None].to(device)
    }
    #Match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matches = matches01['matches'].to(device)
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]
    desc0 = feats0['descriptors'][matches[..., 0]]
    desc1 = feats1['descriptors'][matches[..., 1]]

    # Compute L2 distance between descriptors (to match matcher output style)
    descriptor_distances = torch.norm(desc0 - desc1, dim=1)
    #descriptor_distances = descriptor_distances.unsqueeze(1)  # from [N] to [N,1]
    distance = torch.norm(points0 - points1, dim=1)
    if show_comparison: 
        axes = viz2d.plot_images([image0, image1])
        viz2d.plot_matches(points0, points1, color="lime", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
        kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
        viz2d.plot_images([image0, image1])
        viz2d.plot_keypoints([feats0["keypoints"], feats1["keypoints"]], colors=[kpc0, kpc1], ps=10)
        
    n_matches = len(matches)  
    if n_matches >= min_matches: 
        group = f_matches.require_group(key1)
        
        new_data = matches.detach().cpu().numpy().reshape(-1, 2)
        
        if key2 in group:
            ds = group[key2]
            old_shape = ds.shape[0]
            ds.resize((old_shape + new_data.shape[0], 2))
            ds[old_shape:] = new_data
        else:
            group.create_dataset(
                key2,
                data=new_data,
                maxshape=(None, 2),
                chunks=True
        )
            

#Importing h5 file to colmap database
def import_into_colmap(path, feature_dir, database_path): 
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, path, "", "simple-pinhole", single_camera)
    add_matches(db, 
                feature_dir, 
                fname_to_id)
    db.commit()
    db.close()

#Reconstruct the 3d image by finding the rotation matrix and translation vector
def reconstruct_images(output_path, database_path, images_dir):
    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.min_model_size = 3
    mapper_options.max_num_models = 2
    
    maps = pycolmap.incremental_mapping(
        database_path=database_path, 
        image_path=images_dir,
        output_path=Path.cwd() / output_path, 
        options=mapper_options,
    )

    #Create
    data = []
    for model in maps.values(): 
        for image_id, image in model.images.items(): 
            rotation = image.cam_from_world.rotation.matrix().flatten().tolist()
            translation = image.cam_from_world.translation.tolist()
            row = [image.name, ";".join(str(x) for x in rotation) , ";".join(str(x) for x in translation)]
            data.append(row)
    
    columns = (['image_name'] + ['rotation_matrix'] + ['translation_vector'])
    
    #Save to CSV
    df = pd.DataFrame(data, columns=columns)
    file_name = 'camera_poses.csv'
    write_header = not os.path.exists(file_name)
    
    df.to_csv(file_name, mode='a', header=write_header, index=False)
    
    print(f'Camera poses saved to {file_name}')


# In[7]:


#To 3d visualize the reconstructed image
def visualize_reconstructed(reconstruct_path):  
    recon = pycolmap.Reconstruction(reconstruct_path)
    points = [point.xyz for point in recon.points3D.values()]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


# Running everything together

# In[9]:


def run_reconstruction(output_npz_path, 
                       dino_embedding, 
                       image_list_csv_path, 
                       base_image_dir,
                       df, 
                       generate_key_des=True, 
                       extractor=extractor , 
                       matcher=matcher, 
                       feature_dir_ori = feature_dir, 
                       database_path = "colmap.db", 
                       device="cuda"): 

    
    if not os.path.exists(dino_embedding):
        gdino.extract_features_and_save(
            image_list_csv_path=image_list_csv_path, 
            output_npz_path = output_npz_path, 
            base_image_dir=base_image_dir
        )
    data = np.load(dino_embedding)
    pairs = pair_selector.select_pairs_by_embedding_similarity(data.files, data)
    index_pairs = [(data.files.index(a), data.files.index(b)) for a, b in pairs]

    data_dict = {}
    for i in range(df.shape[0]): 
        dataset = df.iloc[i]["dataset"]
        scene = df.iloc[i]["scene"]
        path = df.iloc[i]["image_path"]
        data_dict.setdefault(dataset, {}).setdefault(scene, []).append(path)

    datasets = list(data_dict.keys())
    
    for dataset in datasets: 
            
        print(dataset)
        feature_dir = Path(os.path.join(feature_dir_ori, dataset))
        os.makedirs(feature_dir, exist_ok=True)
        
        for scene in data_dict[dataset]: 
            images_dir = Path(data_dict[dataset][scene][0]).parent
            image_paths = data_dict[dataset][scene]
            if generate_key_des:
                find_keypoints(extractor, image_paths , feature_dir)

    
    file_path = os.path.join(feature_dir, "matches.h5")
    if os.path.exists(file_path): 
        os.remove(file_path)
        print("Old matches.h5 deleted")
    
    data_dict = {}
    for i in range(df.shape[0]): 
        dataset = df.iloc[i]["dataset"]
        scene = df.iloc[i]["scene"]
        path = df.iloc[i]["image_path"]
        data_dict.setdefault(dataset, {}).setdefault(scene, []).append(path)

    datasets = list(data_dict.keys())

    total=0
    for dataset in datasets: 
        feature_dir = Path(os.path.join(feature_dir_ori, dataset))

        print(dataset)
        for scene in data_dict[dataset]: 
            images_dir = Path(data_dict[dataset][scene][0]).parent
            image_paths = data_dict[dataset][scene]
            print(f"There're {len(image_paths)} images!")
            print(f"From Index {total} - {total+len(image_paths)}")
            # if generate_key_des:
            #     print("Generating keypoints and descriptors...")
            #     find_keypoints(extractor, image_paths , feature_dir)
            with h5py.File(feature_dir / "keypoints.h5", mode="r") as f_keypoints, h5py.File(feature_dir / "descriptors.h5", mode="r") as f_descriptors, h5py.File(feature_dir/"matches.h5", mode="a") as f_matches:
                path_pair_index = [(a, b) for a, b in index_pairs if total <= a < total+len(image_paths) and total <= b < total+len(image_paths)]
                for i1, i2 in tqdm(path_pair_index, desc="Computing keypoint distances"):
                    key1, key2 = Path(df["image_path"][i1]), Path(df["image_path"][i2])
                    compare_image(f_keypoints, f_descriptors, f_matches, device, key1, key2, matcher, False)
                print(list(f_matches.keys()))
                
            images_dir = Path(image_paths[0]).parent
            if os.path.exists("colmap.db"): 
                os.unlink("colmap.db")
                print("Old colmap.db is deleted")
            import_into_colmap(images_dir, feature_dir, database_path)

            
            pycolmap.match_exhaustive(database_path)
            reconstruct_path = os.path.join(os.getcwd(), "reconstruct_pipeline_outputs")
            reconstruct_images(reconstruct_path, database_path, images_dir)
        
            total += len(image_paths)
        


# In[10]:

#Dummy Test

if __name__ == "__main__":
    #First, create dataset that contains [dataset, scene, image_path]
    df = read_csv("train_labels.csv")
    df["image_path"] = df.apply(lambda row: os.path.join(os.getcwd(), "image-matching-challenge-2025", "train",  row["dataset"], row["image"]), axis=1)
    
    #secondly, load extractor and matcher
    device="cuda"
    output_npz_path = "./dino_embeddings/train_embeddings_vits.npz"
    dino_embedding = "./dino_embeddings/train_embeddings_vits.npz"
    extractor = ALIKED(max_num_keypoints=2048, resize = 1024).eval()
    feature_dir = Path(os.path.join(os.getcwd() , "feature_extraction"))
    matcher = LightGlue(features="aliked", depth_confidence=1.0, width_confidence=1.0).eval().to(device)
    
    #lastly, run all things at once
    run_reconstruction(output_npz_path, dino_embedding, "./image-matching-challenge-2025/train_labels.csv", "./image-matching-challenge-2025/train", df)






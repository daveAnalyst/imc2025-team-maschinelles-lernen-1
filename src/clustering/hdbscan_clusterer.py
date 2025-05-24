import numpy as np
import pandas as pd
import os
import hdbscan
import umap # Optional, if UMAP is a fixed part of your best strategy
# from sklearn.preprocessing import StandardScaler # Optional
import argparse

def load_embeddings(embeddings_npz_path):
    """Loads embeddings from an NPZ file."""
    print(f"Loading embeddings from: {embeddings_npz_path}")
    try:
        loaded_data = np.load(embeddings_npz_path)
        image_ids = []
        embeddings_list = []
        for key in loaded_data.files:
            image_ids.append(key)
            embeddings_list.append(loaded_data[key])
        loaded_data.close()
        embeddings_matrix = np.array(embeddings_list)
        print(f"Loaded embeddings matrix with shape: {embeddings_matrix.shape}")
        return image_ids, embeddings_matrix
    except FileNotFoundError:
        print(f"Error: Embeddings NPZ file not found at {embeddings_npz_path}")
        return None, None
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None, None

def run_hdbscan_clustering(embeddings_matrix, 
                           use_umap=True, # Flag to control UMAP usage
                           umap_n_neighbors=15, umap_n_components=30, 
                           umap_min_dist=0.0, umap_metric='cosine',
                           hdbscan_min_cluster_size=5, hdbscan_metric='euclidean', 
                           hdbscan_min_samples=None):
    """
    Performs UMAP (optional) and HDBSCAN clustering on embeddings.
    Returns cluster labels.
    """
    data_for_clustering = embeddings_matrix

    if use_umap:
        print(f"Applying UMAP: n_neighbors={umap_n_neighbors}, n_components={umap_n_components}, min_dist={umap_min_dist}, metric={umap_metric}")
        try:
            reducer = umap.UMAP(
                n_neighbors=umap_n_neighbors,
                n_components=umap_n_components,
                min_dist=umap_min_dist,
                metric=umap_metric,
                random_state=42 
            )
            data_for_clustering = reducer.fit_transform(embeddings_matrix)
            print(f"Reduced embeddings shape after UMAP: {data_for_clustering.shape}")
        except Exception as e:
            print(f"Error during UMAP: {e}. Proceeding with original embeddings for HDBSCAN.")
            data_for_clustering = embeddings_matrix # Fallback to original if UMAP fails

    print(f"Applying HDBSCAN: min_cluster_size={hdbscan_min_cluster_size}, metric={hdbscan_metric}, min_samples={hdbscan_min_samples if hdbscan_min_samples is not None else 'default'}")
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric=hdbscan_metric,
            # allow_single_cluster=True, # Consider this if some datasets might be one scene
            prediction_data=False # Usually not needed if just fitting and predicting once
        )
        cluster_labels = clusterer.fit_predict(data_for_clustering)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        print(f"HDBSCAN found {n_clusters} clusters and {n_noise} noise points.")
        return cluster_labels
    except Exception as e:
        print(f"Error during HDBSCAN: {e}")
        return None

def save_cluster_assignments(image_ids, cluster_labels, output_csv_path):
    """Saves image_ids and their cluster labels to a CSV file."""
    if image_ids is None or cluster_labels is None or len(image_ids) != len(cluster_labels):
        print("Error: Invalid input for saving cluster assignments.")
        return

    print(f"Saving cluster assignments to: {output_csv_path}")
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    results_df = pd.DataFrame({
        'image_id': image_ids, # These are 'dataset__filename'
        'cluster_label': cluster_labels
    })
    try:
        results_df.to_csv(output_csv_path, index=False)
        print("Cluster assignments saved successfully.")
    except Exception as e:
        print(f"Error saving CSV: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform HDBSCAN clustering on pre-computed image embeddings.")
    parser.add_argument("embeddings_npz_path", help="Path to the .npz file containing image embeddings.")
    parser.add_argument("output_csv_path", help="Path to save the output CSV file with image_id and cluster_label.")
    
    # UMAP parameters
    parser.add_argument("--no_umap", action="store_true", help="Skip UMAP dimensionality reduction.")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP: n_neighbors.")
    parser.add_argument("--umap_n_components", type=int, default=30, help="UMAP: n_components (target dimension).")
    parser.add_argument("--umap_min_dist", type=float, default=0.0, help="UMAP: min_dist.")
    parser.add_argument("--umap_metric", type=str, default="cosine", help="UMAP: distance metric.")
    
    # HDBSCAN parameters
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=5, help="HDBSCAN: min_cluster_size.")
    parser.add_argument("--hdbscan_metric", type=str, default="euclidean", help="HDBSCAN: distance metric for condensed tree.")
    parser.add_argument("--hdbscan_min_samples", type=int, default=None, help="HDBSCAN: min_samples (leave as None for default).")

    args = parser.parse_args()

    image_ids, embeddings_matrix = load_embeddings(args.embeddings_npz_path)

    if image_ids is not None and embeddings_matrix is not None:
        cluster_labels = run_hdbscan_clustering(
            embeddings_matrix,
            use_umap=(not args.no_umap),
            umap_n_neighbors=args.umap_n_neighbors,
            umap_n_components=args.umap_n_components,
            umap_min_dist=args.umap_min_dist,
            umap_metric=args.umap_metric,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
            hdbscan_metric=args.hdbscan_metric,
            hdbscan_min_samples=args.hdbscan_min_samples
        )
        
        if cluster_labels is not None:
            save_cluster_assignments(image_ids, cluster_labels, args.output_csv_path)
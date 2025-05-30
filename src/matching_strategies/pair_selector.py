import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os # For the test block
import pandas as pd # For the test block

def select_pairs_by_embedding_similarity(
        image_ids_in_cluster, 
        all_image_embeddings_dict, 
        top_k=5, 
        similarity_threshold=0.8, # Adjust this threshold based on experimentation
        verbose=False):
    """
    Selects image pairs within a cluster based on DINOv2 embedding similarity.

    Args:
        image_ids_in_cluster (list): List of 'dataset_name__image_filename' for the current cluster.
        all_image_embeddings_dict (dict): Dict mapping 'dataset_name__image_filename' to its DINOv2 embedding.
        top_k (int): For each image, consider pairs with its top K most similar neighbors.
        similarity_threshold (float): Minimum cosine similarity to consider a pair.
        verbose (bool): If True, prints more information.

    Returns:
        list of tuples: A list of unique (image_id1, image_id2) pairs.
    """
    selected_pairs = set() # Use a set to store unique pairs (order doesn't matter for a pair)
    
    if not image_ids_in_cluster or len(image_ids_in_cluster) < 2:
        if verbose: print("Not enough images in cluster to form pairs.")
        return []

    # Get embeddings only for the images in the current cluster
    # Maintain the order to map back from the similarity matrix
    cluster_image_ids_ordered = []
    cluster_embeddings_list = []
    
    for img_id in image_ids_in_cluster:
        if img_id in all_image_embeddings_dict:
            cluster_embeddings_list.append(all_image_embeddings_dict[img_id])
            cluster_image_ids_ordered.append(img_id)
        elif verbose:
            print(f"Warning: Embedding not found for {img_id} in current cluster processing.")

    if not cluster_image_ids_ordered or len(cluster_image_ids_ordered) < 2:
        if verbose: print("Not enough images with embeddings in cluster to form pairs.")
        return []

    cluster_embeddings_matrix = np.array(cluster_embeddings_list)

    # Calculate pairwise cosine similarity matrix for images within the cluster
    try:
        sim_matrix = cosine_similarity(cluster_embeddings_matrix)
    except Exception as e:
        if verbose: print(f"Error calculating similarity matrix: {e}")
        return []
    
    if verbose: print(f"Similarity matrix shape for cluster: {sim_matrix.shape}")

    for i in range(len(cluster_image_ids_ordered)):
        current_image_id = cluster_image_ids_ordered[i]
        
        # Get similarities of image i with all other images in the cluster
        similarities_to_others = sim_matrix[i, :]
        
        # Get indices of top_k similar images (excluding self)
        # argsort sorts ascending, so we take from the end for descending similarity
        # The most similar image will be itself (index i), so we skip it or start from the second most similar
        sorted_indices_desc = np.argsort(similarities_to_others)[::-1] 

        num_neighbors_to_consider = 0
        for k_idx in range(len(sorted_indices_desc)): 
            neighbor_original_idx = sorted_indices_desc[k_idx]
            
            # Skip self-pairing
            if neighbor_original_idx == i:
                continue
            
            neighbor_image_id = cluster_image_ids_ordered[neighbor_original_idx]
            similarity_score = sim_matrix[i, neighbor_original_idx]
            
            if similarity_score >= similarity_threshold:
                # Add pair, ensuring a canonical order (e.g., smaller id first lexicographically) 
                # to avoid duplicates like (A,B) and (B,A) and self-loops
                if current_image_id != neighbor_image_id: # Should always be true due to previous continue
                    pair = tuple(sorted((current_image_id, neighbor_image_id)))
                    selected_pairs.add(pair)
                    if verbose: print(f"  Pair added: {pair} with similarity {similarity_score:.4f}")
            
            num_neighbors_to_consider += 1
            if num_neighbors_to_consider >= top_k: # Consider only top_k actual neighbors (excluding self)
                break 
                    
    final_pairs = list(selected_pairs)
    if verbose: print(f"Selected {len(final_pairs)} unique pairs for this cluster.")
    return final_pairs

# --- Test block ---
if __name__ == '__main__':
    print("--- Testing pair_selector.py ---")
    
    # Create dummy embeddings data similar to what DINOv2 would output
    # (dataset_name__image_filename : np.array)
    dummy_all_embeddings = {
        "datasetA__img1.png": np.random.rand(384).astype(np.float32),
        "datasetA__img2.png": np.random.rand(384).astype(np.float32),
        "datasetA__img3.png": np.random.rand(384).astype(np.float32),
        "datasetA__img4.png": np.random.rand(384).astype(np.float32),
        "datasetA__img5.png": np.random.rand(384).astype(np.float32),
        "datasetB__imgA.png": np.random.rand(384).astype(np.float32), # Different dataset
    }
    # Make some embeddings more similar to test pairing
    dummy_all_embeddings["datasetA__img2.png"] = dummy_all_embeddings["datasetA__img1.png"] * 0.95 + np.random.rand(384) * 0.05
    dummy_all_embeddings["datasetA__img3.png"] = dummy_all_embeddings["datasetA__img1.png"] * 0.90 + np.random.rand(384) * 0.10
    dummy_all_embeddings["datasetA__img5.png"] = dummy_all_embeddings["datasetA__img4.png"] * 0.92 + np.random.rand(384) * 0.08


    # Simulate a cluster of image IDs
    cluster1_image_ids = ["datasetA__img1.png", "datasetA__img2.png", "datasetA__img3.png", "datasetA__img4.png", "datasetA__img5.png"]
    
    print(f"\nTesting with cluster: {cluster1_image_ids}")
    selected_pairs_cluster1 = select_pairs_by_embedding_similarity(
        cluster1_image_ids,
        dummy_all_embeddings,
        top_k=2, # Look for top 2 neighbors for each image
        similarity_threshold=0.7, # Example threshold
        verbose=True
    )
    print("\nSelected pairs for Cluster 1:")
    for pair in selected_pairs_cluster1:
        print(pair)
    
    print(f"\nTotal pairs selected for Cluster 1: {len(selected_pairs_cluster1)}")

    # Test with a small cluster
    cluster2_image_ids = ["datasetA__img1.png", "datasetB__imgA.png"]
    print(f"\nTesting with cluster: {cluster2_image_ids}")
    selected_pairs_cluster2 = select_pairs_by_embedding_similarity(
        cluster2_image_ids,
        dummy_all_embeddings,
        top_k=2,
        similarity_threshold=0.1, # Lower threshold to get a match with random data
        verbose=True
    )
    print("\nSelected pairs for Cluster 2:")
    for pair in selected_pairs_cluster2:
        print(pair)
    print(f"\nTotal pairs selected for Cluster 2: {len(selected_pairs_cluster2)}")

    # Test with a cluster with a missing embedding
    cluster3_image_ids = ["datasetA__img1.png", "datasetA__MISSING.png", "datasetA__img2.png"]
    print(f"\nTesting with cluster (one missing embedding): {cluster3_image_ids}")
    selected_pairs_cluster3 = select_pairs_by_embedding_similarity(
        cluster3_image_ids,
        dummy_all_embeddings,
        top_k=2,
        similarity_threshold=0.7,
        verbose=True
    )
    print("\nSelected pairs for Cluster 3:")
    for pair in selected_pairs_cluster3:
        print(pair)
    print(f"\nTotal pairs selected for Cluster 3: {len(selected_pairs_cluster3)}")
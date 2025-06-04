# IMC 2025 - 3D Scene Reconstruction from Image Collections
Team: Maschinelles Lernen (Dave, Davin, Raman)

## 🚀 Project Overview

This repository contains our team's submission attempt for the [Kaggle Image Matching Challenge 2025](https://www.kaggle.com/competitions/image-matching-challenge-2025). The challenge objective was to develop machine learning algorithms to reconstruct accurate 3D scenes from messy, unstructured image collections. This involved two primary tasks:
1.  **Clustering:** Partitioning images within various datasets into distinct scenes or identifying them as outliers.
2.  **Pose Estimation:** For each identified scene, determining the camera pose (rotation matrix R, translation vector T) for every image belonging to it.

## 🛠️ Our Pipeline Approach

We designed and implemented a multi-stage pipeline to address this challenge:

1.  **Image Preprocessing (`src/data/preprocessing.py`):**
    *   Standardized image loading (ensuring RGB format).
    *   Implemented aspect-ratio-aware image resizing to handle diverse input dimensions and prepare images for feature extraction.
    *   Included basic EXIF orientation correction.

2.  **Global Feature Extraction (`src/features/global_dino_extractor.py`):**
    *   Utilized a pre-trained Vision Transformer, **DINOv2 (ViT-Small)**, to extract powerful global image embeddings (384-dim).
    *   These embeddings serve as a rich "fingerprint" for each image, capturing its overall content and style, crucial for distinguishing different scenes.
    *   Embeddings for all images were cached to an NPZ file.

3.  **Clustering (`src/clustering/hdbscan_clusterer.py`):**
    *   Applied **UMAP** for dimensionality reduction on the DINOv2 embeddings (e.g., to 30 dimensions, using cosine metric).
    *   Employed **HDBSCAN** on the reduced embeddings to perform density-based clustering, automatically identifying scene clusters and noise points (outliers).
    *   Parameters (UMAP: n_neighbors=15, n_components=30, min_dist=0.0; HDBSCAN: min_cluster_size=5) were chosen based on initial visual inspection of training data clusters.

4.  **Image Pair Selection (`src/matching_strategies/pair_selector.py`):**
    *   To optimize the local feature matching stage, this module selects promising image pairs *within each identified scene cluster*.
    *   It leverages the DINOv2 global embeddings to calculate pairwise cosine similarity between images in a cluster.
    *   Pairs are selected based on a top-K nearest neighbors strategy and a similarity threshold, significantly reducing the number of pairs for the more computationally intensive local matching.

5.  **Local Feature Matching & SfM (Conceptualized in `src/sfm/scene_reconstructor.py` - Primary development by Davin):**
    *   **Local Features:** Intended to use **ALIKED** for keypoint and descriptor extraction on selected image pairs (after resizing images to a suitable large dimension, e.g., 1024px, using `preprocessing.py`).
    *   **Matching:** Intended to use **LightGlue** (ALIKED variant) for robustly matching local features between image pairs.
        *   Rotation Test-Time Augmentation (TTA) was planned to improve matching for incorrectly oriented images.
    *   **COLMAP Database Preparation:** Logic using custom `database.py` and `h5_to_db.py` (adapted from standard utilities) was developed to ingest custom keypoints and matches into COLMAP's SQLite database format.
    *   **Structure from Motion (SfM):** Intended to use **`pycolmap.incremental_mapping`** on the prepared database for each scene cluster to perform 3D reconstruction and estimate camera poses (R, T).
    *   **Output:** R, T matrices for successfully registered images.

6.  **Submission Generation (Kaggle Notebook - `IMC2025_Submission_Pipeline.ipynb`):**
    *   Orchestrates the entire pipeline.
    *   Handles offline loading of model weights (DINOv2, ALIKED, LightGlue) from Kaggle Datasets.
    *   Calls the Python scripts from the `src/` directory.
    *   Formats the final cluster assignments and poses into the required `submission.csv` format.

## ⚙️ Setup & Usage

1.  **Environment:**
    *   A Python virtual environment (`venv`) is recommended.
    *   Install dependencies using the provided `requirements.txt`:
        ```bash
        python -m venv venv
        source venv/bin/activate # or venv\Scripts\activate on Windows
        pip install -r requirements.txt
        ```
2.  **Data:**
    *   Download the competition data from [Kaggle](https://www.kaggle.com/competitions/image-matching-challenge-2025/data).
    *   Place it in a `./data/image-matching-challenge-2025/` directory relative to the project root (or adjust paths in scripts).
3.  **Model Weights:**
    *   Pre-trained model weights for DINOv2, ALIKED, and LightGlue are expected to be provided as Kaggle Datasets when running the main submission notebook. The notebook includes logic to copy/use these offline.
4.  **Running the Pipeline (Conceptual for individual scripts):**
    *   `python src/features/global_dino_extractor.py <path_to_image_list_csv> <path_to_output_embeddings_npz> --base_image_dir <path_to_train_images>`
    *   `python src/clustering/hdbscan_clusterer.py <path_to_embeddings_npz> <path_to_output_clusters_csv> [UMAP/HDBSCAN_params...]`
    *   *(Similar command-line interfaces would be finalized for `pair_selector.py` and `scene_reconstructor.py`)*
    *   The main execution is orchestrated by `IMC2025_Submission_Pipeline.ipynb`.

## 📚 Learnings & Challenges

*   **EDA is Crucial:** Thorough EDA revealed significant variations in image dimensions, common outlier characteristics (rotations, foliage, silhouettes), and the need for robust preprocessing.
*   **Modular Design:** Developing the pipeline in modular Python scripts (`src/`) proved effective for organization and individual contributions.
*   **Offline Execution in Kaggle:** Managing dependencies and model weights for Kaggle's internet-off submission environment is a critical step. We planned for this using Kaggle Datasets for model weights and discussed strategies for `src` code and pip packages.
*   **Pair Selection:** Realized early that all-vs-all local matching is computationally prohibitive. Implementing an intelligent pair selector based on global embeddings (DINOv2) was a key optimization strategy.
*   **COLMAP Integration:** Interfacing custom features/matches with COLMAP requires careful preparation of its database format. Davin's work on `database.py` and `h5_to_db.py` addressed this.
*   **Time Constraints:** As with any timed competition, prioritizing and integrating complex components under pressure was a significant challenge. While we developed all core modules, final end-to-end integration and debugging for submission within the time limit proved difficult.

## 퓨 Future Work & Potential Improvements

*   **Full Integration & Debugging of Davin's SfM Module:** Completing the refactoring of the local matching and `pycolmap` execution into a robust, callable `src/sfm/scene_reconstructor.py` and fully integrating it into the main pipeline notebook.
*   **Rotation Test-Time Augmentation (TTA):** Implementing TTA during local feature matching (ALIKED+LightGlue) to improve robustness to rotated images.
*   **COLMAP Parameter Tuning (Raman):** Leveraging Raman's research to optimize `pycolmap.IncrementalPipelineOptions` for better reconstruction quality and robustness across different scene types.
*   **Local Cross-Validation:** Implementing a robust local CV strategy to measure clustering performance (e.g., ARI, NMI) and pose accuracy (approximating mAA) using the training data.
*   **Advanced Outlier Handling:** Beyond HDBSCAN's noise points, explore heuristics based on EDA or SfM consistency to further refine outlier detection.
*   **Optimizing `pair_selector.py`:** Experiment with different `top_K` and `similarity_threshold` values.
*   **Packaging for Kaggle:** Transition from `!git clone` to using a Kaggle Dataset for the `src/` code and potentially for pip dependencies (`.whl` files) for maximum submission robustness.

## 🏆 Competition Outcome

Due to last-minute integration challenges and submission errors related to the offline environment execution, we were unable to achieve a scored submission on the leaderboard. However, the project involved significant learning and development of a comprehensive pipeline.

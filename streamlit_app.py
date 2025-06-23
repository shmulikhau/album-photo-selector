import streamlit as st
import os
import shutil
import logging
from PIL import Image
import numpy as np
import torch
from time import time
from src.image_clusterer.clusterer import Clusterer
from src.image_clusterer.k_means import KMeansClusterer
from src.image_clusterer.distance import DistanceClusterer
from src.image_embedder.jinaclip_embedder import JinaClipEmbedder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

use_gpu = os.getenv("USE_GPU", "False").upper() == "TRUE"
tmp_dir = os.getenv("TMP_DIR", '/tmp/image_selector')

# === clustering logic ===
def get_clusterer(algorithm, data):
    if algorithm == "KMeans":
        return KMeansClusterer(1024, data=data, gpu=use_gpu)
    elif algorithm == "DBSCAN":
        return DistanceClusterer(1024, data=data, gpu=use_gpu)
    else:
        raise Exception("algorithm not chosen!")
    
@st.cache_data(show_spinner=False)
def embed_images(image_paths):
    logger.info('Embed_images...')
    embedder = JinaClipEmbedder(gpu=use_gpu)
    batch_size = int(os.getenv("BATCH_SIZE", "1"))

    embeddings = torch.zeros(0,1024).to('cuda' if use_gpu else 'cpu')
    batch = []
    for i in range(len(image_paths)):
        if i % batch_size == 0 and len(batch):
            features = embedder.get_image_embeddings(batch)
            embeddings = torch.cat((embeddings, torch.tensor(features).to('cuda' if use_gpu else 'cpu')), dim=0)
            batch = []
        batch.append(Image.open(image_paths[i]))
    if len(batch):
        features = embedder.get_image_embeddings(batch)
        embeddings = torch.cat((embeddings, torch.tensor(features).to('cuda' if use_gpu else 'cpu')), dim=0)
    logger.info('Finish embed images')
    return embeddings

@st.cache_data(show_spinner=False)
def get_clusters(image_paths, algorithm, algorithm_value, _data):
    start_time = time()
    clusterer = get_clusterer(algorithm, _data)
    if algortihm_value is None:
        raise Exception("algorithm not chosen!")
    clusterer.make_cluster(algorithm_value)
    
    clusters = {}
    
    for i, c in enumerate(clusterer.vectors2kernels.tolist()):
        if c not in clusters.keys():
            clusters[c] = []
        clusters[c].append(image_paths[i])

    clusters = {key:clusters[key] for key in sorted(clusters.keys())}

    logger.info(f'Finish clustering in {time() - start_time}ms')
    return clusters

# === Streamlit App ===

# Save uploaded files to temp directory
@st.cache_data(show_spinner=False)
def upload_files(uploaded_files):
    try:
        shutil.rmtree(tmp_dir)
    except:
        pass
    os.mkdir(tmp_dir)
    
    image_paths = []
    for uploaded_file in uploaded_files:
        img_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_paths.append(img_path)
    return image_paths

st.set_page_config(page_title="Image Clustering for Album", layout="wide")
st.title("📸 Image Clustering & Selection Interface for Album Selection")

# Upload Images
uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} images uploaded.")

    image_paths = upload_files(uploaded_files)

    # Run Embedder Model
    data = embed_images(image_paths)

    # Select Algortihm
    algorithm = st.selectbox("Select Clustering Algorithm", ("KMeans", "DBSCAN"))
    algortihm_value = None
    if algorithm == "KMeans":
        algortihm_value = st.slider("Number of clusters", min_value=2, max_value=500, value=3)
    elif algorithm == "DBSCAN":
        algortihm_value = st.slider("DBSCAN Epsilon (eps)", min_value=-1., max_value=1., value=.5, step=1e-2)

    # Cluster Images
    clusters = get_clusters(image_paths, algorithm, algortihm_value, data)
    logger.info(clusters)

    # Display Clusters and Deletion UI
    st.subheader("🗂️ Review and delete unwanted images")
    images_to_keep = []

    for cluster_id, paths in clusters.items():
        st.markdown(f"### Cluster {cluster_id + 1}")
        cols = st.columns(4)
        for i, img_path in enumerate(paths):
            col = cols[i % 4]
            with col:
                img = Image.open(img_path)
                st.image(img, caption=os.path.basename(img_path), use_container_width=True)
                keep = st.checkbox(f"Keep", value=True, key=f"{cluster_id}_{i}")
                if keep:
                    images_to_keep.append(img_path)

    # Export/Save Kept Images
    if st.button("Save Selected Images"):
        output_dir = os.path.join(tmp_dir, "selected_images")
        os.makedirs(output_dir, exist_ok=True)

        for img_path in images_to_keep:
            shutil.copy(img_path, os.path.join(output_dir, os.path.basename(img_path)))
        shutil.make_archive(tmp_dir + '/photos', 'zip', output_dir)

        with open(tmp_dir + '/photos.zip', "rb") as file:
            if st.download_button(
                    label="Download zip",
                    data=file,
                    file_name="photos.zip"
                ):
                st.success(f"{len(images_to_keep)} images downloaded :)")

else:
    st.info("Please upload images to get started.")
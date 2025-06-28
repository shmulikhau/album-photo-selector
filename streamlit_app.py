import streamlit as st
import os
import gc
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
    logger.info('Embed images...')
    embed_status = st.progress(0, text="Extracting embeddings...")
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
        embed_status.progress(int((i+1)*100/len(image_paths)), text="Extracting embeddings...")
    if len(batch):
        features = embedder.get_image_embeddings(batch)
        embeddings = torch.cat((embeddings, torch.tensor(features).to('cuda' if use_gpu else 'cpu')), dim=0)
    embedder = None
    gc.collect()
    logger.info('Finish embed images')
    embed_status.empty()
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

@st.cache_data(show_spinner=False)
def open_image(path):
    return Image.open(path)

def unselect_all_checkbox(startwith):
    for key in st.session_state.keys():
        if key.startswith(startwith):
            st.session_state[key] = False

st.set_page_config(page_title="Image Clustering for Album", layout="wide")
st.title("📸 Image Clustering & Selection Interface for Album Selection")

# Upload Images
uploaded_files = st.file_uploader("Upload minimum 2 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) >= 2:
    st.success(f"{len(uploaded_files)} images uploaded.")

    image_paths = upload_files(uploaded_files)

    # Run Embedder Model
    data = embed_images(image_paths)

    # Select Algortihm
    algorithm = st.selectbox("Select Clustering Algorithm", ("KMeans", "DBSCAN"))
    algortihm_value = None
    if algorithm == "KMeans":
        algortihm_value = st.slider("Number of clusters", min_value=2, max_value=len(image_paths), value=2)
    elif algorithm == "DBSCAN":
        algortihm_value = st.slider("DBSCAN Epsilon (eps)", min_value=-1., max_value=1., value=.5, step=1e-2)

    # Cluster Images
    clusters = get_clusters(image_paths, algorithm, algortihm_value, data)
    logger.info(clusters)

    # Display Clusters and Deletion UI
    st.subheader("🗂️ Review and delete unwanted images")
    st.session_state.images_to_chat = {}
    if "deleted_imgs" not in st.session_state:
        st.session_state.deleted_imgs = []

    for cluster_id, paths in clusters.items():
        st.markdown(f"### Cluster {cluster_id + 1}")
        i = 0
        for img_path in paths:
            if img_path in st.session_state.deleted_imgs:
                continue
            if i % 4 == 0:
                cols = st.columns(4)
            col = cols[i % 4]
            with col:
                img = open_image(img_path)
                st.image(img, caption=os.path.basename(img_path), use_container_width=True)
                img_cols = st.columns(2)
                with img_cols[0]:
                    if st.checkbox(f"to chat 💬", value=False, key=f"cb_chat_{cluster_id}_{i}"):
                        st.session_state.images_to_chat[img_path] = None
                    elif img_path in st.session_state.images_to_chat.keys():
                        del st.session_state.images_to_chat[img_path]
                with img_cols[1]:
                    if st.button("Delete", icon="❌", key=f'bt_delete_{cluster_id}_{i}'):
                        st.session_state.deleted_imgs.append(img_path)
                        st.rerun()
            i += 1

    # Export/Save Kept Images
    if st.button("Save Selected Images"):
        output_dir = os.path.join(tmp_dir, "selected_images")
        try:
            shutil.rmtree(output_dir)
        except:
            pass
        os.makedirs(output_dir, exist_ok=True)

        for img_path in image_paths:
            if img_path not in st.session_state.deleted_imgs:
                shutil.copy(img_path, os.path.join(output_dir, os.path.basename(img_path)))
        if os.path.isfile(tmp_dir + '/photos.zip'):
            os.remove(tmp_dir + '/photos.zip')
        shutil.make_archive(tmp_dir + '/photos', 'zip', output_dir)

        with open(tmp_dir + '/photos.zip', "rb") as file:
            if st.download_button(
                    label="Download zip",
                    data=file,
                    file_name="photos.zip"
                ):
                st.success(f"{len(image_paths) - len(st.session_state.deleted_imgs)} images downloaded :)")

    # chat interface
    with st.sidebar:
        st.button('Unselect all images', on_click=lambda: unselect_all_checkbox('cb_chat_'))
        st.title("Chat with Qwen2.5-vl")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.markdown(content["text"])
                    elif content["type"] == "image":
                        img_path = content["image"][8:]
                        st.image(open_image(img_path), caption=os.path.basename(img_path), use_container_width=True)

        def chat_input_onsubmit():
            st.session_state.chat_images = list(st.session_state.images_to_chat.keys())
            unselect_all_checkbox('cb_chat_')
        # Accept user input
        if prompt := st.chat_input("What is up?", on_submit=chat_input_onsubmit):
            # Convert images to chat
            image_prompt = []
            for i, path in enumerate(st.session_state.chat_images):
                image_prompt.append({"type": "text", "text": f"image {i+1}:"})
                image_prompt.append({"type": "image", "image": f"file:///{path}"})
            del st.session_state.chat_images
            # Add user message to chat history
            st.session_state.messages.append(
                {
                    "role": "user", 
                    "content": [
                        *image_prompt,
                        {"type": "text", "text": prompt}
                    ]
                }
            )
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Display assistant response in chat message container
            # with st.chat_message("assistant"):
            #     stream = client.chat.completions.create(
            #         model=st.session_state["openai_model"],
            #         messages=[
            #             {"role": m["role"], "content": m["content"]}
            #             for m in st.session_state.messages
            #         ],
            #         stream=True,
            #     )
            #     response = st.write_stream(stream)
            # st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

else:
    st.info("Please upload images to get started.")

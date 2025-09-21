import streamlit as st
import os
import sys
import time
import asyncio
import torch
import nest_asyncio
import tempfile
from loguru import logger
from pathlib import Path
from typing import List, Tuple
import io
import zipfile
from glob import glob

# Apply patch
nest_asyncio.apply()

# Add project root
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import DEFAULT_ALPHA, MAX_BATCH_SIZE, DEFAULT_BATCH_SIZE, OUTPUT_DIR, DEVICE
from main import pipeline
from core.models import load_models


def chunker(lst: List, size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def cleanup(paths: List[str]):
    for path in paths:
        try:
            os.remove(path)
            logger.debug(f"Deleted temporary file: {path}")
        except Exception as e:
            logger.warning(f"Failed to delete {path}: {e}")

def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache.")


@st.cache_resource
def get_models():
    """
    Loads and caches the models. This will only run ONCE.
    """
    logger.info("Loading models for the first time...")
    models = load_models()
    logger.info("Models loaded and cached successfully.")
    return models


def run_pipeline_sync(batch: List[Tuple[str, str, str, float]], models: tuple) -> None:
    """Run the async pipeline, now passing in the models."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(pipeline(batch, models))
    loop.close()

def save_temp_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    logger.debug(f"Saved uploaded file to temp path: {tmp_path}")
    return tmp_path

def setup_output_dir(path: str):
    os.makedirs(path, exist_ok=True)

def create_zip(images: List[str]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for img_path in images:
            img_name = os.path.basename(img_path)
            zf.write(img_path, arcname=img_name)
    zip_buffer.seek(0)
    return zip_buffer.read()

def main():
    st.set_page_config(page_title="NST", layout="wide")
    st.title("Region-aware Neural Style Transfer")

    # --- LOAD MODELS ONCE ---
    models = get_models()


    with st.sidebar:
        # ...
        st.header("Settings")
        style_file = st.file_uploader("1. Upload Style Image", type=["png", "jpg", "jpeg"])
        st.markdown("---")
        batch_size = st.number_input("Max images processed per batch", min_value=1, max_value=MAX_BATCH_SIZE, value=DEFAULT_BATCH_SIZE, step=1)

    tab1, tab2 = st.tabs(["Style Transfer", "Results"])

    if "results" not in st.session_state:
        st.session_state["results"] = []

    with tab1:
        
        col1, col2 = st.columns([2, 1])
        with col1:
            content_files = st.file_uploader("2. Upload Content Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
            start_button = st.button("Start Style Transfer", disabled=not (style_file and content_files))
        
        alphas = []
        with col2:
            if content_files:
                with st.expander("3. Style Strength per Image (optional)"):
                    for idx, file in enumerate(content_files):
                        alpha = st.slider(f"Style strength for {file.name}", 0.0, 1.0, DEFAULT_ALPHA, 0.1, key=f"alpha_{idx}")
                        alphas.append(alpha)

        if start_button:
            st.session_state.clear() # Clear old state
            st.session_state["results"] = []
            st.session_state["current_batch"] = 0
            st.session_state["style_path"] = save_temp_file(style_file)
            st.session_state["content_paths"] = [save_temp_file(f) for f in content_files]
            
            setup_output_dir(OUTPUT_DIR)
            timestamp = int(time.time())
            st.session_state["image_jobs"] = [
                (content_path, st.session_state["style_path"], os.path.join(OUTPUT_DIR, f"output_{timestamp}_{Path(content_path).stem}.png"), alpha)
                for content_path, alpha in zip(st.session_state["content_paths"], alphas)
            ]
            st.rerun()

        if "image_jobs" in st.session_state and st.session_state.get("current_batch") is not None:
            image_jobs = st.session_state["image_jobs"]
            total_batches = (len(image_jobs) + batch_size - 1) // batch_size
            current_batch = st.session_state["current_batch"]

            progress_bar = st.progress(0)
            status_text = st.empty()

            if current_batch < total_batches:
                batch = list(chunker(image_jobs, batch_size))[current_batch]
                with st.spinner(f"Processing batch {current_batch + 1} of {total_batches}..."):
                    try:
                        
                        run_pipeline_sync(batch, models)
                        st.session_state["results"].extend([(True, job[2]) for job in batch])
                    except Exception as e:
                        logger.error(f"Batch {current_batch + 1} failed: {e}")
                        st.session_state["results"].extend([(False, str(e)) for _ in batch])
                    
                    cleanup([job[0] for job in batch])
                    clear_gpu()

                progress_bar.progress((current_batch + 1) / total_batches)
                status_text.info(f"Completed batch {current_batch + 1} of {total_batches}")

                st.session_state["current_batch"] += 1
                st.rerun()

            else:
                cleanup([st.session_state.get("style_path", "")])
                st.success("Style Transfer Completed!")
                
                for key in ["image_jobs", "current_batch", "style_path", "content_paths"]:
                    if key in st.session_state:
                        del st.session_state[key]
    
    with tab2:
        st.subheader("Results")

        output_images = sorted(glob(os.path.join(OUTPUT_DIR, "output_*.png")))

        if not output_images:
            st.info("No results to show yet. Perform style transfer first in the 'Style Transfer' tab.")
        else:
            zip_bytes = create_zip(output_images)
            st.download_button(
                label="📥 Download All Stylized Images (ZIP)",
                data=zip_bytes,
                file_name="stylized_images.zip",
                mime="application/zip",
            )

            if st.button("🗑️ Clear All Results"):
                for f in output_images:
                    try:
                        os.remove(f)
                    except Exception as e:
                        logger.warning(f"Failed to delete {f}: {e}")
                st.success("All results cleared.")
                st.rerun()

            cols = st.columns(2)
            for idx, image_path in enumerate(output_images):
                with cols[idx % 2]:
                    st.image(image_path, caption=f"Stylized Image {idx + 1}")
                    with open(image_path, "rb") as f:
                        st.download_button(
                            label="Download",
                            data=f.read(),
                            file_name=os.path.basename(image_path),
                            mime="image/png",
                            key=f"download_{idx}",
                        )


if __name__ == "__main__":
    main()

import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from rust_analyzer import analyze_rust_bgr

st.set_page_config(page_title="Rust Quantification", layout="wide")

# --- Logo + title header ---
if os.path.exists("logo.png"):
    c1, c2 = st.columns([1, 6])
    with c1:
        st.image("logo.png", use_container_width=True)
    with c2:
        st.title("Rust Quantification (Upload Photo)")
else:
    st.title("Rust Quantification (Upload Photo)")
# --- end header ---

uploaded = st.file_uploader("Upload a photo (PNG/JPG)", type=["png", "jpg", "jpeg"])

with st.sidebar:
    st.header("Settings")
    exclude_shadows = st.checkbox("Exclude dark/shadow pixels", value=True)
    min_v = st.slider("Shadow threshold (V)", 0, 255, 35, 1)

    st.subheader("Cleanup")
    k = st.slider("Kernel size", 1, 21, 5, 2)
    open_iters = st.slider("Open iterations", 0, 5, 1, 1)
    close_iters = st.slider("Close iterations", 0, 5, 2, 1)

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    rust_pct, rust_pixels, valid_pixels, mask, overlay_bgr = analyze_rust_bgr(
        img_bgr,
        exclude_dark_pixels=exclude_shadows,
        min_v_for_valid=min_v,
        kernel_size=k,
        open_iters=open_iters,
        close_iters=close_iters,
    )

    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(img_rgb, use_container_width=True)

    with col2:
        st.subheader("Rust Overlay (red)")
        st.image(overlay_rgb, use_container_width=True)

    st.markdown("### Results")
    st.write(f"**Estimated rust area:** {rust_pct:.2f}%")
    st.write(f"Rust pixels: {rust_pixels:,}")
    st.write(f"Valid pixels (analysis area): {valid_pixels:,}")

    st.subheader("Rust Mask (white = rust)")
    st.image(mask, clamp=True, use_container_width=True)
else:
    st.info("Upload a photo to calculate rust % and see the rust mask/overlay.")


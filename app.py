import os
import io
from datetime import datetime, date

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import openpyxl  # for extracting embedded images from Excel

from rust_analyzer import analyze_rust_bgr

# ============================================================
# Page setup + header (logo)
# ============================================================
st.set_page_config(page_title="Rust Quantification (One Report)", layout="wide")

if os.path.exists("logo.png"):
    c1, c2 = st.columns([1, 6])
    with c1:
        st.image("logo.png", use_container_width=True)
    with c2:
        st.title("Rust Quantification (One Report)")
else:
    st.title("Rust Quantification (One Report)")

st.caption("Upload either an Excel maintenance report (.xlsx) with embedded photos OR a single photo. Output is ONE combined report.")
st.divider()

# ============================================================
# Helpers
# ============================================================
def get_severity(rust_pct: float, minor_thr: float, moderate_thr: float) -> str:
    if rust_pct < minor_thr:
        return "Minor"
    if rust_pct < moderate_thr:
        return "Moderate"
    return "Severe"

def extract_images_from_excel(xlsx_bytes: bytes):
    """
    Extract embedded images from all sheets in an Excel workbook.
    Returns list of tuples: [(sheet_name, row_1based_or_None, col_1based_or_None, img_bytes), ...]
    NOTE: Works for embedded images. If images are linked (not embedded), Excel may not contain them.
    """
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
    extracted = []

    for ws in wb.worksheets:
        imgs = getattr(ws, "_images", [])
        for img in imgs:
            r = None
            c = None
            anc = img.anchor
            if hasattr(anc, "_from"):
                r = anc._from.row + 1
                c = anc._from.col + 1

            img_bytes = img._data()  # bytes of embedded image
            extracted.append((ws.title, r, c, img_bytes))

    return extracted

# ============================================================
# Sidebar: analysis settings
# ============================================================
with st.sidebar:
    st.header("Analysis Settings")

    exclude_shadows = st.checkbox("Exclude dark / shadow pixels", value=True)
    min_v = st.slider("Shadow threshold (V)", 0, 255, 35, 1)

    st.subheader("Mask cleanup")
    kernel_size = st.slider("Kernel size", 1, 21, 5, 2)
    open_iters = st.slider("Open iterations", 0, 5, 1, 1)
    close_iters = st.slider("Close iterations", 0, 5, 2, 1)

    st.subheader("Rust severity thresholds (%)")
    minor_thr = st.number_input("Minor < (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    moderate_thr = st.number_input("Moderate < (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)

# ============================================================
# 🆕 NEW INSPECTION (ONE REPORT) — ONLY SECTION
# ============================================================
st.subheader("New Inspection (One Report)")

with st.form("inspection_form"):
    a, b, c = st.columns(3)
    with a:
        inspection_date = st.date_input("Inspection Date", value=date.today())
        vessel_name = st.text_input("Vessel Name", placeholder="e.g., SUMATERA EXPRESS")
    with b:
        tank_no = st.text_input("Tank / Hold No.", placeholder="e.g., WB Tank P/S, Hold 1")
        location = st.text_input("Location / Area", placeholder="e.g., Ballast Tank, Cargo Hold")
    with c:
        inspector = st.text_input("Inspector", placeholder="Name / Rank")
        remarks = st.text_area("Remarks (optional)", height=80)

    st.info(
        "✅ Choose ONE option below:\n\n"
        "• Option A: Upload Excel report (.xlsx) with embedded photos → analyzed as ONE combined report.\n"
        "• Option B: Upload a single photo (PNG/JPG/JPEG) → analyzed as ONE report.\n\n"
        "⚠️ Do NOT upload both."
    )

    uploaded_excel_report = st.file_uploader(
        "Option A: Upload Excel report with embedded photos (.xlsx)",
        type=["xlsx"],
        key="excel_report_uploader",
    )

    st.info(
        "📸 Option B: Upload rust PHOTO only.\n\n"
        "Accepted formats: PNG / JPG / JPEG.\n\n"
        "📄 Do NOT upload Excel files here."
    )

    uploaded_photo = st.file_uploader(
        "Option B: Upload rust photo (PNG / JPG / JPEG)",
        type=["png", "jpg", "jpeg"],
        key="photo_uploader",
    )

    submitted = st.form_submit_button("Analyze (One Report)")

# ============================================================
# Run analysis
# ============================================================
if submitted:
    # Validate choice
    if uploaded_excel_report is None and uploaded_photo is None:
        st.error("Please upload either an Excel report (.xlsx) OR a photo (PNG/JPG/JPEG).")
        st.stop()

    if uploaded_excel_report is not None and uploaded_photo is not None:
        st.error("Please upload only ONE: Excel report OR Photo (not both).")
        st.stop()

    # Outputs (one report)
    rust_pct_total = 0.0
    rust_pixels_total = 0
    valid_pixels_total = 0
    severity = "Minor"

    # -----------------------------
    # CASE A: Excel report (all photos combined as ONE report)
    # -----------------------------
    if uploaded_excel_report is not None:
        xlsx_bytes = uploaded_excel_report.getvalue()
        extracted = extract_images_from_excel(xlsx_bytes)

        if len(extracted) == 0:
            st.error(
                "No embedded photos found in the Excel file.\n\n"
                "Please ensure photos are INSERTED/EMBEDDED in the sheet (not only linked)."
            )
            st.stop()

        per_photo_rows = []

        with st.spinner("Analyzing embedded photos from Excel..."):
            for idx, (sheet, r, c, img_bytes) in enumerate(extracted, start=1):
                try:
                    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    continue

                img_rgb = np.array(img_pil)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                rust_pct, rust_pixels, valid_pixels, mask, overlay_bgr = analyze_rust_bgr(
                    img_bgr,
                    exclude_dark_pixels=exclude_shadows,
                    min_v_for_valid=min_v,
                    kernel_size=kernel_size,
                    open_iters=open_iters,
                    close_iters=close_iters,
                )

                rust_pixels_total += int(rust_pixels)
                valid_pixels_total += int(valid_pixels)

                per_photo_rows.append({
                    "photo_no": idx,
                    "sheet": sheet,
                    "row": r,
                    "col": c,
                    "rust_%": round(float(rust_pct), 2),
                    "rust_pixels": int(rust_pixels),
                    "valid_pixels": int(valid_pixels),
                })

        if valid_pixels_total == 0:
            st.error("Could not compute valid pixels from extracted images. Please check the photos and try again.")
            st.stop()

        rust_pct_total = 100.0 * rust_pixels_total / max(valid_pixels_total, 1)
        severity = get_severity(float(rust_pct_total), float(minor_thr), float(moderate_thr))

        st.markdown("## Excel Report Results (All Photos Combined)")
        st.write(f"**Total embedded photos found:** {len(extracted)}")
        st.write(f"**TOTAL Rust Area:** {rust_pct_total:.2f}%")
        st.write(f"**Overall Severity:** {severity}")
        st.write(f"Total rust pixels: {rust_pixels_total:,}")
        st.write(f"Total valid pixels: {valid_pixels_total:,}")

        st.markdown("### Per-photo breakdown")
        if len(per_photo_rows) == 0:
            st.warning("Photos were detected, but none could be decoded. Please re-insert images and try again.")
        else:
            st.dataframe(pd.DataFrame(per_photo_rows), use_container_width=True)

    # -----------------------------
    # CASE B: Single photo (one report)
    # -----------------------------
    else:
        img_pil = Image.open(uploaded_photo).convert("RGB")
        img_rgb = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        rust_pct, rust_pixels, valid_pixels, mask, overlay_bgr = analyze_rust_bgr(
            img_bgr,
            exclude_dark_pixels=exclude_shadows,
            min_v_for_valid=min_v,
            kernel_size=kernel_size,
            open_iters=open_iters,
            close_iters=close_iters,
        )

        rust_pct_total = float(rust_pct)
        rust_pixels_total = int(rust_pixels)
        valid_pixels_total = int(valid_pixels)
        severity = get_severity(float(rust_pct_total), float(minor_thr), float(moderate_thr))

        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Original Photo")
            st.image(img_rgb, use_container_width=True)
        with col2:
            st.markdown("### Rust Overlay")
            st.image(overlay_rgb, use_container_width=True)

        st.markdown("### Rust Mask (white = rust)")
        st.image(mask, clamp=True, use_container_width=True)

        st.markdown("## Results")
        st.write(f"**Rust Area:** {rust_pct_total:.2f}%")
        st.write(f"**Severity:** {severity}")
        st.write(f"Rust pixels: {rust_pixels_total:,}")
        st.write(f"Valid pixels: {valid_pixels_total:,}")

    # -----------------------------
    # Final Summary (common for both cases)
    # -----------------------------
    st.divider()
    st.subheader("Final One-Report Summary")
    st.write(f"**Vessel:** {vessel_name.strip() or 'N/A'}")
    st.write(f"**Tank / Hold:** {tank_no.strip() or 'N/A'}")
    st.write(f"**Location:** {location.strip() or 'N/A'}")
    st.write(f"**Inspection Date:** {inspection_date}")
    st.write(f"**Inspector:** {inspector.strip() or 'N/A'}")
    if remarks.strip():
        st.write(f"**Remarks:** {remarks.strip()}")

    st.markdown("### Total Rust (Combined)")
    st.write(f"**TOTAL Rust Area:** {rust_pct_total:.2f}%")
    st.write(f"**Overall Severity:** {severity}")
    st.write(f"Total rust pixels: {rust_pixels_total:,}")
    st.write(f"Total valid pixels: {valid_pixels_total:,}")

    st.success("✅ One report completed.")


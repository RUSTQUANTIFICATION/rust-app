import os
import io
from datetime import datetime, date

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import openpyxl

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

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

st.caption(
    "Upload either an Excel maintenance report (.xlsx) with embedded photos "
    "OR a single photo. Output is ONE combined report."
)
st.divider()

# ============================================================
# Helpers
# ============================================================
def get_severity(rust_pct, minor_thr, moderate_thr):
    if rust_pct < minor_thr:
        return "Minor"
    if rust_pct < moderate_thr:
        return "Moderate"
    return "Severe"

def extract_images_from_excel(xlsx_bytes):
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
    extracted = []
    for ws in wb.worksheets:
        for img in getattr(ws, "_images", []):
            img_bytes = img._data()
            extracted.append((ws.title, img_bytes))
    return extracted

def generate_pdf(report_meta, totals, per_photo_df=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Rust Inspection Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    for k, v in report_meta.items():
        story.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))

    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
    story.append(Paragraph(f"Total Rust Area: {totals['rust_pct']:.2f}%", styles["Normal"]))
    story.append(Paragraph(f"Severity: {totals['severity']}", styles["Normal"]))
    story.append(Paragraph(f"Rust Pixels: {totals['rust_pixels']:,}", styles["Normal"]))
    story.append(Paragraph(f"Valid Pixels: {totals['valid_pixels']:,}", styles["Normal"]))

    if per_photo_df is not None and not per_photo_df.empty:
        story.append(Spacer(1, 16))
        story.append(Paragraph("<b>Per-Photo Breakdown</b>", styles["Heading2"]))

        table_data = [per_photo_df.columns.tolist()] + per_photo_df.values.tolist()
        tbl = Table(table_data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(tbl)

    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================================
# Sidebar: analysis settings
# ============================================================
with st.sidebar:
    st.header("Analysis Settings")
    exclude_shadows = st.checkbox("Exclude dark / shadow pixels", True)
    min_v = st.slider("Shadow threshold (V)", 0, 255, 35)
    kernel_size = st.slider("Kernel size", 1, 21, 5, 2)
    open_iters = st.slider("Open iterations", 0, 5, 1)
    close_iters = st.slider("Close iterations", 0, 5, 2)
    minor_thr = st.number_input("Minor < (%)", value=5.0)
    moderate_thr = st.number_input("Moderate < (%)", value=15.0)

# ============================================================
# New Inspection (ONE REPORT)
# ============================================================
st.subheader("New Inspection (One Report)")

with st.form("inspection_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        inspection_date = st.date_input("Inspection Date", date.today())
        vessel_name = st.text_input("Vessel Name")
    with col2:
        tank_no = st.text_input("Tank / Hold No.")
        location = st.text_input("Location")
    with col3:
        inspector = st.text_input("Inspector")
        remarks = st.text_area("Remarks", height=80)

    st.info(
        "Option A: Excel (.xlsx) with embedded photos\n"
        "Option B: Single photo (PNG/JPG/JPEG)\n\n"
        "Upload ONLY ONE."
    )

    uploaded_excel = st.file_uploader("Option A – Excel Report", type=["xlsx"])
    uploaded_photo = st.file_uploader("Option B – Photo", type=["png", "jpg", "jpeg"])

    submitted = st.form_submit_button("Analyze")

# ============================================================
# Analysis
# ============================================================
if submitted:
    if not uploaded_excel and not uploaded_photo:
        st.error("Upload Excel OR Photo.")
        st.stop()
    if uploaded_excel and uploaded_photo:
        st.error("Upload only ONE option.")
        st.stop()

    rust_pixels_total = 0
    valid_pixels_total = 0
    per_photo_rows = []

    if uploaded_excel:
        images = extract_images_from_excel(uploaded_excel.getvalue())
        if not images:
            st.error("No embedded images found.")
            st.stop()

        for idx, (sheet, img_bytes) in enumerate(images, 1):
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            rust_pct, rust_px, valid_px, *_ = analyze_rust_bgr(
                img_bgr,
                exclude_dark_pixels=exclude_shadows,
                min_v_for_valid=min_v,
                kernel_size=kernel_size,
                open_iters=open_iters,
                close_iters=close_iters,
            )
            rust_pixels_total += rust_px
            valid_pixels_total += valid_px
            per_photo_rows.append({
                "Photo": idx,
                "Sheet": sheet,
                "Rust %": round(rust_pct, 2)
            })

    else:
        img = Image.open(uploaded_photo).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        rust_pct, rust_pixels_total, valid_pixels_total, *_ = analyze_rust_bgr(
            img_bgr,
            exclude_dark_pixels=exclude_shadows,
            min_v_for_valid=min_v,
            kernel_size=kernel_size,
            open_iters=open_iters,
            close_iters=close_iters,
        )

    rust_pct_total = 100 * rust_pixels_total / max(valid_pixels_total, 1)
    severity = get_severity(rust_pct_total, minor_thr, moderate_thr)

    st.success(f"TOTAL Rust: {rust_pct_total:.2f}% | Severity: {severity}")

    report_meta = {
        "Vessel": vessel_name,
        "Tank / Hold": tank_no,
        "Location": location,
        "Inspection Date": inspection_date,
        "Inspector": inspector,
        "Remarks": remarks or "-"
    }

    totals = {
        "rust_pct": rust_pct_total,
        "severity": severity,
        "rust_pixels": rust_pixels_total,
        "valid_pixels": valid_pixels_total,
    }

    per_photo_df = pd.DataFrame(per_photo_rows) if per_photo_rows else None

    pdf_bytes = generate_pdf(report_meta, totals, per_photo_df)

    st.download_button(
        "📄 Download Inspection Report (PDF)",
        data=pdf_bytes,
        file_name=f"rust_report_{vessel_name or 'inspection'}.pdf",
        mime="application/pdf"
    )

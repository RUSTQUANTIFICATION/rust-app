import os
import io
from datetime import date

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import openpyxl

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

from rust_analyzer import analyze_rust_bgr

# ============================================================
# Page setup + header
# ============================================================
st.set_page_config(page_title="Rust Quantification – One Report", layout="wide")

if os.path.exists("logo.png"):
    c1, c2 = st.columns([1, 6])
    with c1:
        st.image("logo.png", use_container_width=True)
    with c2:
        st.title("Rust Quantification – One Report")
else:
    st.title("Rust Quantification – One Report")

st.caption(
    "Option A: Excel (.xlsx) with embedded photos → ONE report. "
    "Option B: Upload one or multiple photos (PNG/JPG/JPEG) → ONE report. "
    "PDF includes thumbnails (Original / Overlay / Mask)."
)
st.divider()

MAX_PHOTOS_IN_PDF = 30  # hard cap as requested

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
    out = []
    for ws in wb.worksheets:
        for img in getattr(ws, "_images", []):
            out.append((ws.title, img._data()))
    return out

def pil_to_png_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def make_thumb(pil_img, max_w=900):
    if pil_img.width <= max_w:
        return pil_img
    scale = max_w / pil_img.width
    return pil_img.resize(
        (int(pil_img.width * scale), int(pil_img.height * scale)),
        Image.LANCZOS
    )

def generate_pdf_with_thumbnails(report_meta, totals, per_photo_rows, photo_panels):
    buf = io.BytesIO()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=28, leftMargin=28)
    story = []

    story.append(Paragraph("<b>Rust Inspection Report</b>", styles["Title"]))
    story.append(Spacer(1, 8))

    for k, v in report_meta.items():
        story.append(Paragraph(f"<b>{k}:</b> {v or '-'}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
    story.append(Paragraph(f"Total Rust Area: <b>{totals['rust_pct']:.2f}%</b>", styles["Normal"]))
    story.append(Paragraph(f"Severity: <b>{totals['severity']}</b>", styles["Normal"]))
    story.append(Paragraph(f"Rust Pixels: {totals['rust_pixels']:,}", styles["Normal"]))
    story.append(Paragraph(f"Valid Pixels: {totals['valid_pixels']:,}", styles["Normal"]))
    story.append(Spacer(1, 12))

    if per_photo_rows:
        story.append(Paragraph("<b>Per-Photo Breakdown</b>", styles["Heading2"]))
        df = pd.DataFrame(per_photo_rows)
        tbl = Table([df.columns.tolist()] + df.values.tolist(), repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Photo Evidence</b>", styles["Heading2"]))
    story.append(Paragraph("Overlay: rust in red | Mask: white = rust", styles["Normal"]))
    story.append(Spacer(1, 10))

    capped = photo_panels[:MAX_PHOTOS_IN_PDF]
    for i, p in enumerate(capped, 1):
        story.append(Paragraph(f"<b>{p['title']}</b>", styles["Heading3"]))
        imgs = [
            RLImage(io.BytesIO(p["orig"]), 170, 120),
            RLImage(io.BytesIO(p["overlay"]), 170, 120),
            RLImage(io.BytesIO(p["mask"]), 170, 120),
        ]
        grid = Table(
            [["Original", "Overlay", "Mask"], imgs],
            colWidths=[170, 170, 170]
        )
        grid.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(grid)
        story.append(Spacer(1, 12))
        if i % 3 == 0:
            story.append(PageBreak())

    if len(photo_panels) > MAX_PHOTOS_IN_PDF:
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"Note: PDF includes only first {MAX_PHOTOS_IN_PDF} photos (limit).",
            styles["Italic"]
        ))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# ============================================================
# Sidebar settings
# ============================================================
with st.sidebar:
    st.header("Analysis Settings")
    exclude_dark = st.checkbox("Exclude dark pixels", True)
    min_v = st.slider("Shadow threshold (V)", 0, 255, 35)
    kernel_size = st.slider("Kernel size", 1, 21, 5, 2)
    open_iters = st.slider("Open iters", 0, 5, 1)
    close_iters = st.slider("Close iters", 0, 5, 2)
    minor_thr = st.number_input("Minor < (%)", value=5.0)
    moderate_thr = st.number_input("Moderate < (%)", value=15.0)

# ============================================================
# New Inspection (ONE REPORT)
# ============================================================
st.subheader("New Inspection (One Report)")

with st.form("inspect"):
    c1, c2, c3 = st.columns(3)
    with c1:
        insp_date = st.date_input("Inspection Date", date.today())
        vessel = st.text_input("Vessel Name")
    with c2:
        tank = st.text_input("Tank / Hold")
        location = st.text_input("Location")
    with c3:
        inspector = st.text_input("Inspector")
        remarks = st.text_area("Remarks")

    st.info(
        "✅ Choose ONE option:\n\n"
        "• Option A: Excel (.xlsx) with embedded photos\n"
        "• Option B: Upload one OR multiple photos (PNG/JPG/JPEG)\n\n"
        "⚠️ Do NOT use both options at the same time."
    )

    excel = st.file_uploader("Option A – Excel report (.xlsx)", ["xlsx"], key="excel_upl")

    st.info(
        "📸 Option B: Upload one OR multiple photos.\n"
        "Use Ctrl/Shift to select multiple files."
    )
    photos = st.file_uploader(
        "Option B – Photo(s) (PNG/JPG/JPEG)",
        ["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="photos_upl"
    )

    run = st.form_submit_button("Analyze (One Report)")

# ============================================================
# Analysis
# ============================================================
if run:
    use_excel = excel is not None
    use_photos = photos is not None and len(photos) > 0

    if not use_excel and not use_photos:
        st.error("Upload Excel OR at least one photo.")
        st.stop()

    if use_excel and use_photos:
        st.error("Please use only ONE option: Excel OR Photo(s).")
        st.stop()

    rust_px_total = 0
    valid_px_total = 0
    per_photo_rows = []
    photo_panels = []

    # -----------------------------
    # Option A: Excel embedded photos
    # -----------------------------
    if use_excel:
        images = extract_images_from_excel(excel.getvalue())
        if not images:
            st.error("No embedded images found in Excel. Please ensure photos are inserted/embedded.")
            st.stop()

        for i, (sheet, img_bytes) in enumerate(images, 1):
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

            pct, rp, vp, mask, overlay = analyze_rust_bgr(
                bgr,
                exclude_dark_pixels=exclude_dark,
                min_v_for_valid=min_v,
                kernel_size=kernel_size,
                open_iters=open_iters,
                close_iters=close_iters,
            )

            rust_px_total += int(rp)
            valid_px_total += int(vp)

            per_photo_rows.append({
                "Photo": i,
                "Source": f"Excel:{sheet}",
                "Rust %": round(float(pct), 2)
            })

            photo_panels.append({
                "title": f"Photo {i} (Excel sheet: {sheet})",
                "orig": pil_to_png_bytes(make_thumb(pil)),
                "overlay": pil_to_png_bytes(make_thumb(Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)))),
                "mask": pil_to_png_bytes(make_thumb(Image.fromarray(mask).convert("RGB"))),
            })

    # -----------------------------
    # Option B: Multiple photos -> ONE report
    # -----------------------------
    else:
        # Sort by filename for consistent reporting
        photos_sorted = sorted(photos, key=lambda f: f.name.lower())

        for i, f in enumerate(photos_sorted, 1):
            pil = Image.open(f).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

            pct, rp, vp, mask, overlay = analyze_rust_bgr(
                bgr,
                exclude_dark_pixels=exclude_dark,
                min_v_for_valid=min_v,
                kernel_size=kernel_size,
                open_iters=open_iters,
                close_iters=close_iters,
            )

            rust_px_total += int(rp)
            valid_px_total += int(vp)

            per_photo_rows.append({
                "Photo": i,
                "Source": f.name,
                "Rust %": round(float(pct), 2)
            })

            photo_panels.append({
                "title": f"Photo {i} ({f.name})",
                "orig": pil_to_png_bytes(make_thumb(pil)),
                "overlay": pil_to_png_bytes(make_thumb(Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)))),
                "mask": pil_to_png_bytes(make_thumb(Image.fromarray(mask).convert("RGB"))),
            })

    if valid_px_total <= 0:
        st.error("Valid pixels total is 0. Please check the photos (too dark / invalid).")
        st.stop()

    rust_pct_total = 100 * rust_px_total / max(valid_px_total, 1)
    severity = get_severity(rust_pct_total, float(minor_thr), float(moderate_thr))

    st.success(f"TOTAL Rust: {rust_pct_total:.2f}% | Severity: {severity}")

    report_meta = {
        "Vessel": vessel,
        "Tank / Hold": tank,
        "Location": location,
        "Inspection Date": str(insp_date),
        "Inspector": inspector,
        "Remarks": remarks,
        "Input Type": "Excel (embedded photos)" if use_excel else f"Photos uploaded ({len(photos)} files)"
    }

    totals = {
        "rust_pct": float(rust_pct_total),
        "severity": severity,
        "rust_pixels": int(rust_px_total),
        "valid_pixels": int(valid_px_total),
    }

    pdf_bytes = generate_pdf_with_thumbnails(
        report_meta=report_meta,
        totals=totals,
        per_photo_rows=per_photo_rows,
        photo_panels=photo_panels
    )

    safe_vessel = (vessel or "inspection").replace(" ", "_")
    st.download_button(
        "📄 Download Inspection Report (PDF)",
        data=pdf_bytes,
        file_name=f"rust_report_{safe_vessel}.pdf",
        mime="application/pdf"
    )

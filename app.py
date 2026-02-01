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
st.set_page_config(page_title="Rust Quantification (Inspection)", layout="wide")

if os.path.exists("logo.png"):
    c1, c2 = st.columns([1, 6])
    with c1:
        st.image("logo.png", use_container_width=True)
    with c2:
        st.title("Rust Quantification (Inspection)")
else:
    st.title("Rust Quantification (Inspection)")

st.caption(
    "Top section: upload fleet log (CSV/Excel). "
    "New Inspection: upload either Excel report (embedded photos) OR a single photo."
)
st.divider()

# ============================================================
# Expected log columns
# ============================================================
LOG_COLUMNS = [
    "timestamp_utc",
    "inspection_date",
    "vessel_name",
    "tank_no",
    "location",
    "inspector",
    "rust_pct",
    "severity",
    "rust_pixels",
    "valid_pixels",
]

# ============================================================
# Session state for fleet log
# ============================================================
if "log_df" not in st.session_state:
    st.session_state.log_df = pd.DataFrame(columns=LOG_COLUMNS)

# ============================================================
# Helpers
# ============================================================
def get_severity(rust_pct: float, minor_thr: float, moderate_thr: float) -> str:
    if rust_pct < minor_thr:
        return "Minor"
    if rust_pct < moderate_thr:
        return "Moderate"
    return "Severe"

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

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
# 🔝 TOP SECTION: Upload Fleet Log (CSV / Excel ONLY)
# ============================================================
st.subheader("Fleet Inspection Log (Upload / Continue)")

st.warning(
    "⚠️ Upload inspection history files ONLY.\n\n"
    "Accepted formats: CSV or Excel (.xlsx).\n\n"
    "📸 Do NOT upload photos in this section."
)

uploaded_log = st.file_uploader(
    "Upload existing inspection log (CSV or Excel)",
    type=["csv", "xlsx"],
    key="log_uploader",
)

if uploaded_log is not None:
    try:
        if uploaded_log.name.lower().endswith(".csv"):
            df_in = pd.read_csv(uploaded_log)
        else:
            df_in = pd.read_excel(uploaded_log, engine="openpyxl")

        # Ensure required columns exist
        for col in LOG_COLUMNS:
            if col not in df_in.columns:
                df_in[col] = ""

        st.session_state.log_df = df_in[LOG_COLUMNS].copy()
        st.success(f"Inspection log loaded: {uploaded_log.name}")

    except Exception as e:
        st.error(f"Failed to read log file: {e}")

st.divider()

# ============================================================
# 🆕 NEW INSPECTION: Excel report OR single photo
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
        "• Option A: Upload Excel maintenance report (.xlsx) with embedded photos → analyzed as ONE combined report.\n"
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
        "📄 Do NOT upload CSV or Excel files here."
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

    rust_pct_for_log = 0.0
    rust_pixels_for_log = 0
    valid_pixels_for_log = 0
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

        total_rust_pixels = 0
        total_valid_pixels = 0
        per_photo_rows = []

        with st.spinner("Analyzing embedded photos from Excel..."):
            for idx, (sheet, r, c, img_bytes) in enumerate(extracted, start=1):
                try:
                    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    # Skip non-image or unsupported formats
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

                total_rust_pixels += int(rust_pixels)
                total_valid_pixels += int(valid_pixels)

                per_photo_rows.append({
                    "photo_no": idx,
                    "sheet": sheet,
                    "row": r,
                    "col": c,
                    "rust_%": round(float(rust_pct), 2),
                    "rust_pixels": int(rust_pixels),
                    "valid_pixels": int(valid_pixels),
                })

        if total_valid_pixels == 0:
            st.error("Could not compute valid pixels from extracted images. Please check the photos and try again.")
            st.stop()

        total_rust_pct = 100.0 * total_rust_pixels / max(total_valid_pixels, 1)
        severity = get_severity(float(total_rust_pct), float(minor_thr), float(moderate_thr))

        st.markdown("## Excel Report Results (All Photos Combined as ONE Report)")
        st.write(f"**Total embedded photos found:** {len(extracted)}")
        st.write(f"**TOTAL Rust Area:** {total_rust_pct:.2f}%")
        st.write(f"**Overall Severity:** {severity}")
        st.write(f"Total rust pixels: {total_rust_pixels:,}")
        st.write(f"Total valid pixels: {total_valid_pixels:,}")

        st.markdown("### Per-photo breakdown")
        if len(per_photo_rows) == 0:
            st.warning("Photos were detected in the workbook, but none could be decoded. Please re-insert images and try again.")
        else:
            st.dataframe(pd.DataFrame(per_photo_rows), use_container_width=True)

        # values for fleet log (one row = one excel report)
        rust_pct_for_log = round(float(total_rust_pct), 2)
        rust_pixels_for_log = int(total_rust_pixels)
        valid_pixels_for_log = int(total_valid_pixels)

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

        severity = get_severity(float(rust_pct), float(minor_thr), float(moderate_thr))
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Original Photo")
            st.image(img_rgb, use_container_width=True)
        with col2:
            st.markdown("### Rust Overlay")
            st.image(overlay_rgb, use_container_width=True)

        st.markdown("## Results")
        st.write(f"**Rust Area:** {rust_pct:.2f}%")
        st.write(f"**Severity:** {severity}")
        st.write(f"Rust pixels: {rust_pixels:,}")
        st.write(f"Valid pixels: {valid_pixels:,}")

        st.markdown("### Rust Mask (white = rust)")
        st.image(mask, clamp=True, use_container_width=True)

        rust_pct_for_log = round(float(rust_pct), 2)
        rust_pixels_for_log = int(rust_pixels)
        valid_pixels_for_log = int(valid_pixels)

    # ============================================================
    # Append ONE row to fleet log (one report)
    # ============================================================
    row = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "inspection_date": str(inspection_date),
        "vessel_name": vessel_name.strip() or "N/A",
        "tank_no": tank_no.strip() or "N/A",
        "location": location.strip() or "N/A",
        "inspector": inspector.strip() or "N/A",
        "rust_pct": rust_pct_for_log,
        "severity": severity,
        "rust_pixels": rust_pixels_for_log,
        "valid_pixels": valid_pixels_for_log,
    }

    st.session_state.log_df = pd.concat(
        [st.session_state.log_df, pd.DataFrame([row])],
        ignore_index=True
    )

    st.success("✅ One report completed and saved to fleet log.")

st.divider()

# ============================================================
# 📊 Fleet Dashboard + Download
# ============================================================
st.subheader("Fleet-wide Rust Dashboard")

df = st.session_state.log_df.copy()

if df.empty:
    st.info("No inspections available. Upload a log or run a new inspection.")
else:
    st.download_button(
        "Download updated inspection_log.csv",
        data=df_to_csv_bytes(df),
        file_name="inspection_log.csv",
        mime="text/csv"
    )

    f1, f2 = st.columns(2)
    with f1:
        vessel_list = ["All"] + sorted([v for v in df["vessel_name"].dropna().unique() if str(v).strip()])
        vessel_sel = st.selectbox("Filter by Vessel", vessel_list)
    with f2:
        sev_list = ["All"] + sorted(df["severity"].dropna().unique())
        sev_sel = st.selectbox("Filter by Severity", sev_list)

    df_view = df
    if vessel_sel != "All":
        df_view = df_view[df_view["vessel_name"] == vessel_sel]
    if sev_sel != "All":
        df_view = df_view[df_view["severity"] == sev_sel]

    df_view["rust_pct"] = pd.to_numeric(df_view["rust_pct"], errors="coerce")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Inspections", int(df_view.shape[0]))
    with c2:
        st.metric("Average Rust %", f"{df_view['rust_pct'].mean():.2f}%")
    with c3:
        st.metric("Worst Rust %", f"{df_view['rust_pct'].max():.2f}%")

    st.markdown("### Inspection Records")
    st.dataframe(
        df_view.sort_values(["inspection_date", "vessel_name", "tank_no"], ascending=[False, True, True]),
        use_container_width=True
    )

    st.markdown("### Severity Count")
    sev_counts = df_view["severity"].value_counts()
    st.bar_chart(sev_counts)

    st.markdown("### Top 10 Worst Tanks / Areas")
    top10 = df_view.sort_values("rust_pct", ascending=False).head(10)[
        ["inspection_date", "vessel_name", "tank_no", "location", "rust_pct", "severity", "inspector"]
    ]
    st.dataframe(top10, use_container_width=True)

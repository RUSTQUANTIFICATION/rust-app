import os
import io
from datetime import datetime, date

import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from rust_analyzer import analyze_rust_bgr

# -----------------------------
# Settings
# -----------------------------
APP_TITLE = "Rust Quantification (Inspection)"
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

DEFAULT_MINOR = 5.0     # <5% = Minor
DEFAULT_MODERATE = 15.0 # 5-<15 = Moderate, >=15 = Severe

st.set_page_config(page_title=APP_TITLE, layout="wide")

# -----------------------------
# Header with logo
# -----------------------------
if os.path.exists("logo.png"):
    c1, c2 = st.columns([1, 6])
    with c1:
        st.image("logo.png", use_container_width=True)
    with c2:
        st.title(APP_TITLE)
else:
    st.title(APP_TITLE)

st.caption("Upload photo → calculate rust % → severity grading → download PDF/Excel → build fleet log & dashboard")
st.divider()

# -----------------------------
# Helper functions
# -----------------------------
def get_severity(rust_pct: float, minor_thr: float, moderate_thr: float) -> str:
    if rust_pct < minor_thr:
        return "Minor"
    elif rust_pct < moderate_thr:
        return "Moderate"
    return "Severe"

def bgr_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Failed to encode PNG")
    return buf.tobytes()

def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise ValueError("Failed to encode PNG mask")
    return buf.tobytes()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def make_excel_report(meta: dict, results: dict) -> bytes:
    df_meta = pd.DataFrame([{"Field": k, "Value": v} for k, v in meta.items()])
    df_res = pd.DataFrame([results])

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_meta.to_excel(writer, sheet_name="Inspection", index=False)
        df_res.to_excel(writer, sheet_name="Results", index=False)
    return out.getvalue()

def make_pdf_report(meta: dict, results: dict, logo_path: str = "logo.png") -> bytes:
    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=A4)
    w, h = A4
    y = h - 50

    # Logo
    if os.path.exists(logo_path):
        c.drawImage(logo_path, 40, h - 95, width=120, height=45, preserveAspectRatio=True, mask="auto")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(180, h - 70, "Rust Inspection Report")

    c.setFont("Helvetica", 9)
    c.drawString(40, h - 110, f"Generated (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    y = h - 140
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Inspection Details")
    y -= 18
    c.setFont("Helvetica", 10)
    for k, v in meta.items():
        c.drawString(55, y, f"{k}: {v}")
        y -= 14
        if y < 90:
            c.showPage()
            y = h - 50

    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Results")
    y -= 18
    c.setFont("Helvetica", 10)
    for k, v in results.items():
        c.drawString(55, y, f"{k}: {v}")
        y -= 14
        if y < 90:
            c.showPage()
            y = h - 50

    c.showPage()
    c.save()
    return out.getvalue()

# -----------------------------
# Sidebar: analysis settings
# -----------------------------
with st.sidebar:
    st.header("Analysis Settings")

    exclude_shadows = st.checkbox("Exclude dark/shadow pixels", value=True)
    min_v = st.slider("Shadow threshold (V)", 0, 255, 35, 1)

    st.subheader("Cleanup")
    k = st.slider("Kernel size", 1, 21, 5, 2)
    open_iters = st.slider("Open iterations", 0, 5, 1, 1)
    close_iters = st.slider("Close iterations", 0, 5, 2, 1)

    st.subheader("Severity thresholds (rust %)")
    minor_thr = st.number_input("Minor < (%)", min_value=0.0, max_value=100.0, value=DEFAULT_MINOR, step=0.5)
    moderate_thr = st.number_input("Moderate < (%)", min_value=0.0, max_value=100.0, value=DEFAULT_MODERATE, step=0.5)

    st.caption("Minor if rust% < Minor. Moderate if rust% < Moderate. Else Severe.")

# -----------------------------
# Fleet log: upload/continue
# -----------------------------
st.subheader("Fleet Inspection Log (Upload / Continue)")
log_upload = st.file_uploader("Upload existing inspection_log.csv (optional)", type=["csv"], key="logcsv")

if "log_df" not in st.session_state:
    st.session_state.log_df = pd.DataFrame(columns=LOG_COLUMNS)

if log_upload is not None:
    try:
        df_in = pd.read_csv(log_upload)
        for col in LOG_COLUMNS:
            if col not in df_in.columns:
                df_in[col] = ""
        st.session_state.log_df = df_in[LOG_COLUMNS].copy()
        st.success("Inspection log loaded.")
    except Exception as e:
        st.error(f"Could not read log: {e}")

st.divider()

# -----------------------------
# New inspection form
# -----------------------------
st.subheader("New Inspection")
with st.form("inspection_form"):
    a, b, c = st.columns(3)
    with a:
        inspection_date = st.date_input("Inspection Date", value=date.today())
        vessel_name = st.text_input("Vessel Name", placeholder="e.g., ASIA UNITY")
    with b:
        tank_no = st.text_input("Tank / Hold No.", placeholder="e.g., WB Tank P/S, Hold 1")
        location = st.text_input("Location / Area", placeholder="e.g., Ballast Tank, Cargo Hold, Void Space")
    with c:
        inspector = st.text_input("Inspector", placeholder="Name / Rank")
        remarks = st.text_area("Remarks (optional)", height=80)

    uploaded = st.file_uploader("Upload a photo (PNG/JPG)", type=["png", "jpg", "jpeg"])
    submitted = st.form_submit_button("Analyze & Generate Report")

if submitted:
    if uploaded is None:
        st.error("Please upload a photo.")
        st.stop()

    # Load image
    img_pil = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Analyze
    rust_pct, rust_pixels, valid_pixels, mask, overlay_bgr = analyze_rust_bgr(
        img_bgr,
        exclude_dark_pixels=exclude_shadows,
        min_v_for_valid=min_v,
        kernel_size=k,
        open_iters=open_iters,
        close_iters=close_iters,
    )

    severity = get_severity(rust_pct, minor_thr, moderate_thr)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Original")
        st.image(img_rgb, use_container_width=True)
    with col2:
        st.markdown("### Rust Overlay (red)")
        st.image(overlay_rgb, use_container_width=True)

    st.markdown("## Results")
    st.write(f"**Rust Area:** {rust_pct:.2f}%")
    st.write(f"**Severity:** {severity}")
    st.write(f"Rust pixels: {rust_pixels:,}")
    st.write(f"Valid pixels: {valid_pixels:,}")

    st.markdown("### Rust Mask (white = rust)")
    st.image(mask, clamp=True, use_container_width=True)

    # Metadata + report content
    meta = {
        "Inspection Date": str(inspection_date),
        "Vessel Name": vessel_name.strip() or "N/A",
        "Tank / Hold No.": tank_no.strip() or "N/A",
        "Location / Area": location.strip() or "N/A",
        "Inspector": inspector.strip() or "N/A",
        "Remarks": remarks.strip() or "",
    }
    results = {
        "Rust %": f"{rust_pct:.2f}",
        "Severity": severity,
        "Rust pixels": rust_pixels,
        "Valid pixels": valid_pixels,
        "Exclude shadows": str(exclude_shadows),
        "Shadow threshold (V)": min_v,
        "Kernel size": k,
        "Open iterations": open_iters,
        "Close iterations": close_iters,
    }

    # Downloads
    pdf_bytes = make_pdf_report(meta, results)
    excel_bytes = make_excel_report(meta, results)
    mask_bytes = mask_to_png_bytes(mask)
    overlay_bytes = bgr_to_png_bytes(overlay_bgr)

    st.markdown("## Downloads")
    st.download_button("Download PDF Report", data=pdf_bytes, file_name="rust_inspection_report.pdf", mime="application/pdf")
    st.download_button("Download Excel Report", data=excel_bytes, file_name="rust_inspection_report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Download Rust Mask (PNG)", data=mask_bytes, file_name="rust_mask.png", mime="image/png")
    st.download_button("Download Overlay (PNG)", data=overlay_bytes, file_name="rust_overlay.png", mime="image/png")

    # Add row to fleet log
    row = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "inspection_date": str(inspection_date),
        "vessel_name": meta["Vessel Name"],
        "tank_no": meta["Tank / Hold No."],
        "location": meta["Location / Area"],
        "inspector": meta["Inspector"],
        "rust_pct": round(float(rust_pct), 2),
        "severity": severity,
        "rust_pixels": rust_pixels,
        "valid_pixels": valid_pixels,
    }
    st.session_state.log_df = pd.concat([st.session_state.log_df, pd.DataFrame([row])], ignore_index=True)
    st.success("Saved to Fleet Inspection Log below.")

st.divider()

# -----------------------------
# Fleet-wide dashboard
# -----------------------------
st.subheader("Fleet-wide Rust Dashboard")

df = st.session_state.log_df.copy()
if df.empty:
    st.info("No inspections yet. Run an inspection or upload inspection_log.csv above.")
else:
    f1, f2, f3 = st.columns(3)
    with f1:
        vessel_list = ["All"] + sorted([v for v in df["vessel_name"].dropna().unique().tolist() if str(v).strip() != ""])
        vessel_sel = st.selectbox("Vessel Filter", vessel_list)
    with f2:
        sev_list = ["All"] + sorted(df["severity"].dropna().unique().tolist())
        sev_sel = st.selectbox("Severity Filter", sev_list)
    with f3:
        st.download_button("Download inspection_log.csv", data=df_to_csv_bytes(df), file_name="inspection_log.csv", mime="text/csv")

    df_view = df
    if vessel_sel != "All":
        df_view = df_view[df_view["vessel_name"] == vessel_sel]
    if sev_sel != "All":
        df_view = df_view[df_view["severity"] == sev_sel]

    st.markdown("### Records")
    st.dataframe(df_view.sort_values(["inspection_date", "vessel_name", "tank_no"], ascending=[False, True, True]),
                 use_container_width=True)

    st.markdown("### Summary")
    df_view["rust_pct"] = pd.to_numeric(df_view["rust_pct"], errors="coerce")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total inspections", int(df_view.shape[0]))
    with c2:
        st.metric("Average rust %", f"{df_view['rust_pct'].mean():.2f}%")
    with c3:
        st.metric("Worst rust %", f"{df_view['rust_pct'].max():.2f}%")

    st.markdown("### Severity count")
    sev_counts = df_view["severity"].value_counts().reset_index()
    sev_counts.columns = ["severity", "count"]
    st.bar_chart(sev_counts.set_index("severity"))

    st.markdown("### Top 10 worst tanks/areas")
    top10 = df_view.sort_values("rust_pct", ascending=False).head(10)[
        ["inspection_date", "vessel_name", "tank_no", "location", "rust_pct", "severity", "inspector"]
    ]
    st.dataframe(top10, use_container_width=True)

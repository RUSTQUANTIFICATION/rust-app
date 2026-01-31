import cv2
import numpy as np

DEFAULT_RUST_RANGES_HSV = [
    ((5,  60,  40), (30, 255, 255)),
    ((0,  40,  20), (20, 255, 200)),
]

def analyze_rust_bgr(
    img_bgr: np.ndarray,
    rust_ranges_hsv=DEFAULT_RUST_RANGES_HSV,
    exclude_dark_pixels: bool = True,
    min_v_for_valid: int = 35,
    kernel_size: int = 5,
    open_iters: int = 1,
    close_iters: int = 2,
):
    h, w = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask_rust = np.zeros((h, w), dtype=np.uint8)
    for lo, hi in rust_ranges_hsv:
        lo = np.array(lo, dtype=np.uint8)
        hi = np.array(hi, dtype=np.uint8)
        mask_rust = cv2.bitwise_or(mask_rust, cv2.inRange(img_hsv, lo, hi))

    if exclude_dark_pixels:
        v = img_hsv[:, :, 2]
        valid = (v >= min_v_for_valid).astype(np.uint8) * 255
    else:
        valid = np.ones((h, w), dtype=np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_clean = cv2.morphologyEx(mask_rust, cv2.MORPH_OPEN, kernel, iterations=open_iters)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

    mask_final = cv2.bitwise_and(mask_clean, valid)

    rust_pixels = int(np.count_nonzero(mask_final))
    valid_pixels = int(np.count_nonzero(valid))
    rust_pct = 100.0 * rust_pixels / max(valid_pixels, 1)

    overlay = img_bgr.copy()
    overlay[mask_final > 0] = (0, 0, 255)
    alpha = 0.45
    overlay_bgr = cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)

    return rust_pct, rust_pixels, valid_pixels, mask_final, overlay_bgr

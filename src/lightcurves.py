import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd

from src.data import load_koi

PROCESSED_DIR = Path("data") / "processed"
FAILED_LOG = Path("data") / "failed_stars.txt"

N_STARS_TOTAL = 3000   # random stars at the natural planet rate
N_WORKERS = 6          # downloads in parallel; lower to 3 if you see errors
GLOBAL_BINS = 2001
LOCAL_BINS = 201
MAX_RETRIES = 3


def pick_subset(n_total=N_STARS_TOTAL, seed=0):
    """Pick random labeled stars, one row per star."""
    df = load_koi()
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int)
    cols = ["kepid", "label", "koi_period", "koi_time0bk", "koi_duration"]
    df = df[cols].dropna().drop_duplicates(subset="kepid")
    return df.sample(n=min(n_total, len(df)), random_state=seed).reset_index(drop=True)


def download_star(kepid):
    """Download all long-cadence Kepler quarters for one star, with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            result = lk.search_lightcurve(
                f"KIC {kepid}", mission="Kepler", author="Kepler", cadence="long"
            )
            if len(result) == 0:
                return None
            return result.download_all().stitch()
        except Exception as e:
            print(f"  kepid {kepid} attempt {attempt + 1} failed: {e}")
            time.sleep(2 * (attempt + 1))
    return None


def bin_view(phase, flux, n_bins, half_width):
    """Median-bin flux over phase in [-half_width, half_width]."""
    edges = np.linspace(-half_width, half_width, n_bins + 1)
    idx = np.digitize(phase, edges) - 1
    view = np.full(n_bins, np.nan)
    for b in range(n_bins):
        in_bin = flux[idx == b]
        if len(in_bin) > 0:
            view[b] = np.median(in_bin)
    good = ~np.isnan(view)
    if good.sum() < 2:
        return None
    view = np.interp(np.arange(n_bins), np.where(good)[0], view[good])
    view = view - np.median(view)
    depth = abs(view.min())
    if depth == 0:
        return None
    return (view / depth).astype(np.float32)


def process_star(lc, period, t0, duration_hours):
    """Clean, flatten, fold and bin one light curve into two views."""
    duration = duration_hours / 24.0
    lc = lc.remove_nans()
    mask = lc.create_transit_mask(period=period, transit_time=t0, duration=duration * 1.5)
    lc = lc.flatten(window_length=901, mask=mask)
    lc = lc.remove_outliers(sigma_upper=4, sigma_lower=20)
    folded = lc.fold(period=period, epoch_time=t0)

    phase = folded.time.value
    flux = folded.flux.value

    global_view = bin_view(phase, flux, GLOBAL_BINS, half_width=period / 2)
    local_view = bin_view(phase, flux, LOCAL_BINS, half_width=2 * duration)
    if global_view is None or local_view is None:
        return None
    return global_view, local_view


def handle_star(row):
    """Download, process and save one star. Returns (kepid, status)."""
    kepid = int(row["kepid"])
    out = PROCESSED_DIR / f"{kepid}.npz"
    if out.exists():
        return kepid, "cached"

    lc = download_star(kepid)
    result = None
    if lc is not None:
        try:
            result = process_star(
                lc, row["koi_period"], row["koi_time0bk"], row["koi_duration"]
            )
        except Exception as e:
            print(f"  kepid {kepid} processing failed: {e}")

    if result is None:
        return kepid, "failed"

    np.savez(out, global_view=result[0], local_view=result[1], label=int(row["label"]))
    return kepid, "saved"


def build_dataset():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    subset = pick_subset()
    print("Stars to process:", len(subset), "| workers:", N_WORKERS)

    counts = {"saved": 0, "cached": 0, "failed": 0}
    rows = [row for _, row in subset.iterrows()]
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(handle_star, row) for row in rows]
        for i, fut in enumerate(as_completed(futures), start=1):
            kepid, status = fut.result()
            counts[status] += 1
            if status == "failed":
                with open(FAILED_LOG, "a") as f:
                    f.write(f"{kepid}\n")
            if i % 25 == 0:
                print(f"[{i}/{len(rows)}] {counts}")
    print("Done:", counts)


if __name__ == "__main__":
    build_dataset()
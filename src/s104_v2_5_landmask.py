#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFSU → S-104 HDF5 PIPELINE WITH INTEGRATED LAND MASKING
================================================================================
IHO S-104 Edition 2.0.0 | DCF 2 | WGS84 | MLLW
================================================================================

PIPELINE TERINTEGRASI:
  1. Konversi DFSU → Grid S-104 (regrid + datum conversion + trend)
  2. Build Land Mask dari Shapefile (OSM/Custom)
  3. Apply Land Mask ke semua timestep
  4. Output: HDF5 S-104 masked (daily/hourly/3day/single)

Struktur S-104 DCF=2 (Regular Grid) yang dihasilkan:
  /WaterLevel/WaterLevel.01
      attrs: gridOriginLongitude, gridOriginLatitude,
        gridSpacingLongitudinal, gridSpacingLatitudinal,
        numPointsLongitudinal, numPointsLatitudinal
    /Group_001/
        attr: timePoint (YYYYMMDDTHHMMSSZ)
        dataset: values — 2D compound array [NY, NX]
        compound fields: waterLevelHeight (f32), waterLevelTrend (uint8)
        fillValue height: -9999.0
        fillValue trend: 0 (Unknown)

Fitur:
  - Single-pass processing (tidak ada file intermediate)
  - Land mask reusable (disimpan sebagai dataset di HDF5)
  - Auto-patch atribut grid
  - Multiple output modes: daily | hourly | 3day | single
  - Validasi hasil masking
  - Metadata IHO S-104 Ed. 2.0.0 compliant

Dependencies:
  mikeio, s100py, scipy, numpy, h5py, geopandas, shapely
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import gc
import os
import sys
import re
import warnings
import logging
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
import h5py

# Optional imports dengan graceful fallback
try:
    import mikeio
    HAS_MIKEIO = True
except ImportError:
    HAS_MIKEIO = False
    warnings.warn("mikeio tidak tersedia — mode masking-only aktif")

try:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import s100py.s104 as s104
    HAS_S100PY = True
except ImportError:
    HAS_S100PY = False
    warnings.warn("s100py tidak tersedia — mode masking-only aktif")

try:
    import geopandas as gpd
    import shapely
    from shapely import contains_xy
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    warnings.warn("geopandas/shapely tidak tersedia — land mask tidak bisa dibuild")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib tidak tersedia — tile catalog PNG tidak bisa dibuat")

warnings.filterwarnings("ignore")

# ==============================================================================
# KONFIGURASI GLOBAL
# ==============================================================================

# ── MODE OPERASI ──────────────────────────────────────────────────────────────
# "full"        : DFSU → S-104 → Mask (butuh mikeio, s100py, scipy, geopandas)
# "mask_only"   : HDF5 S-104 existing → Apply mask (butuh h5py, geopandas)
# "dfsu_only"   : DFSU → S-104 unmasked (butuh mikeio, s100py, scipy)
# "catalog_only": Klasifikasi + penomoran ulang tile + peta katalog PNG + CSV
# mapping — TANPA DFSU/s100py (butuh geopandas+shapely;
# matplotlib opsional untuk PNG).
PIPELINE_MODE = "full"

# ── PATH INPUT/OUTPUT ─────────────────────────────────────────────────────────
DFSU_PATH    = r"D:\S-100\S111\HD_PLI_78.dfsu"
OUTPUT_DIR   = r"D:\S-100\S104\s104_v2_5_4"
FILE_PREFIX  = "104_70_85_"

# Shapefile daratan (OSM land polygon atau custom)
LAND_SHP     = r"D:\S-100\S104\Shp_78\Area_78.shp"

# Daftar 21 grid Teluk Jakarta (CSV: Name,min_lat,min_lon,max_lat,max_lon)
AREALIST_PATH = r"D:\S-100\S104\area_list_v1.txt"

# ── MULTI-GRID MODE ───────────────────────────────────────────────────────────
# True  : baca AREALIST_PATH → hasilkan HDF5 TERPISAH per grid (21 grid → 21 set
# file, satu subfolder per grid). DFSU dibuka SEKALI, reuse untuk semua
# grid. Alur per-grid identik dengan pipeline single-BBOX (tidak diubah).
# False : pakai BBOX tunggal di bawah (perilaku lama, single grid).
MULTI_GRID = True

# ── GRID & BBOX ───────────────────────────────────────────────────────────────
BBOX = [106.85, -6.12, 106.95, -6.07]

# Resolusi grid (meter → derajat)
GRID_RESOLUTION_M = 70.0
DX = GRID_RESOLUTION_M / 111000.0
DY = GRID_RESOLUTION_M / 110500.0

# ── PARAMETER DATA ────────────────────────────────────────────────────────────
WL_ITEM            = "Surface elevation"
WL_TREND_THRESHOLD = 0.2

# Fill values S-104 compliant
# waterLevelHeight: -9999.0 (float32)
# waterLevelTrend: 0 (Unknown) — uint8 enumeration
NODATA_WL    = np.float32(-9999.0)
NODATA_TREND = np.uint8(0)   # 0 = Unknown per S-104 spec

# Trend enumeration: 1=Decreasing, 2=Increasing, 3=Steady
TREND_DECREASING = np.uint8(1)
TREND_INCREASING = np.uint8(2)
TREND_STEADY    = np.uint8(3)

# ── KONVERSI DATUM MSL → MLLW ─────────────────────────────────────────────────
MSL_TO_MLLW_OFFSET = 0.60   # meter — VERIFIKASI dengan data pasut!

# ── LAND MASK CONFIG ──────────────────────────────────────────────────────────
LAND_BUFFER_DEG = -0.00005   # ≈ -5 meter (shrink ke dalam daratan)

# ── OUTPUT SPLIT MODE ─────────────────────────────────────────────────────────
SPLIT_MODE = "daily"   # hourly | daily | 3day | single

# ── FILTER WAKTU (None = semua) ───────────────────────────────────────────────
DATE_START = None
DATE_END   = None

# ── AUTO PATCH GRID ATTRIBUTES ────────────────────────────────────────────────
PATCH_GRID_ATTRS = True

# ── PATH STRUKTUR HDF5 S-104 ──────────────────────────────────────────────────
PATH_INSTANCE  = "WaterLevel/WaterLevel.01"
PATH_CONTAINER = "WaterLevel"

# ── HUNDREDS-OF-TILES / MEMORY MANAGEMENT ─────────────────────────────────────
PRELOAD_WL_CACHE = True      # preload full WL time-series ONCE into RAM (fix redundant per-grid reads)
MAX_CACHE_MB     = 2048.0    # if estimated cache > this, fall back to per-timestep read
GC_EVERY_GRID    = True      # gc.collect() + del transient arrays after each tile

# ── TILE CLASSIFICATION & RE-NUMBERING ────────────────────────────────────────
ENABLE_TILE_RENUMBER = True
LAND_SKIP_THRESHOLD  = 0.995 # land_frac >= this → PURE LAND → tile EXCLUDED (no output)
CLASSIFY_SAMPLE_N    = 25    # NxN sample points per tile for land fraction
TILE_NAME_FORMAT     = "Tile{n:03d}"
TILE_MAPPING_CSV     = "tile_mapping.csv"   # written into OUTPUT_DIR

# ── TILE CATALOG MAP (PNG) ────────────────────────────────────────────────────
MAKE_TILE_CATALOG = True
CATALOG_PNG       = "tile_catalog.png"      # written into OUTPUT_DIR
CATALOG_DPI       = 200

# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(output_dir: str):
    """Setup logging ke file dan console dengan UTF-8."""
    logger = logging.getLogger("s104_pipeline")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    try:
        log_path = Path(output_dir) / "s104_pipeline_masked.log"
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception as e:
        print(f"Warning: Tidak bisa buat log file: {e}")

    return logger

log = logging.getLogger("s104_pipeline")
log.addHandler(logging.StreamHandler(sys.stdout))

# ==============================================================================
# STEP 0: VALIDASI DEPENDENSI
# ==============================================================================

def validate_dependencies(mode: str) -> bool:
    """Validasi dependensi berdasarkan mode operasi."""
    required = {
        "full": [(HAS_MIKEIO, "mikeio"), (HAS_SCIPY, "scipy"),
                 (HAS_S100PY, "s100py"), (HAS_GEO, "geopandas+shapely")],
        "mask_only": [(HAS_GEO, "geopandas+shapely")],
        "dfsu_only": [(HAS_MIKEIO, "mikeio"), (HAS_SCIPY, "scipy"),
                      (HAS_S100PY, "s100py")],
        "catalog_only": [(HAS_GEO, "geopandas+shapely")],
    }

    missing = [name for available, name in required.get(mode, []) if not available]

    if missing:
        log.error(f"Dependensi tidak tersedia untuk mode '{mode}': {missing}")
        log.error("Install: conda install -c conda-forge mikeio s100py scipy geopandas shapely")
        return False

    if mode == "catalog_only" and not HAS_MPL:
        log.warning("matplotlib tidak tersedia — tile catalog PNG akan dilewati, "
                    "hanya CSV mapping yang dihasilkan")

    return True

# ==============================================================================
# STEP 1: BUILD GRID S-104
# ==============================================================================

def build_grid_and_properties(bbox, dx, dy):
    """Bangun grid terstruktur S-104 dari BBOX dan resolusi."""
    lon_arr = np.arange(bbox[0], bbox[2], dx)
    lat_arr = np.arange(bbox[1], bbox[3], dy)

    if len(lon_arr) == 0 or len(lat_arr) == 0:
        raise ValueError(f"Grid kosong! BBOX={bbox}, DX={dx}, DY={dy}")

    grid_xx, grid_yy = np.meshgrid(lon_arr, lat_arr)
    nx, ny = len(lon_arr), len(lat_arr)

    grid_properties = {
        "minx"       : float(lon_arr[0]),
        "maxx"       : float(lon_arr[-1]),
        "miny"       : float(lat_arr[0]),
        "maxy"       : float(lat_arr[-1]),
        "cellsize_x" : float(dx),
        "cellsize_y" : float(dy),
        "nx"         : int(nx),
        "ny"         : int(ny),
    }

    log.info(f"Grid S-104: nx={nx} x ny={ny} = {nx*ny:,} sel")
    log.info(f"  Resolusi: {dx*111000:.1f}m x {dy*110500:.1f}m")
    log.info(f"  BBOX: {lon_arr[0]:.6f},{lat_arr[0]:.6f} → {lon_arr[-1]:.6f},{lat_arr[-1]:.6f}")

    return grid_xx, grid_yy, grid_properties, lon_arr, lat_arr


def parse_area_list(path: str) -> list:
    """Parse area_list.txt (21 grid) menjadi list of dict.

    Format tiap baris (CSV tanpa header):
        Name,min_lat,min_lon,max_lat,max_lon

    Returns:
        list[dict] keys: name, min_lat, min_lon, max_lat, max_lon.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Area list tidak ditemukan: {path}")

    areas = []
    with open(p, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 5:
                log.warning(f"  area_list baris {line_num}: kolom tidak lengkap, skip: {line!r}")
                continue

            name = parts[0]
            try:
                min_lat, min_lon, max_lat, max_lon = (float(x) for x in parts[1:5])
            except ValueError as e:
                log.warning(f"  area_list baris {line_num}: gagal parse angka, skip: {e}")
                continue

            areas.append({
                "name": name,
                "min_lat": min_lat, "min_lon": min_lon,
                "max_lat": max_lat, "max_lon": max_lon,
            })

    if not areas:
        raise ValueError(f"Tidak ada area valid di {path}")

    log.info(f"  area_list: {len(areas)} grid terbaca dari {path}")
    return areas


# ==============================================================================
# STEP 2: BUILD LAND MASK (dari Shapefile)
# ==============================================================================

def load_land_union(shp_path: str, full_bbox: tuple, buffer_deg: float = LAND_BUFFER_DEG):
    """
    Baca shapefile daratan SEKALI (clipped ke full_bbox gabungan semua tile),
    lalu union_all() + buffer. Dipakai agar 154 tile tidak masing-masing
    membaca shapefile sendiri-sendiri (mahal untuk ratusan tile).

    Args:
        shp_path : path shapefile.
        full_bbox: (minlon, minlat, maxlon, maxlat) — bbox gabungan seluruh tile.
        buffer_deg: buffer dalam derajat (default LAND_BUFFER_DEG).

    Returns:
        shapely geometry (union, sudah di-buffer) atau None jika kosong.
    """
    if not HAS_GEO:
        raise RuntimeError("geopandas/shapely tidak tersedia")

    if not Path(shp_path).exists():
        raise FileNotFoundError(f"Shapefile tidak ditemukan: {shp_path}")

    log.info(f"Load land union (sekali) dari: {shp_path}")
    log.info(f"  Full bbox: {full_bbox}")

    gdf = gpd.read_file(shp_path, bbox=full_bbox)
    log.info(f"  Polygon daratan di full bbox: {len(gdf)} feature")

    if gdf.empty:
        log.warning("  Tidak ada polygon daratan di full bbox — land_union=None")
        return None

    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        log.info(f"  Reproject dari {gdf.crs} ke EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")

    land_union = gdf.geometry.union_all()
    log.info(f"  Geometry type: {land_union.geom_type}")

    if buffer_deg != 0.0:
        land_union = land_union.buffer(buffer_deg)
        log.info(f"  Buffer {buffer_deg} deg ({buffer_deg*111000:.1f}m) diterapkan")

    if land_union.is_empty:
        log.warning("  Land union kosong setelah buffer — land_union=None")
        return None

    return land_union


def build_land_mask(shp_path: str, lon_arr: np.ndarray, lat_arr: np.ndarray,
                     land_union=None) -> tuple:
    """
    Build land mask dari shapefile polygon daratan.

    Args:
        land_union: jika diberikan (hasil load_land_union), shapefile TIDAK
            dibaca ulang — langsung dipakai untuk point-in-polygon test.
            Jika None, perilaku SAMA PERSIS seperti sebelumnya (baca shapefile
            per-panggilan, backward compatible).

    Returns:
        land_mask: Boolean array (NY, NX), True = daratan
        mask_info: Dict info masking
    """
    if not HAS_GEO:
        raise RuntimeError("geopandas/shapely tidak tersedia")

    if land_union is not None:
        log.info("Building land mask dari land_union yang sudah dimuat (reuse)")
    else:
        if not Path(shp_path).exists():
            raise FileNotFoundError(f"Shapefile tidak ditemukan: {shp_path}")

        log.info(f"Building land mask dari: {shp_path}")

        bbox = (float(lon_arr[0]), float(lat_arr[0]),
                float(lon_arr[-1]), float(lat_arr[-1]))

        gdf = gpd.read_file(shp_path, bbox=bbox)
        log.info(f"  Polygon daratan di bbox: {len(gdf)} feature")

        if gdf.empty:
            log.warning("Tidak ada polygon daratan — mask kosong")
            ny, nx = len(lat_arr), len(lon_arr)
            return np.zeros((ny, nx), dtype=bool), {"n_land": 0, "n_total": ny*nx}

        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            log.info(f"  Reproject dari {gdf.crs} ke EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")

        land_union = gdf.geometry.union_all()
        log.info(f"  Geometry type: {land_union.geom_type}")

        if LAND_BUFFER_DEG != 0.0:
            land_union = land_union.buffer(LAND_BUFFER_DEG)
            log.info(f"  Buffer {LAND_BUFFER_DEG} deg ({LAND_BUFFER_DEG*111000:.1f}m) diterapkan")

    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
    lon_flat = lon_grid.ravel(order="C")
    lat_flat = lat_grid.ravel(order="C")

    log.info(f"  Testing {len(lon_flat):,} titik grid...")

    shapely_ver = tuple(int(x) for x in shapely.__version__.split(".")[:2])
    if shapely_ver >= (2, 0):
        land_mask_flat = contains_xy(land_union, lon_flat, lat_flat)
    else:
        from shapely.vectorized import contains as vcontains
        land_mask_flat = vcontains(land_union, lon_flat, lat_flat)

    land_mask = land_mask_flat.reshape((len(lat_arr), len(lon_arr)))

    n_land = int(land_mask.sum())
    n_total = land_mask.size

    log.info(f"  Daratan : {n_land:,} sel ({100*n_land/n_total:.2f}%)")
    log.info(f"  Perairan: {n_total-n_land:,} sel ({100*(n_total-n_land)/n_total:.2f}%)")

    if n_land == 0:
        log.warning("Tidak ada sel daratan terdeteksi — periksa CRS shapefile")
    if n_land == n_total:
        log.error("SEMUA sel daratan — shapefile mungkin terbalik (water polygon?)")

    mask_info = {
        "n_land": n_land,
        "n_total": n_total,
        "n_water": n_total - n_land,
        "percent_land": 100*n_land/n_total,
        "shp_file": Path(shp_path).name,
        "buffer_deg": LAND_BUFFER_DEG,
    }

    return land_mask, mask_info


# ==============================================================================
# STEP 2b: KLASIFIKASI TILE + PENOMORAN ULANG SPASIAL (S→N, W→E)
# ==============================================================================

def classify_and_renumber_tiles(areas: list, land_union, elem_x=None, elem_y=None) -> list:
    """
    Klasifikasi tiap tile (PURE_LAND/MIXED/PURE_SEA/NO_DATA) dan penomoran
    ulang spasial (row=0 di selatan, col=0 di barat; urutan S→N lalu W→E).

    Tile yang PURE_LAND (land_frac >= LAND_SKIP_THRESHOLD) atau NO_DATA
    (tidak ada elemen DFSU di dalam bbox, bila elem_x/elem_y diberikan)
    dikeluarkan (keep=False) dari output S-104 — tapi tetap dilaporkan
    di CSV mapping & PNG catalog untuk keperluan evaluasi.

    Args:
        areas    : list of dict dari parse_area_list (name, min_lat, min_lon,
                   max_lat, max_lon).
        land_union: shapely geometry (hasil load_land_union) atau None
                   (→ land_frac=0.0 untuk semua tile, semua dianggap PURE_SEA).
        elem_x, elem_y: koordinat elemen DFSU (opsional). Jika None, n_elem
                   tidak dihitung (None) dan tidak dipakai untuk exclude.

    Returns:
        list[dict] — satu entry per tile (termasuk yang di-skip), keys:
            name, bbox, center_lon, center_lat, land_frac, n_elem, tile_class,
            keep, row, col, new_name, new_num.
    """
    log.info("=" * 62)
    log.info("  KLASIFIKASI & PENOMORAN ULANG TILE")
    log.info("=" * 62)

    classified = []

    for area in areas:
        minlon, minlat = area["min_lon"], area["min_lat"]
        maxlon, maxlat = area["max_lon"], area["max_lat"]
        bbox = [minlon, minlat, maxlon, maxlat]
        center_lon = 0.5 * (minlon + maxlon)
        center_lat = 0.5 * (minlat + maxlat)

        # --- land_frac: sample CLASSIFY_SAMPLE_N x CLASSIFY_SAMPLE_N titik ---
        if land_union is not None and HAS_GEO:
            samp_lon = np.linspace(minlon, maxlon, CLASSIFY_SAMPLE_N)
            samp_lat = np.linspace(minlat, maxlat, CLASSIFY_SAMPLE_N)
            slon_grid, slat_grid = np.meshgrid(samp_lon, samp_lat)
            slon_flat = slon_grid.ravel()
            slat_flat = slat_grid.ravel()

            shapely_ver = tuple(int(x) for x in shapely.__version__.split(".")[:2])
            if shapely_ver >= (2, 0):
                inside = contains_xy(land_union, slon_flat, slat_flat)
            else:
                from shapely.vectorized import contains as vcontains
                inside = vcontains(land_union, slon_flat, slat_flat)

            land_frac = float(np.count_nonzero(inside)) / inside.size
        else:
            land_frac = 0.0

        # --- n_elem: hitung elemen DFSU di dalam bbox (bila tersedia) ---
        if elem_x is not None and elem_y is not None:
            in_bbox = ((elem_x >= minlon) & (elem_x <= maxlon) &
                       (elem_y >= minlat) & (elem_y <= maxlat))
            n_elem = int(in_bbox.sum())
        else:
            n_elem = None

        # --- klasifikasi ---
        if elem_x is not None and n_elem == 0:
            tile_class, keep = "NO_DATA", False
        elif land_frac >= LAND_SKIP_THRESHOLD:
            tile_class, keep = "PURE_LAND", False
        elif land_frac > 0.0:
            tile_class, keep = "MIXED", True
        else:
            tile_class, keep = "PURE_SEA", True

        if not ENABLE_TILE_RENUMBER:
            keep = True

        classified.append({
            "name": area["name"],
            "bbox": bbox,
            "center_lon": center_lon,
            "center_lat": center_lat,
            "land_frac": land_frac,
            "n_elem": n_elem,
            "tile_class": tile_class,
            "keep": keep,
            "row": None,
            "col": None,
            "new_name": None,
            "new_num": None,
        })

    # --- clustering row (lat) & kolom (lon) berbasis jarak antar center ---
    heights = [c["bbox"][3] - c["bbox"][1] for c in classified]
    widths  = [c["bbox"][2] - c["bbox"][0] for c in classified]
    median_h = float(np.median(heights)) if heights else 0.01
    median_w = float(np.median(widths)) if widths else 0.01

    def _cluster(values, gap_threshold):
        """Cluster nilai 1D yang berdekatan (gap < threshold) jadi band index
        berurutan (0=nilai terkecil)."""
        order = np.argsort(values)
        sorted_vals = np.array(values)[order]
        band_of_sorted = np.zeros(len(sorted_vals), dtype=int)
        band = 0
        for i in range(1, len(sorted_vals)):
            if (sorted_vals[i] - sorted_vals[i - 1]) >= gap_threshold:
                band += 1
            band_of_sorted[i] = band
        band_result = np.zeros(len(values), dtype=int)
        band_result[order] = band_of_sorted
        return band_result

    center_lats = [c["center_lat"] for c in classified]
    center_lons = [c["center_lon"] for c in classified]

    row_bands = _cluster(center_lats, 0.5 * median_h) if classified else []
    col_bands = _cluster(center_lons, 0.5 * median_w) if classified else []

    for c, r, cc in zip(classified, row_bands, col_bands):
        c["row"] = int(r)
        c["col"] = int(cc)

    # --- ordering & penomoran: tile KEPT, urut (row asc, col asc) ---
    kept = [c for c in classified if c["keep"]]
    kept_sorted = sorted(kept, key=lambda c: (c["row"], c["col"]))

    for i, c in enumerate(kept_sorted, start=1):
        c["new_num"] = i
        c["new_name"] = (c["name"] if not ENABLE_TILE_RENUMBER
                         else TILE_NAME_FORMAT.format(n=i))

    # --- summary log ---
    n_total = len(classified)
    n_kept = len(kept)
    n_pure_land = sum(1 for c in classified if c["tile_class"] == "PURE_LAND")
    n_no_data = sum(1 for c in classified if c["tile_class"] == "NO_DATA")
    n_mixed = sum(1 for c in classified if c["tile_class"] == "MIXED")
    n_pure_sea = sum(1 for c in classified if c["tile_class"] == "PURE_SEA")

    log.info(f"  Total tile     : {n_total}")
    log.info(f"  Kept (output)  : {n_kept}")
    log.info(f"  Skip PURE_LAND : {n_pure_land}")
    log.info(f"  Skip NO_DATA   : {n_no_data}")
    log.info(f"  MIXED (kept)   : {n_mixed}")
    log.info(f"  PURE_SEA (kept): {n_pure_sea}")
    log.info("=" * 62)

    return classified


def save_tile_mapping_csv(classified: list, out_csv: str) -> str:
    """Tulis CSV mapping old_name → new_name (+ row/col/class/land_frac/dll)."""
    import csv

    header = ["old_name", "new_name", "new_num", "row", "col", "tile_class",
              "keep", "land_frac", "n_elem", "minlon", "minlat", "maxlon", "maxlat"]

    kept = sorted([c for c in classified if c["keep"]], key=lambda c: c["new_num"])
    skipped = [c for c in classified if not c["keep"]]
    ordered = kept + skipped

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for c in ordered:
            minlon, minlat, maxlon, maxlat = c["bbox"]
            writer.writerow([
                c["name"], c["new_name"] or "", c["new_num"] or "",
                c["row"], c["col"], c["tile_class"], c["keep"],
                f"{c['land_frac']:.4f}",
                c["n_elem"] if c["n_elem"] is not None else "",
                f"{minlon:.6f}", f"{minlat:.6f}", f"{maxlon:.6f}", f"{maxlat:.6f}",
            ])

    log.info(f"  Tile mapping CSV ditulis: {out_csv} ({len(ordered)} baris)")
    return out_csv


# ==============================================================================
# STEP 2c: PETA KATALOG TILE (PNG)
# ==============================================================================

def plot_tile_catalog(classified: list, land_shp: str, out_png: str,
                       dpi: int = CATALOG_DPI) -> str:
    """
    Render peta katalog tile: poligon daratan sebagai latar, kotak tile
    berwarna per kelas, nomor baru untuk tile yang kept.
    """
    if not HAS_MPL:
        log.warning("  matplotlib tidak tersedia — plot_tile_catalog dilewati")
        return ""

    if not classified:
        log.warning("  Tidak ada tile untuk di-plot — dilewati")
        return ""

    all_lons = [b for c in classified for b in (c["bbox"][0], c["bbox"][2])]
    all_lats = [b for c in classified for b in (c["bbox"][1], c["bbox"][3])]
    minlon, maxlon = min(all_lons), max(all_lons)
    minlat, maxlat = min(all_lats), max(all_lats)

    lon_span = max(maxlon - minlon, 1e-6)
    lat_span = max(maxlat - minlat, 1e-6)
    aspect = lon_span / lat_span
    fig_h = max(12.0, 10.0)
    fig_w = max(10.0, fig_h * aspect)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # --- latar poligon daratan ---
    if HAS_GEO and land_shp and Path(land_shp).exists():
        try:
            pad = 0.02
            gdf = gpd.read_file(land_shp, bbox=(minlon - pad, minlat - pad,
                                                 maxlon + pad, maxlat + pad))
            if not gdf.empty:
                if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs("EPSG:4326")
                gdf.plot(ax=ax, color="#e8dcc0", edgecolor="#b9a97f",
                         linewidth=0.3, zorder=1)
        except Exception as e:
            log.warning(f"  Gagal plot latar daratan: {e}")

    class_style = {
        "PURE_LAND": dict(edgecolor="#999999", linestyle="--", facecolor="none", zorder=2),
        "NO_DATA":   dict(edgecolor="#999999", linestyle="--", facecolor="none", zorder=2),
        "MIXED":     dict(edgecolor="#e07b00", facecolor="none", linewidth=1.0, zorder=3),
        "PURE_SEA":  dict(edgecolor="#1f6feb", facecolor="none", linewidth=1.0, zorder=3),
    }
    text_color = {"MIXED": "#e07b00", "PURE_SEA": "#1f6feb"}

    for c in classified:
        blon, blat, blon2, blat2 = c["bbox"]
        w, h = blon2 - blon, blat2 - blat
        style = class_style.get(c["tile_class"], dict(edgecolor="black", facecolor="none"))
        rect = Rectangle((blon, blat), w, h, linewidth=style.get("linewidth", 0.8),
                          edgecolor=style["edgecolor"], facecolor=style["facecolor"],
                          linestyle=style.get("linestyle", "-"), zorder=style["zorder"])
        ax.add_patch(rect)

        if c["keep"] and c["new_num"] is not None:
            ax.annotate(str(c["new_num"]), (c["center_lon"], c["center_lat"]),
                        fontsize=6.5, ha="center", va="center",
                        color=text_color.get(c["tile_class"], "black"), zorder=4)

    ax.set_xlim(minlon - 0.005, maxlon + 0.005)
    ax.set_ylim(minlat - 0.005, maxlat + 0.005)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("S-104 Tile Catalog — renumbered (S→N, W→E), land tiles excluded")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_aspect("equal", adjustable="box")

    legend_handles = [
        Rectangle((0, 0), 1, 1, edgecolor="#1f6feb", facecolor="none", label="PURE_SEA (kept)"),
        Rectangle((0, 0), 1, 1, edgecolor="#e07b00", facecolor="none", label="MIXED (kept)"),
        Rectangle((0, 0), 1, 1, edgecolor="#999999", facecolor="none",
                  linestyle="--", label="Excluded (land/no-data)"),
    ]
    # Legend di LUAR area peta (di bawah) agar tidak menutupi tile bernomor.
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(0.0, -0.05), ncol=3, fontsize=8, frameon=True)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    log.info(f"  Tile catalog PNG disimpan: {out_png}")
    return out_png


# ==============================================================================
# STEP 3: REGRID DFSU → S-104 GRID
# ==============================================================================

def regrid_to_s104(elem_x, elem_y, values, grid_xx, grid_yy):
    """Interpolasi dari mesh tidak terstruktur ke grid terstruktur."""
    valid = np.isfinite(values)
    if valid.sum() < 3:
        log.warning("  regrid: < 3 titik valid, return NODATA penuh")
        return np.full(grid_xx.shape, NODATA_WL, dtype=np.float32)

    pts = np.column_stack([elem_x[valid], elem_y[valid]])
    vals = values[valid]

    grid_out = LinearNDInterpolator(pts, vals)(grid_xx, grid_yy)

    nan_mask = np.isnan(grid_out)
    if nan_mask.any():
        query_pts = np.column_stack([grid_xx[nan_mask], grid_yy[nan_mask]])
        grid_out[nan_mask] = NearestNDInterpolator(pts, vals)(query_pts)

    return np.where(np.isfinite(grid_out), grid_out, NODATA_WL).astype(np.float32)


def compute_wl_trend(wl_current, wl_prev, threshold=0.2):
    """
    Hitung trend water level per S-104 spec:
    1=Decreasing, 2=Increasing, 3=Steady, 0=Unknown (NODATA)
    """
    if wl_prev is None:
        trend = np.full(wl_current.shape, TREND_STEADY, dtype=np.uint8)
        trend[wl_current < (NODATA_WL + 1.0)] = NODATA_TREND
        return trend

    diff = np.round(wl_current.astype(np.float64) - wl_prev.astype(np.float64), decimals=2)
    trend = np.where(
        diff >= threshold, TREND_INCREASING,
        np.where(diff <= -threshold, TREND_DECREASING, TREND_STEADY)
    ).astype(np.uint8)

    nodata_mask = ((wl_current < (NODATA_WL + 1.0)) | (wl_prev < (NODATA_WL + 1.0)))
    trend[nodata_mask] = NODATA_TREND

    return trend


# ==============================================================================
# STEP 4: OPEN & DETEKSI DFSU
# ==============================================================================

def open_and_detect_dfsu(dfsu_path: str):
    """Buka file DFSU dan deteksi tipe (2D/3D)."""
    if not HAS_MIKEIO:
        raise RuntimeError("mikeio tidak tersedia")

    try:
        ds_open = mikeio.open(dfsu_path)
        log.info(f"Membuka DFSU: {dfsu_path}")
        log.info(f"  Items: {[i.name for i in ds_open.items]}")
        log.info(f"  Waktu: {ds_open.time[0]} → {ds_open.time[-1]} ({len(ds_open.time)} timestep)")
    except Exception as e:
        log.error(f"Gagal membuka DFSU: {e}")
        raise

    geom = ds_open.geometry
    elem_x = geom.element_coordinates[:, 0]
    elem_y = geom.element_coordinates[:, 1]
    n_layers = getattr(geom, "n_layers", None)

    if n_layers is None or n_layers <= 1:
        log.info(f"  Tipe: 2D | {len(elem_x):,} elemen")
        return ds_open, "2D", 1, None, None, elem_x, elem_y, None, None

    try:
        layer_ids = geom.layer_ids
        bbox_z_col = geom.element_coordinates[:, 2]
    except AttributeError:
        log.warning("  layer_ids tidak tersedia → fallback 2D")
        return ds_open, "2D", 1, None, None, elem_x, elem_y, None, None

    unique_lids = np.unique(layer_ids)
    z_unique = np.array([bbox_z_col[layer_ids == lid].mean() for lid in unique_lids])
    log.info(f"  Tipe: 3D | {n_layers} layer | z={np.abs(z_unique).round(2).tolist()}m")

    return (ds_open, "3D", n_layers, z_unique, unique_lids,
            elem_x, elem_y, layer_ids, bbox_z_col)


def build_bbox_mask(elem_x, elem_y, layer_ids, unique_lids, bbox, dfsu_type):
    """Filter elemen DFSU yang berada dalam BBOX."""
    minx, miny, maxx, maxy = bbox
    geo_mask = ((elem_x >= minx) & (elem_x <= maxx) &
                (elem_y >= miny) & (elem_y <= maxy))

    if dfsu_type == "2D" or layer_ids is None:
        if geo_mask.sum() == 0:
            raise ValueError(f"Tidak ada elemen dalam BBOX {bbox}")
        log.info(f"  BBOX mask 2D: {geo_mask.sum():,}/{len(elem_x):,} elemen")
        return geo_mask, None, elem_x[geo_mask], elem_y[geo_mask]

    top_layer_id = unique_lids[-1]
    top_layer_mask = (layer_ids == top_layer_id)
    combined_surf = geo_mask & top_layer_mask

    if combined_surf.sum() == 0:
        raise ValueError(f"Tidak ada elemen surface dalam BBOX {bbox}")

    log.info(f"  BBOX mask 3D surface: {combined_surf.sum():,} elemen")
    return combined_surf, geo_mask, elem_x[combined_surf], elem_y[combined_surf]


# ==============================================================================
# STEP 5: METADATA S-104
# ==============================================================================

def build_s104_metadata(geographic_id="Selat Lombok, Indonesia"):
    """Build metadata IHO S-104 Ed. 2.0.0."""
    return {
        "horizontalCRS"                : 4326,
        "verticalCS"                   : 6499,
        "verticalCoordinateBase"       : 2,
        "verticalDatumReference"       : 1,
        "verticalDatum"                : 12,      # MLLW
        "commonPointRule"              : 4,
        "verticalUncertainty"          : -1.0,
        "horizontalPositionUncertainty": -1.0,
        "methodWaterLevelProduct"      : "MIKE_FM_Hydrodynamic_Model_MSL_to_MLLW_offset",
        "interpolationType"            : 10,
        "waterLevelTrendThreshold"     : WL_TREND_THRESHOLD,
        "geographicIdentifier"         : geographic_id,
        "epoch"                        : "G1762",
        "datasetDeliveryInterval"      : "PT1H",
        "trendInterval"                : 60,
    }


# ==============================================================================
# STEP 6: HDF5 OUTPUT UTILITIES
# ==============================================================================

def safe_close_hdf5(output_h5: str):
    """Force close semua handle HDF5 yang terbuka untuk file tertentu."""
    gc.collect()
    try:
        open_ids = h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE)
        for fid in open_ids:
            try:
                fname = h5py.h5f.get_name(fid).decode("utf-8")
                if os.path.normcase(os.path.abspath(fname)) == os.path.normcase(os.path.abspath(output_h5)):
                    h5py.h5f.close(fid)
                    log.info(f"  Handle HDF5 ditutup: {fname}")
            except Exception:
                pass
    except Exception as e:
        log.warning(f"  Tidak bisa iterasi HDF5 IDs: {e}")

    if os.path.exists(output_h5):
        try:
            os.remove(output_h5)
            log.info(f"  File lama dihapus: {output_h5}")
        except PermissionError as e:
            log.error(f"  File masih dikunci: {e}")
            raise


def create_s104_hdf5(output_h5: str, dcf: int = 2):
    """Create file S-104 HDF5 baru menggunakan s100py."""
    if not HAS_S100PY:
        raise RuntimeError("s100py tidak tersedia")

    safe_close_hdf5(output_h5)
    data_file = s104.utils.create_s104(output_h5, dcf)
    log.info(f"  S104File dibuat: {output_h5}")
    return data_file


# ==============================================================================
# STEP 7: APPLY LAND MASK KE DATA
# ==============================================================================

def apply_land_mask_to_data(wl_grid: np.ndarray, wl_trend: np.ndarray, 
                            land_mask: np.ndarray) -> tuple:
    """
    Apply land mask ke grid water level dan trend.
    waterLevelHeight → NODATA_WL (-9999.0)
    waterLevelTrend → NODATA_TREND (0 = Unknown)
    """
    wl_masked = wl_grid.copy()
    trend_masked = wl_trend.copy()

    wl_masked[land_mask] = float(NODATA_WL)
    trend_masked[land_mask] = int(NODATA_TREND)

    return wl_masked, trend_masked


def compute_masked_stats(wl_grid: np.ndarray, land_mask: np.ndarray) -> dict:
    """Hitung statistik untuk sel perairan valid saja."""
    valid = wl_grid[~land_mask]
    valid = valid[valid > float(NODATA_WL) + 1.0]

    if valid.size > 0:
        return {"min": float(valid.min()), "max": float(valid.max()), 
                "count": int(valid.size), "mean": float(valid.mean())}
    return {"min": float(NODATA_WL), "max": float(NODATA_WL), 
            "count": 0, "mean": float(NODATA_WL)}


# ==============================================================================
# STEP 8: SPLIT OUTPUT PLANNING
# ==============================================================================

def build_split_plan(n_groups: int, mode: str) -> list:
    """Bangun rencana split output."""
    indices = list(range(n_groups))

    if mode == "single":
        return [("masked_all", indices)]
    elif mode == "hourly":
        return [(f"hour_{i+1:03d}", [i]) for i in indices]
    elif mode == "daily":
        plan = []
        for day_start in range(0, n_groups, 24):
            day_end = min(day_start + 24, n_groups)
            day_num = (day_start // 24) + 1
            plan.append((f"day_{day_num:03d}", list(range(day_start, day_end))))
        return plan
    elif mode == "3day":
        plan = []
        for block_start in range(0, n_groups, 72):
            block_end = min(block_start + 72, n_groups)
            block_num = (block_start // 72) + 1
            plan.append((f"3day_{block_num:03d}", list(range(block_start, block_end))))
        return plan
    else:
        raise ValueError(f"SPLIT_MODE tidak valid: {mode}")


# ==============================================================================
# STEP 9: WRITE MASKED HDF5 (manual, untuk mode mask_only)
# ==============================================================================

def write_masked_hdf5_manual(src_h5_path: str, dst_h5_path: str,
                              land_mask: np.ndarray, split_indices: list,
                              mask_info: dict, mode: str) -> str:
    """
    Salin struktur HDF5 dan apply land mask untuk subset groups.
    Compound dtype: waterLevelHeight (f32) + waterLevelTrend (uint8)
    """
    n_land = int(land_mask.sum())

    with h5py.File(src_h5_path, "r") as src_f:
        with h5py.File(dst_h5_path, "w") as dst_f:

            # Copy root attrs
            for key, val in src_f.attrs.items():
                dst_f.attrs[key] = val

            # Buat struktur WaterLevel/WaterLevel.01
            dst_f.create_group(PATH_CONTAINER)
            dst_inst = dst_f.create_group(PATH_INSTANCE)

            # Copy atribut instance
            src_inst = src_f[PATH_INSTANCE]
            for key, val in src_inst.attrs.items():
                dst_inst.attrs[key] = val

            # Copy atribut container
            src_cont = src_f[PATH_CONTAINER]
            for key, val in src_cont.attrs.items():
                dst_f[PATH_CONTAINER].attrs[key] = val

            # Salin dan mask setiap group
            group_names = sorted([n for n in src_inst.keys() if n.startswith("Group_")])

            for local_idx, global_idx in enumerate(split_indices, start=1):
                src_grp_name = group_names[global_idx]
                src_grp = src_inst[src_grp_name]
                dst_grp_name = f"Group_{local_idx:03d}"
                dst_grp = dst_inst.create_group(dst_grp_name)

                # Copy atribut group (termasuk timePoint)
                for key, val in src_grp.attrs.items():
                    dst_grp.attrs[key] = val

                # Baca data compound, apply mask
                src_ds = src_grp["values"]
                arr = src_ds[:]   # structured array

                wl_field = arr.dtype.names[0]   # waterLevelHeight
                trend_field = arr.dtype.names[1] if len(arr.dtype.names) > 1 else None  # waterLevelTrend

                # Apply mask — land_mask adalah 2D [NY, NX]
                arr[wl_field][land_mask] = float(NODATA_WL)
                if trend_field:
                    arr[trend_field][land_mask] = int(NODATA_TREND)

                # Tulis dataset compound
                dst_ds = dst_grp.create_dataset("values", data=arr, dtype=arr.dtype)
                for key, val in src_ds.attrs.items():
                    dst_ds.attrs[key] = val

                # Update statistik per group
                valid_wl = arr[wl_field][~land_mask]
                valid_wl = valid_wl[valid_wl > float(NODATA_WL) + 1.0]

                if valid_wl.size > 0:
                    wl_min, wl_max = float(valid_wl.min()), float(valid_wl.max())
                    for attr_min in ["minimumWaterLevelHeight", "minWaterLevelHeight"]:
                        if attr_min in dst_grp.attrs:
                            dst_grp.attrs[attr_min] = np.float32(wl_min)
                    for attr_max in ["maximumWaterLevelHeight", "maxWaterLevelHeight"]:
                        if attr_max in dst_grp.attrs:
                            dst_grp.attrs[attr_max] = np.float32(wl_max)

            # Tambah metadata masking
            now = datetime.now(timezone.utc).isoformat()
            dst_f.attrs["landMaskApplied"]    = b"TRUE"
            dst_f.attrs["landMaskSource"]     = mask_info.get("shp_file", "").encode("utf-8")
            dst_f.attrs["landMaskDate"]       = now.encode("utf-8")
            dst_f.attrs["landMaskCellCount"]  = np.uint32(n_land)
            dst_f.attrs["landMaskBuffer_deg"] = np.float32(mask_info.get("buffer_deg", 0))
            dst_f.attrs["landMaskSplitMode"]  = mode.encode("utf-8")

            existing = dst_f.attrs.get("history", b"")
            if isinstance(existing, bytes):
                existing = existing.decode("utf-8", errors="replace")
            dst_f.attrs["history"] = (
                f"{existing}{now} — Land mask applied. "
                f"Source: {mask_info.get('shp_file', '')}, "
                f"buffer: {mask_info.get('buffer_deg', 0)} deg, "
                f"cells masked: {n_land:,}, "
                f"split mode: {mode}."
            ).encode("utf-8")

    return dst_h5_path


# ==============================================================================
# STEP 10: VALIDASI OUTPUT
# ==============================================================================

def validate_masked_h5(h5_path: str, land_mask: np.ndarray) -> bool:
    """Validasi file HDF5 hasil masking."""
    log.info(f"Validasi: {Path(h5_path).name}")
    errors = []

    with h5py.File(h5_path, "r") as f:
        for attr in ["landMaskApplied", "landMaskSource", "landMaskDate"]:
            if attr not in f.attrs:
                errors.append(f"Attr '{attr}' tidak ada")

        if PATH_INSTANCE not in f:
            errors.append(f"Instance '{PATH_INSTANCE}' tidak ada")
            return False

        inst = f[PATH_INSTANCE]
        group_names = sorted([n for n in inst.keys() if n.startswith("Group_")])

        if not group_names:
            errors.append("Tidak ada Group_NNN")
            return False

        check_names = (group_names[:3] + group_names[-1:]) if len(group_names) >= 4 else group_names

        for grp_name in check_names:
            ds_path = f"{PATH_INSTANCE}/{grp_name}/values"
            if ds_path not in f:
                errors.append(f"Dataset tidak ada: {ds_path}")
                continue

            arr = f[ds_path][:]
            wl_field = arr.dtype.names[0]
            wl_vals = arr[wl_field]

            # Cek sel daratan = NODATA
            land_vals = wl_vals[land_mask]
            wrong = land_vals[land_vals > float(NODATA_WL) + 1.0]
            if len(wrong) > 0:
                errors.append(f"{grp_name}: {len(wrong):,} sel daratan masih valid")

            # Statistik perairan
            water_vals = wl_vals[~land_mask]
            valid_water = water_vals[water_vals > float(NODATA_WL) + 1.0]
            if valid_water.size > 0:
                log.info(f"  {grp_name}: valid={valid_water.size:,}, "
                        f"WL=[{valid_water.min():.3f}, {valid_water.max():.3f}]m")

    for e in errors:
        log.error(f"  ERROR: {e}")

    ok = len(errors) == 0
    log.info(f"  Validasi: {'LULUS' if ok else 'GAGAL'}")
    return ok


# ==============================================================================
# STEP 11: PROSES SATU HARI (FULL PIPELINE)
# ==============================================================================

def process_one_day_full(date_key, day_timesteps, ds_open, dfsu_type,
                          bbox_mask_2d, elem_x_surf, elem_y_surf,
                          grid_xx, grid_yy, grid_properties,
                          metadata, output_dir, land_mask, mask_info,
                          wl_prev_carry=None, prefix=FILE_PREFIX,
                          wl_cache=None, time_to_idx=None, bbox_elem_idx=None):
    """
    Proses satu hari: DFSU → Regrid → Mask → Write HDF5.
    Land mask di-apply langsung saat write (single-pass).

    Args:
    wl_cache, time_to_idx, bbox_elem_idx: bila wl_cache diberikan (preload
    RAM, lihat run_multi_grid_pipeline), timestep dibaca dari array
    in-memory alih-alih ds_open.read()/isel() per timestep — jauh
    lebih cepat untuk ratusan tile. Bila wl_cache None, perilaku
    SAMA PERSIS seperti sebelumnya (baca DFSU langsung per timestep).
    """
    if not HAS_S100PY:
        raise RuntimeError("s100py tidak tersedia untuk mode full pipeline")

    fname = f"{prefix}_{date_key.strftime('%Y%m%d')}.h5"
    output_h5 = os.path.join(output_dir, fname)
    n_times = len(day_timesteps)

    log.info(f"{'='*52}")
    log.info(f"Hari: {date_key} | {n_times} timestep → {fname}")

    data_file = create_s104_hdf5(output_h5, dcf=2)
    s104.utils.add_metadata(metadata, data_file)
    s104.utils.add_water_level_instance(data_file)
    log.info("  WaterLevel.01 instance dibuat")

    wl_prev = wl_prev_carry
    wl_last = None
    n_written = 0

    for i, t in enumerate(day_timesteps):
        log.info(f"  [{i+1:03d}/{n_times}] {t}")

        if wl_cache is not None:
            # --- Path CACHE: baca dari array in-memory (preload SEKALI) ---
            try:
                ti = time_to_idx[t]
                raw = wl_cache[ti, bbox_elem_idx].astype(np.float64)
                raw = np.where(np.abs(raw) < 1e-10, np.nan, raw)
            except Exception as e:
                log.error(f"    Gagal ambil dari wl_cache {t}: {e}. Skip.")
                continue
        else:
            # --- Path LAMA: baca langsung dari DFSU per timestep ---
            try:
                ds_slice = ds_open.read(items=[WL_ITEM], time=[t])
            except Exception as e:
                log.error(f"    Gagal read {t}: {e}. Skip.")
                continue

            try:
                if dfsu_type == "2D":
                    idx = np.where(bbox_mask_2d)[0].tolist()
                else:
                    idx = bbox_mask_2d.tolist()
                ds_bbox = ds_slice.isel(element=idx)
            except IndexError as e:
                log.error(f"    IndexError isel {t}: {e}. Skip.")
                continue

            raw = ds_bbox[WL_ITEM].to_numpy().flatten().astype(np.float64)
            raw = np.where(np.abs(raw) < 1e-10, np.nan, raw)

        # Regrid → MSL
        wl_grid = regrid_to_s104(elem_x_surf, elem_y_surf, raw, grid_xx, grid_yy)

        # Apply MSL→MLLW offset
        valid_mask = wl_grid > (NODATA_WL + 1.0)
        wl_grid[valid_mask] = wl_grid[valid_mask] + MSL_TO_MLLW_OFFSET

        # Compute trend (dari MLLW)
        wl_trend = compute_wl_trend(wl_grid, wl_prev, WL_TREND_THRESHOLD)

        # Apply land mask — SINGLE PASS!
        wl_grid_masked, wl_trend_masked = apply_land_mask_to_data(
            wl_grid, wl_trend, land_mask
        )

        # Konversi timestamp
        if hasattr(t, "to_pydatetime"):
            t_dt = t.to_pydatetime()
        elif isinstance(t, np.datetime64):
            t_dt = t.astype("M8[ms]").astype(datetime)
        else:
            t_dt = t

        # Write ke HDF5
        try:
            s104.utils.add_data_from_arrays(
                wl_grid_masked, wl_trend_masked,
                data_file, grid_properties,
                t_dt, 2
            )

            stats = compute_masked_stats(wl_grid_masked, land_mask)
            if stats["count"] > 0:
                log.info(f"    OK WL(MLLW) [{stats['min']:.3f}, {stats['max']:.3f}]m | "
                        f"valid={stats['count']:,} sel | land={mask_info['n_land']:,} sel")
        except Exception as e:
            log.error(f"    Gagal write {t}: {e}. Skip.")
            continue

        wl_prev = wl_grid.copy()
        wl_last = wl_grid.copy()
        n_written += 1

    # Guard: bila TIDAK ada satu pun group tertulis (mis. semua timestep skip
    # karena error cache), jangan panggil update_metadata/write_data_file —
    # s100py akan crash 'list index out of range' saat baca group terakhir.
    if n_written == 0:
        log.error(f"  Tidak ada timestep tertulis untuk {date_key} — "
                  f"file {fname} dilewati (tidak difinalisasi)")
        try:
            safe_close_hdf5(output_h5)
        except Exception:
            pass
        return wl_prev_carry, None

    # Finalisasi metadata
    t_first_str = (day_timesteps[0].strftime("%Y%m%dT%H%M%SZ")
                   if hasattr(day_timesteps[0], "strftime")
                   else str(day_timesteps[0]))

    s104.utils.update_metadata(
        data_file, grid_properties,
        {"dateTimeOfFirstRecord": t_first_str, "dataDynamicity": 5}
    )
    s104.utils.write_data_file(data_file)

    # Tambah metadata masking ke file
    with h5py.File(output_h5, "r+") as f:
        now = datetime.now(timezone.utc).isoformat()
        f.attrs["landMaskApplied"]    = b"TRUE"
        f.attrs["landMaskSource"]     = mask_info.get("shp_file", "").encode("utf-8")
        f.attrs["landMaskDate"]       = now.encode("utf-8")
        f.attrs["landMaskCellCount"]  = np.uint32(mask_info["n_land"])
        f.attrs["landMaskBuffer_deg"] = np.float32(mask_info.get("buffer_deg", 0))
        f.attrs["landMaskResolution_m"] = np.float32(GRID_RESOLUTION_M)

    log.info(f"  Selesai: {output_h5} ({n_times} timestep)")
    return wl_last, output_h5


# ==============================================================================
# STEP 12: MASK-ONLY MODE (untuk HDF5 existing)
# ==============================================================================

def run_mask_only(input_h5: str, output_dir: str, land_shp: str):
    """
    Mode mask-only: Baca HDF5 S-104 existing → Apply land mask → Split output.
    """
    log.info("=" * 62)
    log.info("  MODE: MASK-ONLY (HDF5 existing → Apply Land Mask)")
    log.info("=" * 62)

    if not Path(input_h5).exists():
        raise FileNotFoundError(f"Input HDF5 tidak ditemukan: {input_h5}")

    # Baca grid dari HDF5
    with h5py.File(input_h5, "r") as f:
        if PATH_INSTANCE not in f:
            raise RuntimeError(f"Instance '{PATH_INSTANCE}' tidak ada di HDF5")

        inst = f[PATH_INSTANCE]

        ox = float(inst.attrs.get("gridOriginLongitude", BBOX[0]))
        oy = float(inst.attrs.get("gridOriginLatitude", BBOX[1]))
        dx = float(inst.attrs.get("gridSpacingLongitudinal", DX))
        dy = float(inst.attrs.get("gridSpacingLatitudinal", DY))
        nx = int(inst.attrs.get("numPointsLongitudinal", 0))
        ny = int(inst.attrs.get("numPointsLatitudinal", 0))

        if nx == 0 or ny == 0:
            lon_arr = np.arange(BBOX[0], BBOX[2], DX)
            lat_arr = np.arange(BBOX[1], BBOX[3], DY)
            nx, ny = len(lon_arr), len(lat_arr)
        else:
            lon_arr = ox + np.arange(nx) * dx
            lat_arr = oy + np.arange(ny) * dy

    log.info(f"Grid dari HDF5: {ny}x{nx} = {ny*nx:,} sel")

    # Build land mask
    land_mask, mask_info = build_land_mask(land_shp, lon_arr, lat_arr)

    # Baca semua groups
    with h5py.File(input_h5, "r") as f:
        inst = f[PATH_INSTANCE]
        group_names = sorted([n for n in inst.keys() if n.startswith("Group_")])
        n_groups = len(group_names)

    log.info(f"Total groups: {n_groups}")

    # Build split plan
    split_plan = build_split_plan(n_groups, SPLIT_MODE)
    log.info(f"Split mode: {SPLIT_MODE} → {len(split_plan)} file output")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_files = []

    for file_idx, (file_label, group_indices) in enumerate(split_plan, start=1):
        output_filename = f"{Path(input_h5).stem}_{file_label}_masked.h5"
        output_path = str(Path(output_dir) / output_filename)

        log.info(f"[{file_idx:03d}/{len(split_plan)}] {output_filename} ({len(group_indices)} groups)")

        write_masked_hdf5_manual(
            input_h5, output_path, land_mask,
            group_indices, mask_info, SPLIT_MODE
        )

        size_mb = Path(output_path).stat().st_size / (1024**2)
        log.info(f"  → {size_mb:.2f} MB")
        output_files.append(output_path)

    # Validasi
    log.info("=" * 62)
    log.info("Validasi output...")
    all_ok = True
    for h5_path in output_files:
        if not validate_masked_h5(h5_path, land_mask):
            all_ok = False

    log.info("=" * 62)
    log.info(f"Total file output: {len(output_files)}")
    for fp in output_files:
        log.info(f"  - {fp}")
    log.info(f"Status: {'SEMUA VALIDASI LULUS' if all_ok else 'ADA YANG GAGAL'}")

    return output_files


# ==============================================================================
# STEP 13: FULL PIPELINE (DFSU → S-104 + Mask)
# ==============================================================================

def run_full_pipeline():
    """Pipeline lengkap: DFSU → S-104 → Land Mask (single-pass)."""
    log.info("=" * 62)
    log.info("  PIPELINE: DFSU → S-104 + LAND MASK (Integrated Single-Pass)")
    log.info("=" * 62)
    log.warning(f"MSL_TO_MLLW_OFFSET = {MSL_TO_MLLW_OFFSET}m (verifikasi sebelum produksi!)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Open DFSU
    (ds_open, dfsu_type, n_layers,
     z_unique, unique_lids,
     elem_x, elem_y,
     layer_ids, bbox_z_col) = open_and_detect_dfsu(DFSU_PATH)

    # 2. Build BBOX mask
    (bbox_mask_2d, bbox_mask_full,
     elem_x_surf, elem_y_surf) = build_bbox_mask(
        elem_x, elem_y, layer_ids, unique_lids, BBOX, dfsu_type
    )

    # 3. Build grid
    grid_xx, grid_yy, grid_properties, lon_arr, lat_arr = build_grid_and_properties(
        BBOX, DX, DY
    )

    # 4. Build land mask — SEBELUM proses timestep (reusable!)
    log.info("")
    log.info("=" * 62)
    log.info("  BUILDING LAND MASK")
    log.info("=" * 62)
    land_mask, mask_info = build_land_mask(LAND_SHP, lon_arr, lat_arr)

    # 5. Metadata
    metadata = build_s104_metadata(geographic_id="Selat Lombok, Indonesia")

    # 6. Group timestep per hari
    daily_groups = defaultdict(list)
    for t in ds_open.time:
        day_key = (t.date() if hasattr(t, "date")
                   else t.astype("M8[D]").astype(datetime.date))
        daily_groups[day_key].append(t)

    all_days = sorted(daily_groups.keys())
    log.info(f"Total hari tersedia: {len(all_days)} ({all_days[0]} → {all_days[-1]})")

    # Filter waktu
    sorted_days = all_days.copy()
    if DATE_START is not None or DATE_END is not None:
        d_start = DATE_START if DATE_START is not None else all_days[0]
        d_end = DATE_END if DATE_END is not None else all_days[-1]
        sorted_days = [d for d in all_days if d_start <= d <= d_end]
        log.info(f"Filter aktif: {d_start} → {d_end}")
        log.info(f"Hari diproses: {len(sorted_days)}")

    # 7. Proses setiap hari
    wl_prev_carry = None
    output_files = []

    for day_idx, day_key in enumerate(sorted_days):
        log.info(f"[{day_idx+1:04d}/{len(sorted_days)}] Memproses: {day_key}")

        wl_last, h5_path = process_one_day_full(
            date_key=day_key,
            day_timesteps=daily_groups[day_key],
            ds_open=ds_open,
            dfsu_type=dfsu_type,
            bbox_mask_2d=bbox_mask_2d,
            elem_x_surf=elem_x_surf,
            elem_y_surf=elem_y_surf,
            grid_xx=grid_xx,
            grid_yy=grid_yy,
            grid_properties=grid_properties,
            metadata=metadata,
            output_dir=OUTPUT_DIR,
            land_mask=land_mask,
            mask_info=mask_info,
            wl_prev_carry=wl_prev_carry,
            prefix=FILE_PREFIX,
        )

        wl_prev_carry = wl_last
        if h5_path is not None:
            output_files.append(h5_path)

    # 8. Summary
    log.info("" + "=" * 62)
    log.info(f"SELESAI — {len(output_files)} file HDF5")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info(f"Land mask: {mask_info['n_land']:,} sel daratan ({mask_info['percent_land']:.1f}%)")
    log.info("=" * 62)

    return output_files


# ==============================================================================
# STEP 14: MULTI-GRID PIPELINE (21 grid dari area_list.txt → file terpisah)
# ==============================================================================

def _group_timesteps_by_day(timesteps):
    """Kelompokkan timestep per hari (dipakai multi-grid). Sama seperti logika
    grouping di run_full_pipeline, diekstrak agar reusable per grid."""
    daily_groups = defaultdict(list)
    for t in timesteps:
        day_key = (t.date() if hasattr(t, "date")
                   else t.astype("M8[D]").astype(datetime.date))
        daily_groups[day_key].append(t)
    return daily_groups


def run_multi_grid_pipeline():
    """Pipeline MULTI-GRID: baca N grid dari area_list.txt → hasilkan HDF5
    TERPISAH per grid (satu subfolder per grid).

    Alur per-grid IDENTIK dengan run_full_pipeline (DFSU→regrid→datum→trend→
    land mask→write, single-pass). Yang berbeda:
    - DFSU dibuka SEKALI lalu di-reuse untuk semua grid (efisiensi).
    - Land geometry (shapefile) dibaca SEKALI via load_land_union, reuse
      untuk build_land_mask tiap tile (bukan baca shapefile per-tile).
    - Tile diklasifikasi (PURE_LAND/MIXED/PURE_SEA/NO_DATA) & dinomori
      ulang secara spasial (S→N, W→E); tile PURE_LAND/NO_DATA dilewati.
    - WL time-series di-preload SEKALI ke RAM (bila ukuran wajar) agar
      tidak ada pembacaan DFSU berulang per tile.
    - BBOX, grid, dan land mask tetap dibangun ulang per tile (murah).
    - Output ditulis ke OUTPUT_DIR/<new_name>/ dengan prefix per tile.
    """
    log.info("=" * 62)
    log.info("  PIPELINE MULTI-GRID: DFSU → S-104 + LAND MASK (per-grid file)")
    log.info("=" * 62)
    log.warning(f"MSL_TO_MLLW_OFFSET = {MSL_TO_MLLW_OFFSET}m (verifikasi sebelum produksi!)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Baca daftar grid
    areas = parse_area_list(AREALIST_PATH)
    log.info(f"Total grid dari area_list: {len(areas)}")

    # 2. Open DFSU SEKALI (reuse untuk semua grid)
    (ds_open, dfsu_type, n_layers,
     z_unique, unique_lids,
     elem_x, elem_y,
     layer_ids, bbox_z_col) = open_and_detect_dfsu(DFSU_PATH)

    # 2b. Land geometry SEKALI (reuse untuk klasifikasi + build_land_mask tiap tile)
    full_bbox = (
        min(a["min_lon"] for a in areas), min(a["min_lat"] for a in areas),
        max(a["max_lon"] for a in areas), max(a["max_lat"] for a in areas),
    )
    land_union = load_land_union(LAND_SHP, full_bbox)

    # 2c. Klasifikasi + penomoran ulang spasial tile
    classified = classify_and_renumber_tiles(areas, land_union, elem_x, elem_y)

    if MAKE_TILE_CATALOG:
        plot_tile_catalog(classified, LAND_SHP, os.path.join(OUTPUT_DIR, CATALOG_PNG))

    save_tile_mapping_csv(classified, os.path.join(OUTPUT_DIR, TILE_MAPPING_CSV))

    # 3. Metadata (sama untuk semua grid)
    metadata = build_s104_metadata(geographic_id="Teluk Jakarta, Indonesia")

    # 4. Kumpulkan & filter timestep SEKALI (sama untuk semua grid)
    daily_groups = _group_timesteps_by_day(ds_open.time)
    all_days = sorted(daily_groups.keys())
    log.info(f"Total hari tersedia: {len(all_days)} ({all_days[0]} → {all_days[-1]})")

    sorted_days = all_days.copy()
    if DATE_START is not None or DATE_END is not None:
        d_start = DATE_START if DATE_START is not None else all_days[0]
        d_end = DATE_END if DATE_END is not None else all_days[-1]
        sorted_days = [d for d in all_days if d_start <= d <= d_end]
        log.info(f"Filter aktif: {d_start} → {d_end} | {len(sorted_days)} hari diproses")

    # 4b. Preload WL cache SEKALI ke RAM (fix pembacaan DFSU berulang per tile)
    wl_cache = None
    time_to_idx = None
    n_time_total = len(ds_open.time)
    n_elem_total = len(elem_x)
    est_mb = n_time_total * n_elem_total * 4 / 1e6

    if PRELOAD_WL_CACHE and est_mb <= MAX_CACHE_MB:
        log.info(f"Preload WL cache: {n_time_total} timestep x {n_elem_total} elemen "
                 f"(~{est_mb:.1f} MB) → RAM")
        try:
            wl_cache = ds_open.read(items=[WL_ITEM]).to_numpy().astype(np.float32)
            # to_numpy() → (n_items, n_time, n_elem); WL_ITEM tunggal, ambil
            # item 0 agar cache jadi 2D (n_time, n_elem). Tanpa ini indexing
            # wl_cache[ti, bbox_elem_idx] salah sumbu (item vs time vs elem).
            if wl_cache.ndim == 3:
                wl_cache = wl_cache[0]
            time_to_idx = {t: i for i, t in enumerate(ds_open.time)}
            log.info(f"  WL cache siap: shape={wl_cache.shape}, "
                     f"~{wl_cache.nbytes/1e6:.1f} MB aktual")
        except Exception as e:
            log.warning(f"  Gagal preload WL cache ({e}) — fallback baca per-timestep")
            wl_cache, time_to_idx = None, None
    elif PRELOAD_WL_CACHE:
        log.warning(f"  WL cache diestimasi ~{est_mb:.1f} MB > MAX_CACHE_MB={MAX_CACHE_MB} "
                    f"— fallback baca per-timestep")
    else:
        log.info("  PRELOAD_WL_CACHE=False — baca per-timestep (perilaku lama)")

    # 5. Loop HANYA tile keep==True, urut new_num (S→N, W→E)
    kept_tiles = sorted([c for c in classified if c["keep"]], key=lambda c: c["new_num"])
    n_skipped = len(classified) - len(kept_tiles)

    all_output_files = []
    grid_summary = []

    for gi, ctile in enumerate(kept_tiles, start=1):
        name = ctile["name"]
        new_name = ctile["new_name"] or name
        bbox = ctile["bbox"]   # [min_lon, min_lat, max_lon, max_lat]

        log.info("")
        log.info("#" * 62)
        log.info(f"# TILE [{gi:03d}/{len(kept_tiles)}] {name} → {new_name}  "
                 f"bbox={bbox}  class={ctile['tile_class']}")
        log.info("#" * 62)

        grid_dir = os.path.join(OUTPUT_DIR, new_name)
        os.makedirs(grid_dir, exist_ok=True)

        # 5a. BBOX mask — skip tile jika tidak ada elemen DFSU di dalamnya
        try:
            (bbox_mask_2d, bbox_mask_full,
             elem_x_surf, elem_y_surf) = build_bbox_mask(
                elem_x, elem_y, layer_ids, unique_lids, bbox, dfsu_type
            )
        except ValueError as e:
            log.warning(f"  Tile {new_name} dilewati (tidak ada elemen DFSU): {e}")
            grid_summary.append((new_name, 0, 0))
            continue

        # 5b. Grid + land mask (per tile, land_union di-reuse)
        grid_xx, grid_yy, grid_properties, lon_arr, lat_arr = build_grid_and_properties(
            bbox, DX, DY
        )
        land_mask, mask_info = build_land_mask(
            LAND_SHP, lon_arr, lat_arr, land_union=land_union
        )

        # Indeks elemen DFSU dalam bbox tile ini, untuk lookup ke wl_cache
        bbox_elem_idx = np.where(bbox_mask_2d)[0]

        # 5c. Proses tiap hari → file per tile
        prefix = f"{FILE_PREFIX}_{new_name}"
        wl_prev_carry = None
        grid_files = []

        for day_idx, day_key in enumerate(sorted_days):
            log.info(f"  [{new_name}] hari [{day_idx+1:03d}/{len(sorted_days)}]: {day_key}")

            wl_last, h5_path = process_one_day_full(
                date_key=day_key,
                day_timesteps=daily_groups[day_key],
                ds_open=ds_open,
                dfsu_type=dfsu_type,
                bbox_mask_2d=bbox_mask_2d,
                elem_x_surf=elem_x_surf,
                elem_y_surf=elem_y_surf,
                grid_xx=grid_xx,
                grid_yy=grid_yy,
                grid_properties=grid_properties,
                metadata=metadata,
                output_dir=grid_dir,
                land_mask=land_mask,
                mask_info=mask_info,
                wl_prev_carry=wl_prev_carry,
                prefix=prefix,
                wl_cache=wl_cache,
                time_to_idx=time_to_idx,
                bbox_elem_idx=bbox_elem_idx,
            )

            wl_prev_carry = wl_last
            if h5_path is not None:
                grid_files.append(h5_path)

        all_output_files.extend(grid_files)
        grid_summary.append((new_name, len(grid_files), mask_info["n_land"]))
        log.info(f"  Tile {new_name} selesai: {len(grid_files)} file → {grid_dir}")

        # 5d. Bersihkan array transient per tile (memori untuk ratusan tile)
        if GC_EVERY_GRID:
            del land_mask, grid_xx, grid_yy, mask_info
            gc.collect()

    # 6. Summary
    log.info("")
    log.info("=" * 62)
    log.info(f"SELESAI MULTI-GRID — {len(all_output_files)} file HDF5 dari "
             f"{len(kept_tiles)} tile kept ({n_skipped} tile dilewati dari "
             f"{len(classified)} total)")
    for name, nf, nland in grid_summary:
        log.info(f"  {name:10s}: {nf} file | land={nland:,} sel")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info(f"Tile mapping CSV: {os.path.join(OUTPUT_DIR, TILE_MAPPING_CSV)}")
    if MAKE_TILE_CATALOG:
        log.info(f"Tile catalog PNG: {os.path.join(OUTPUT_DIR, CATALOG_PNG)}")
    log.info("=" * 62)

    return all_output_files


# ==============================================================================
# STEP 14b: CATALOG-ONLY MODE (klasifikasi + penomoran + PNG/CSV, TANPA DFSU)
# ==============================================================================

def run_catalog_only():
    """
    Mode catalog_only: parse_area_list → load_land_union → klasifikasi +
    penomoran ulang tile → PNG catalog + CSV mapping. TIDAK butuh DFSU/s100py
    — hanya geopandas+shapely (+matplotlib opsional untuk PNG).

    Berguna untuk evaluasi cepat tata-letak tile & land masking sebelum
    menjalankan pipeline penuh (yang butuh DFSU dan berjalan jauh lebih lama).
    """
    log.info("=" * 62)
    log.info("  MODE: CATALOG-ONLY (klasifikasi + penomoran + PNG/CSV, tanpa DFSU)")
    log.info("=" * 62)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Baca daftar tile
    areas = parse_area_list(AREALIST_PATH)
    log.info(f"Total tile dari area_list: {len(areas)}")

    # 2. Land geometry SEKALI
    full_bbox = (
        min(a["min_lon"] for a in areas), min(a["min_lat"] for a in areas),
        max(a["max_lon"] for a in areas), max(a["max_lat"] for a in areas),
    )
    land_union = load_land_union(LAND_SHP, full_bbox)

    # 3. Klasifikasi + penomoran ulang (tanpa info elemen DFSU: elem_x/elem_y=None)
    classified = classify_and_renumber_tiles(areas, land_union, elem_x=None, elem_y=None)

    # 4. PNG catalog + CSV mapping
    output_files = []
    if MAKE_TILE_CATALOG:
        png_path = plot_tile_catalog(classified, LAND_SHP, os.path.join(OUTPUT_DIR, CATALOG_PNG))
        if png_path:
            output_files.append(png_path)

    csv_path = save_tile_mapping_csv(classified, os.path.join(OUTPUT_DIR, TILE_MAPPING_CSV))
    output_files.append(csv_path)

    n_kept = sum(1 for c in classified if c["keep"])
    log.info("=" * 62)
    log.info(f"SELESAI CATALOG-ONLY — {len(classified)} tile total, {n_kept} kept")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info("=" * 62)

    return output_files


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Entry point utama dengan mode selection."""
    global log

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log = setup_logging(OUTPUT_DIR)

    log.info("=" * 62)
    log.info("  DFSU → S-104 + LAND MASK | Integrated Pipeline v2.0")
    log.info("=" * 62)
    log.info(f"Mode: {PIPELINE_MODE}")
    log.info(f"Multi-grid: {MULTI_GRID} (area_list: {AREALIST_PATH})")
    log.info(f"Split: {SPLIT_MODE}")
    log.info(f"Grid: {GRID_RESOLUTION_M}m")
    log.info(f"Land buffer: {LAND_BUFFER_DEG} deg ({LAND_BUFFER_DEG*111000:.1f}m)")

    if not validate_dependencies(PIPELINE_MODE):
        sys.exit(1)

    try:
        if PIPELINE_MODE == "full":
            output_files = run_multi_grid_pipeline() if MULTI_GRID else run_full_pipeline()
        elif PIPELINE_MODE == "mask_only":
            input_h5 = input("Path ke HDF5 S-104 existing: ").strip().strip('"')
            output_files = run_mask_only(input_h5, OUTPUT_DIR, LAND_SHP)
        elif PIPELINE_MODE == "dfsu_only":
            log.info("Mode dfsu_only: land mask dinonaktifkan")
            original_build_land_mask = build_land_mask
            def no_land_mask(*args, **kwargs):
                ny, nx = len(args[2]), len(args[1])
                return np.zeros((ny, nx), dtype=bool), {"n_land": 0, "n_total": ny*nx}
            globals()["build_land_mask"] = no_land_mask
            output_files = run_multi_grid_pipeline() if MULTI_GRID else run_full_pipeline()
        elif PIPELINE_MODE == "catalog_only":
            output_files = run_catalog_only()
        else:
            log.error(f"Mode tidak dikenal: {PIPELINE_MODE}")
            sys.exit(1)

        log.info(f"Pipeline selesai. Output: {len(output_files)} file")

    except Exception as e:
        log.exception(f"Pipeline gagal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

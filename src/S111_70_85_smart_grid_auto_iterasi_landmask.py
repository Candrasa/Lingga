#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFSU -> S-111 HDF5 PIPELINE — v2.10 LANDMASK REFACTOR
=============================================================================
IHO S-111 Edition 2.0.0 | DCF 2 | WGS84
Multi-Area Grid Support — MIKE FM Hydrodynamic Model

FIX v2.10 (perbaikan dari v2.9):
[F40] classify_grid_by_land_ratio(): klasifikasi grid (darat/laut/campuran)
[F41] mask_extrapolation_boundary(): BUG-U6 fix untuk boundary ekstrapolasi
[F42] generate_new_grid_names(): rename grid laut secara berurutan (Grid01, Grid02, ...)
[F43] save_grid_mapping(): simpan mapping nama lama->baru ke file JSON
[F44] process_one_timestep(): terapkan mask_extrapolation_boundary untuk hapus data bbox terluar darat
[F45] Main: filter grid berdasarkan klasifikasi, terapkan rename grid laut
[F46] Grid-specific landmask: terapkan landmask untuk grid campuran, skip grid darat murni
[F28] Grid-specific output subdirectory untuk menghindari tabrakan filename
[F29] configure_area_from_gridfile: support single name atau list
[F30] Refactor main() -> process_single_grid() untuk pipeline per-grid
[F31] Per-gra land mask, adaptive regrid, dan grid properties
=============================================================================
"""

import gc
import h5py
import os
import re
import logging
import datetime
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import mikeio

# [RENUMBER v3.1] Guarded geo/plot imports — port dari S-104 v2.5 template.
# geopandas/shapely SEBELUMNYA di-import unconditional (pipeline memang butuh
# keduanya); guard ini HANYA menambahkan flag HAS_GEO, tidak mengubah perilaku
# saat library tersedia (yang selalu demikian di lingkungan produksi).
try:
    import geopandas as gpd
    import shapely
    from shapely.geometry import box as shapely_box
    try:
        from shapely import contains_xy
    except ImportError:
        contains_xy = None
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    gpd = None
    shapely = None
    shapely_box = None
    contains_xy = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree
from s100py import s111

warnings.filterwarnings("ignore")

# ==============================================================================
# LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("s111_pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ==============================================================================
# KONFIGURASI GLOBAL
# ==============================================================================

DFSU_PATH   = r"D:\Clivar S-100 Benoa\S100\Lat_S100series (1)\Lat_S100series\HD_PLI_78.dfsu"
OUTPUT_DIR  = r"D:\S-100\S111\s111_70_85_smart_grid_Output_lndmsk_v8"
FILE_PREFIX = "111_70_85_smart_grid_"

OUTPUT_MODE = "daily"   # "daily" | "hourly" | "3day"
BLOCK_DAYS  = 3         # jumlah hari per file saat OUTPUT_MODE == "3day"

# --- AREA CONFIGURATION ---
AREA_MODE    = "gridfile"  # "list" | "manual" | "auto" | "gridfile"

GRID_FILE_PATH = r"D:\S-100\S111\area_list_v1.txt"
GRID_FILE_DX   = 70.0 / 111000.0
GRID_FILE_DY   = 70.0 / 110500.0

# [F32] SELECTED_GRID: daftar grid yang akan diproses.
#   - List nama grid: hanya grid tersebut yang diproses ['Grid1', 'Grid2', 'Grid21']
#   - None atau []: SEMUA grid di file akan diproses otomatis (default v2.9)
SELECTED_GRID  = []

# [F33] MULTI_INSTANCE_PER_FILE: bila True dan AREA_MODE=gridfile dengan
#   SELECTED_GRID kosong (auto-ALL), semua grid ditulis sebagai instance
#   SurfaceCurrent.01 .. SurfaceCurrent.N ke dalam satu HDF5 per hari
#   (S-111 DCF2 multi-instance sesuai spec).
#   Bila False: fallback ke mode lama (satu file per grid per hari).
#   Bila True: satu file multi-instance
MULTI_INSTANCE_PER_FILE = False

AREA_LIST = {
    "selat_lombok": {
        "bbox": [115.157, -8.948, 116.1004, -8.539],
        "dx": 100.0 / 111000.0,
        "dy": 100.0 / 110500.0,
        "desc": "Selat Lombok, NTB"
    },
    "teluk_jakarta": {
        "bbox": [106.85, -6.12, 106.95, -6.07],
        "dx": 70.0 / 111000.0,
        "dy": 70.0 / 110500.0,
        "desc": "Teluk Jakarta"
    },
    "selat_malaka": {
        "bbox": [98.0, 2.0, 104.0, 6.0],
        "dx": 200.0 / 111000.0,
        "dy": 200.0 / 110500.0,
        "desc": "Selat Malaka"
    },
}
SELECTED_AREA = "teluk_jakarta"

BBOX_MANUAL = [106.85, -6.12, 106.95, -6.07]
DX_MANUAL   = 70.0 / 111000.0
DY_MANUAL   = 70.0 / 110500.0

# --- FILE SIZE LIMITER ---
MAX_FILE_SIZE_MB  = 10.0   # Batas ukuran file HDF5 (MB) — v2.7: 10 MB
ENABLE_SIZE_LIMIT = True

# ─────────────────────────────────────────────────────────────────────────────
# [F21][F22][F23] SMART ADAPTIVE REGRID CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
REGRID_MIN_RES_M = 15.0   # meter — JANGAN besarkan tanpa kajian arah arus
REGRID_MAX_SKIP_FACTOR = 4.0
REGRID_ADAPTIVE_STRATEGY = "binary_search"
REGRID_SEARCH_TOL_MB = 0.3  # MB

# ─────────────────────────────────────────────────────────────────────────────

BBOX = [106.85, -6.12, 106.95, -6.07]
DX   = 70.0 / 111000.0
DY   = 70.0 / 110500.0

U_ITEM  = "U velocity"
V_ITEM  = "V velocity"
WD_ITEM = "Total water depth"

DRY_CELL_THRESHOLD = 0.001
NODATA = -9999.0

DATE_START = datetime.date(2026, 9, 9)
DATE_END   = datetime.date(2026, 9, 21)


# ─────────────────────────────────────────────────────────────────────────────
# LAND MASK CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
APPLY_LAND_MASK = True
LAND_SHP        = r"D:\S-100\S111\Shp_78\Area_78.shp"

MASK_MODE       = "land_inside"
# [REFACTOR v3.0] Hapus buffer negatif - menyebabkan masking berlebihan pada sel laut
# Buffer negatif membuat area darat lebih kecil sehingga sel laut di tepi ter-mask
LAND_BUFFER_DEG = 0.0  # Tidak ada buffer - masking hanya pada sel yang benar-benar di dalam shapefile

# [BUG-U6] Boundary Extrapolation Masking
# [REFACTOR 2026-07-05] DIMATIKAN — root cause "grid arus tidak muncul di laut".
# Masking jarak ini men-NODATA-kan sel grid yang > EXTRAP_MAX_DIST_DEG dari
# centroid DFSU terdekat. Pada grid lepas pantai (mesh DFSU renggang), sel
# LAUT valid ikut terhapus. Prinsip baru: landmask = grid arus - shapefile,
# JANGAN hilangkan data arus interpolasi DFSU di area yang tidak di-landmask.
# Masking darat kini HANYA dari shapefile (apply_land_mask_to_h5).
MASK_EXTRAP_BOUNDARY = False  # Matikan mask boundary ekstrapolasi (lihat catatan di atas)
EXTRAP_MAX_DIST_DEG = 0.0015  # (tidak dipakai saat MASK_EXTRAP_BOUNDARY=False)

# [REFACTOR v3.0] Konfigurasi klasifikasi grid dihapus - tidak digunakan lagi
# Filosofi baru: SEMUA grid diproses, landmask hanya menghapus sel darat
GRID_MAPPING_FILE = r"D:\S-100\S111\grid_mapping_v2.10.json"  # Mapping nama lama->baru

NODATA_SPEED = np.float32(-9999.0)
NODATA_DIR   = np.float32(-9999.0)

# ── GRID RENUMBER + CATALOG (port dari S-104 v2.5) ──────────────────────────
ENABLE_GRID_RENUMBER = True
LAND_SKIP_THRESHOLD  = 0.995            # land_frac >= ini -> PURE_LAND -> dibuang
CLASSIFY_SAMPLE_N    = 25               # NxN titik sampel land_frac per grid
GRID_NAME_FORMAT     = "Grid{n:03d}"    # nama baru hasil penomoran ulang
ENABLE_CATALOG_PNG   = True
CATALOG_DPI          = 200
CATALOG_PNG_PATH     = os.path.join(OUTPUT_DIR, "grid_catalog.png")
GRID_MAPPING_CSV     = os.path.join(OUTPUT_DIR, "grid_rename_mapping.csv")
# GRID_MAPPING_FILE (JSON) sudah didefinisikan sebelumnya -> dipakai sbg registry

# ==============================================================================
# AREA CONFIGURATION FUNCTIONS
# ==============================================================================

def configure_area_from_list(area_key: str) -> tuple:
    if area_key not in AREA_LIST:
        raise ValueError(f"Area '{area_key}' tidak ada. Tersedia: {list(AREA_LIST.keys())}")
    cfg = AREA_LIST[area_key]
    log.info(f"[AREA] Mode: list | {area_key} ({cfg['desc']})")
    return cfg["bbox"], cfg["dx"], cfg["dy"]


def parse_grid_file(grid_file_path: str) -> dict:
    grids = {}
    if not os.path.exists(grid_file_path):
        raise FileNotFoundError(f"File grid tidak ditemukan: {grid_file_path}")
    with open(grid_file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) != 5:
                continue
            try:
                name    = parts[0].strip()
                min_lat = float(parts[1]); min_lon = float(parts[2])
                max_lat = float(parts[3]); max_lon = float(parts[4])
                if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90): continue
                if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180): continue
                if min_lat >= max_lat or min_lon >= max_lon: continue
                grids[name] = {"bbox": [min_lon, min_lat, max_lon, max_lat]}
            except ValueError:
                continue
    if not grids:
        raise ValueError(f"Tidak ada grid valid di: {grid_file_path}")
    return grids


def configure_area_from_gridfile(grid_file_path, selected_grid, dx, dy):
    """
    [F29] Support single name (str) atau list of names.
    Mengembalikan dict {name: bbox} agar caller bisa iterasi.
    """
    all_grids = parse_grid_file(grid_file_path)

    # Jika selected_grid adalah string tunggal -> compat v2.7
    if isinstance(selected_grid, str):
        if selected_grid not in all_grids:
            raise ValueError(f"Grid '{selected_grid}' tidak ada. Tersedia: {list(all_grids.keys())}")
        log.info(f"[AREA] Mode: gridfile (single) | {selected_grid} BBOX={all_grids[selected_grid]['bbox']}")
        return {selected_grid: all_grids[selected_grid]}, dx, dy

    # Jika list kosong/None -> gunakan SEMUA grid di file
    if selected_grid is None or len(selected_grid) == 0:
        log.info(f"[AREA] Mode: gridfile (auto ALL) | {len(all_grids)} grid ditemukan")
        return all_grids, dx, dy

    # Jika list nama grid -> filter yang valid
    result = {}
    missing = []
    for name in selected_grid:
        if name in all_grids:
            result[name] = all_grids[name]
        else:
            missing.append(name)
    if missing:
        log.warning(f"[AREA] Grid tidak ditemukan (di-skip): {missing}")
    if not result:
        raise ValueError(f"Tidak ada grid valid dari seleksi. Tersedia: {list(all_grids.keys())}")
    log.info(f"[AREA] Mode: gridfile (selected) | {len(result)}/{len(selected_grid)} grid valid")
    return result, dx, dy


def configure_area_manual(bbox, dx, dy):
    log.info(f"[AREA] Mode: manual | BBOX={bbox}")
    return bbox, dx, dy


def configure_area_auto(ds_open):
    geom   = ds_open.geometry
    elem_x = geom.element_coordinates[:, 0]
    elem_y = geom.element_coordinates[:, 1]
    margin = 0.1
    x_range = elem_x.max() - elem_x.min()
    y_range = elem_y.max() - elem_y.min()
    bbox = [
        float(elem_x.min() - margin * x_range),
        float(elem_y.min() - margin * y_range),
        float(elem_x.max() + margin * x_range),
        float(elem_y.max() + margin * y_range),
    ]
    return bbox, 100.0 / 111000.0, 100.0 / 110500.0

# ==============================================================================
# [F21][F22][F23][F24] SMART ADAPTIVE REGRID — v2.7
# ==============================================================================

def detect_dfsu_resolution(elem_x: np.ndarray, elem_y: np.ndarray,
                            n_samples: int = 500) -> float:
    n_elem = len(elem_x)
    n_samples = min(n_samples, n_elem)
    idx_sample = np.random.choice(n_elem, n_samples, replace=False)

    pts_deg = np.column_stack([elem_x[idx_sample], elem_y[idx_sample]])
    pts_m   = np.column_stack([
        elem_x[idx_sample] * 111000.0 * np.cos(np.radians(elem_y[idx_sample])),
        elem_y[idx_sample] * 110500.0
    ])

    tree = cKDTree(pts_m)
    dists, _ = tree.query(pts_m, k=2)
    nearest_dists = dists[:, 1]

    median_d = np.median(nearest_dists)
    clean    = nearest_dists[nearest_dists < 3.0 * median_d]
    res_m    = float(np.median(clean))
    log.info(f"[DFSU RES] Estimasi resolusi model: {res_m:.1f} m "
             f"(dari {n_samples} sampel elemen)")
    return res_m


def calculate_estimated_file_size(nx: int, ny: int, n_groups: int,
                                   bytes_per_cell: int = 12) -> float:
    """Estimasi ukuran file HDF5 dalam MB (12 bytes/sel × 1.2 overhead × groups)."""
    return (nx * ny * bytes_per_cell * 1.2 * n_groups) / (1024.0 * 1024.0)


def _calc_grid_from_dx(bbox: list, dx_m: float) -> tuple[int, int]:
    """Hitung nx, ny dari resolusi dx_m (meter) dan BBOX."""
    dx_deg = dx_m / 111000.0
    dy_deg = dx_m / 110500.0
    lon_arr = np.arange(bbox[0], bbox[2], dx_deg)
    lat_arr = np.arange(bbox[1], bbox[3], dy_deg)
    return len(lon_arr), len(lat_arr)


def adaptive_grid_resolution(bbox: list, dx: float, dy: float,
                              n_timesteps: int, max_mb: float,
                              dfsu_res_m: float = None) -> tuple:
    dx_m_orig = dx * 111000.0
    dy_m_orig = dy * 110500.0

    lon_arr = np.arange(bbox[0], bbox[2], dx)
    lat_arr = np.arange(bbox[1], bbox[3], dy)
    nx_orig, ny_orig = len(lon_arr), len(lat_arr)
    est_orig = calculate_estimated_file_size(nx_orig, ny_orig, n_timesteps)

    log.info(f"[REGRID] ══════ Smart Adaptive Regrid ══════")
    log.info(f"[REGRID] Grid asli   : {nx_orig}×{ny_orig} = {nx_orig*ny_orig:,} sel")
    log.info(f"[REGRID] Resolusi    : {dx_m_orig:.2f}m × {dy_m_orig:.2f}m")
    log.info(f"[REGRID] Est. size   : {est_orig:.2f} MB | Limit: {max_mb:.1f} MB")
    if dfsu_res_m:
        log.info(f"[REGRID] DFSU res    : {dfsu_res_m:.1f} m")

    if est_orig <= max_mb or not ENABLE_SIZE_LIMIT:
        log.info("[REGRID] ✓ Grid OK — tidak perlu adaptasi")
        return dx, dy, nx_orig, ny_orig, est_orig, dx_m_orig

    dx_max_candidates = []
    dx_max_candidates.append(REGRID_MIN_RES_M)

    if REGRID_MAX_SKIP_FACTOR is not None and dfsu_res_m is not None:
        dx_max_candidates.append(REGRID_MAX_SKIP_FACTOR * dfsu_res_m)

    dx_max_m = min(dx_max_candidates)

    bbox_lon_m = (bbox[2] - bbox[0]) * 111000.0
    bbox_lat_m = (bbox[3] - bbox[1]) * 110500.0
    target_cells = (max_mb * 1024.0 * 1024.0) / (12.0 * 1.2 * n_timesteps)
    dx_fit_m = np.sqrt((bbox_lon_m * bbox_lat_m) / max(target_cells, 1.0))
    log.info(f"[REGRID] dx untuk fit {max_mb:.1f} MB: {dx_fit_m:.2f} m (analitik)")

    if dx_fit_m > dx_max_m:
        log.warning(
            f"[REGRID] KONFLIK CONSTRAINT:\n"
            f"          dx butuhkan untuk fit file: {dx_fit_m:.2f} m\n"
            f"          dx max (resolusi/skip):     {dx_max_m:.2f} m\n"
            f"          -> Prioritas KUALITAS ARAH ARUS dipertahankan.\n"
            f"          -> File AKAN > {max_mb:.1f} MB. Pertimbangkan:\n"
            f"             (1) Perkecil DATE range\n"
            f"             (2) Perkecil BBOX area studi\n"
            f"             (3) Besarkan MAX_FILE_SIZE_MB\n"
            f"             (4) Besarkan REGRID_MIN_RES_M"
        )
        dx_new_m = dx_max_m
        dx_new   = dx_new_m / 111000.0
        dy_new   = dx_new_m / 110500.0
        lon_arr  = np.arange(bbox[0], bbox[2], dx_new)
        lat_arr  = np.arange(bbox[1], bbox[3], dy_new)
        nx_new, ny_new = len(lon_arr), len(lat_arr)
        est_new  = calculate_estimated_file_size(nx_new, ny_new, n_timesteps)
    elif REGRID_ADAPTIVE_STRATEGY == "binary_search":
        lo = dx_m_orig
        hi = dx_fit_m * 1.2
        best_dx_m = hi
        best_nx = best_ny = 0

        for _iter in range(50):
            mid = (lo + hi) / 2.0
            dx_deg = mid / 111000.0
            dy_deg = mid / 110500.0
            nx_t = len(np.arange(bbox[0], bbox[2], dx_deg))
            ny_t = len(np.arange(bbox[1], bbox[3], dy_deg))
            est_t = calculate_estimated_file_size(nx_t, ny_t, n_timesteps)

            if est_t <= max_mb:
                best_dx_m = mid
                best_nx, best_ny = nx_t, ny_t
                hi = mid
            else:
                lo = mid

            if (hi - lo) < 0.05:
                break

        dx_new_m = max(best_dx_m, dx_m_orig)
        dx_new_m = min(dx_new_m, dx_max_m)
        dx_new   = dx_new_m / 111000.0
        dy_new   = dx_new_m / 110500.0
        lon_arr  = np.arange(bbox[0], bbox[2], dx_new)
        lat_arr  = np.arange(bbox[1], bbox[3], dy_new)
        nx_new, ny_new = len(lon_arr), len(lat_arr)
        est_new  = calculate_estimated_file_size(nx_new, ny_new, n_timesteps)
    else:
        scale = 1.0
        dx_new, dy_new, nx_new, ny_new = dx, dy, nx_orig, ny_orig
        est_new = est_orig
        while est_new > max_mb and (dx * scale * 111000.0) < dx_max_m:
            scale += 0.1
            dx_new = dx * scale
            dy_new = dy * scale
            nx_new = len(np.arange(bbox[0], bbox[2], dx_new))
            ny_new = len(np.arange(bbox[1], bbox[3], dy_new))
            est_new = calculate_estimated_file_size(nx_new, ny_new, n_timesteps)
        dx_new_m = dx_new * 111000.0

    skip_ratio = dx_new_m / dx_m_orig if dx_m_orig > 0 else 1.0
    log.info(f"[REGRID] ── Hasil Adaptasi ──────────────────")
    log.info(f"[REGRID] Grid output: {nx_new}×{ny_new} = {nx_new*ny_new:,} sel")
    log.info(f"[REGRID] Resolusi    : {dx_new_m:.2f} m")
    log.info(f"[REGRID] Est. size   : {est_new:.2f} MB")
    log.info(f"[REGRID] Skip ratio  : {skip_ratio:.2f}× dari resolusi asli ({dx_m_orig:.1f} m)")
    if dfsu_res_m:
        log.info(f"[REGRID] Skip vs DFSU: {dx_new_m/dfsu_res_m:.1f}× resolusi model DFSU")
    log.info(f"[REGRID] ════════════════════════════════════")

    return dx_new, dy_new, nx_new, ny_new, est_new, dx_new_m

# ==============================================================================
# [F25] REGRID U DAN V SECARA BERSAMAAN — preservasi arah arus
# ==============================================================================

def regrid_uv_to_s111(elem_x: np.ndarray, elem_y: np.ndarray,
                       raw_u: np.ndarray, raw_v: np.ndarray,
                       grid_xx: np.ndarray, grid_yy: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(raw_u) & np.isfinite(raw_v)

    if valid.sum() < 3:
        log.warning(f"  [REGRID] < 3 titik valid U+V ({valid.sum()}), return NODATA")
        nodata_grid = np.full(grid_xx.shape, NODATA, dtype=np.float32)
        return nodata_grid, nodata_grid.copy()

    pts   = np.column_stack([elem_x[valid], elem_y[valid]])
    u_val = raw_u[valid]
    v_val = raw_v[valid]

    interp_u = LinearNDInterpolator(pts, u_val)
    interp_v = LinearNDInterpolator(pts, v_val)

    u_grid = interp_u(grid_xx, grid_yy)
    v_grid = interp_v(grid_xx, grid_yy)

    nan_mask = np.isnan(u_grid) | np.isnan(v_grid)
    if nan_mask.any():
        nn_u = NearestNDInterpolator(pts, u_val)
        nn_v = NearestNDInterpolator(pts, v_val)
        query_pts = np.column_stack([grid_xx[nan_mask], grid_yy[nan_mask]])
        u_grid[nan_mask] = nn_u(query_pts)
        v_grid[nan_mask] = nn_v(query_pts)

    u_out = np.where(np.isfinite(u_grid), u_grid, NODATA).astype(np.float32)
    v_out = np.where(np.isfinite(v_grid), v_grid, NODATA).astype(np.float32)
    return u_out, v_out


def regrid_to_s111(elem_x, elem_y, values, grid_xx, grid_yy):
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return np.full(grid_xx.shape, NODATA, dtype=np.float32)
    pts      = np.column_stack([elem_x[valid], elem_y[valid]])
    vals     = values[valid]
    grid_out = LinearNDInterpolator(pts, vals)(grid_xx, grid_yy)
    nan_mask = np.isnan(grid_out)
    if nan_mask.any():
        query_pts = np.column_stack([grid_xx[nan_mask], grid_yy[nan_mask]])
        grid_out[nan_mask] = NearestNDInterpolator(pts, vals)(query_pts)
    return np.where(np.isfinite(grid_out), grid_out, NODATA).astype(np.float32)

# ==============================================================================
# LAND MASK FUNCTIONS — v2.6
# ==============================================================================

def build_land_mask(shp_path: str, grid_properties: dict, land_union=None) -> np.ndarray:
    """[F16][F17][F20][F47][REFACTOR v3.0] Build land mask dari shapefile dengan GDAL rasterisasi optimal.

    Perbaikan v3.0:
    - Filosofi baru: SEMUA grid diproses, landmask hanya menghapus sel darat
    - Hapus buffer negatif yang menyebabkan masking berlebihan
    - Validasi CRS shapefile sebelum proses
    - Optimasi memori dengan rasterisasi langsung tanpa meshgrid

    [RENUMBER v3.1] Param opsional land_union: bila diberikan (hasil
    load_land_union, dibaca SEKALI untuk seluruh area gabungan), shapefile
    TIDAK dibaca ulang per-grid — mask dihitung langsung dari geometry via
    shapely contains_xy pada titik pusat sel grid. Orientasi flat 1D HARUS
    identik dengan jalur GDAL (row=lat menaik, col=lon menaik, ravel
    order="C") — lihat rekonstruksi lon_arr/lat_arr di bawah, yang
    mereproduksi PERSIS np.arange(bbox[0], bbox[2], dx) dari
    build_grid_and_properties (start & step sama -> hasil arange identik).
    Jika land_union None, perilaku SAMA PERSIS seperti sebelumnya (path GDAL
    asli, backward compatible, default).
    """
    if not APPLY_LAND_MASK:
        return None

    nx   = grid_properties["nx"];   ny   = grid_properties["ny"]
    minx = grid_properties["minx"]; miny = grid_properties["miny"]
    maxx = grid_properties["maxx"]; maxy = grid_properties["maxy"]

    if land_union is not None:
        if not HAS_GEO:
            log.warning("[LANDMASK] land_union diberikan tapi HAS_GEO=False — abaikan reuse")
        else:
            log.info(f"[LANDMASK] reuse land_union (skip baca shapefile per-grid) | Mode: {MASK_MODE}")
            log.info(f"[LANDMASK] BBOX: ({minx:.6f},{miny:.6f}) -> ({maxx:.6f},{maxy:.6f}) | {nx}×{ny}")

            dx = grid_properties.get("cellsize_x")
            dy = grid_properties.get("cellsize_y")
            if dx and dy:
                # Reproduksi PERSIS lon_arr/lat_arr dari build_grid_and_properties
                # (arange dengan start & step identik -> hasil identik).
                lon_arr = np.arange(minx, maxx + dx * 0.5, dx)[:nx]
                lat_arr = np.arange(miny, maxy + dy * 0.5, dy)[:ny]
            else:
                lon_arr = np.linspace(minx, maxx, nx)
                lat_arr = np.linspace(miny, maxy, ny)

            lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
            lon_flat = lon_grid.ravel(order="C")
            lat_flat = lat_grid.ravel(order="C")

            shapely_ver = tuple(int(x) for x in shapely.__version__.split(".")[:2])
            if shapely_ver >= (2, 0):
                from shapely import contains_xy as _cxy
                inside_mask = _cxy(land_union, lon_flat, lat_flat)
            else:
                from shapely.vectorized import contains as vcontains
                inside_mask = vcontains(land_union, lon_flat, lat_flat)

            land_mask = ~inside_mask if MASK_MODE == "clip_outside" else inside_mask

            n_masked = int(land_mask.sum())
            n_total  = len(land_mask)
            log.info(f"[LANDMASK] (reuse) Di-mask: {n_masked:,} | Valid: {n_total-n_masked:,} "
                     f"({100*(n_total-n_masked)/n_total:.1f}%)")
            return land_mask

    if not Path(shp_path).exists():
        raise FileNotFoundError(f"Shapefile tidak ditemukan: {shp_path}")

    log.info(f"[LANDMASK] Shapefile: {shp_path} | Mode: {MASK_MODE}")
    log.info(f"[LANDMASK] BBOX: ({minx:.6f},{miny:.6f}) -> ({maxx:.6f},{maxy:.6f}) | {nx}×{ny}")

    # [F47] Validasi dan baca shapefile dengan GeoPandas
    try:
        gdf = gpd.read_file(shp_path, bbox=(minx, miny, maxx, maxy))
    except Exception as e:
        log.error(f"[LANDMASK] Gagal membaca shapefile: {e}")
        raise

    if gdf.empty:
        log.warning("[LANDMASK] Shapefile kosong di BBOX")
        if MASK_MODE == "clip_outside":
            log.error("[LANDMASK] Mode clip_outside dengan shapefile kosong — semua sel di-mask!")
            return np.ones(nx * ny, dtype=bool)
        return np.zeros(nx * ny, dtype=bool)

    # [F47] Validasi CRS shapefile
    if gdf.crs is None:
        log.warning("[LANDMASK] Shapefile tidak memiliki CRS — asumsikan EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        log.info(f"[LANDMASK] Reproject shapefile dari EPSG:{gdf.crs.to_epsg()} ke EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")

    # [F47] Gunakan GDAL rasterisasi untuk performa optimal
    try:
        from osgeo import gdal, ogr
        import tempfile

        # Buat shapefile sementara untuk GDAL
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_shp = Path(tmpdir) / "temp_landmask.shp"
            gdf.to_file(temp_shp, driver="ESRI Shapefile")

            # Setup raster output
            temp_tiff_raster = Path(tmpdir) / "temp_raster.tif"

            # Hitung resolusi raster
            dx = (maxx - minx) / (nx - 1) if nx > 1 else 0.001
            dy = (maxy - miny) / (ny - 1) if ny > 1 else 0.001

            # Buat raster kosong
            raster_ds = gdal.GetDriverByName('GTiff').Create(
                str(temp_tiff_raster), nx, ny, 1, gdal.GDT_Byte
            )
            # [FIX] Pixel-center alignment: tempatkan pusat piksel tepat pada
            # node grid S-111 (minx + j*dx, maxy - i*dy). Rasterisasi GDAL
            # menandai piksel berdasarkan PUSAT-nya, jadi origin harus digeser
            # setengah piksel agar mask sejajar titik sampel (bukan geser ~½ sel).
            raster_ds.SetGeoTransform([minx - dx / 2.0, dx, 0, maxy + dy / 2.0, 0, -dy])
            raster_ds.SetProjection('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')

            # Rasterisasi shapefile
            shape_ds = ogr.Open(str(temp_shp))
            shape_layer = shape_ds.GetLayer()

            # Burn value: 1 untuk darat, 0 untuk laut
            gdal.RasterizeLayer(raster_ds, [1], shape_layer, burn_values=[1])

            # Baca hasil hasil rasterisasi
            band = raster_ds.GetRasterBand(1)
            raster_array = band.ReadAsArray()

            # Cleanup
            raster_ds = None
            shape_ds = None

            # [FIX] GDAL row 0 = Utara (maxy); grid S-111 row 0 = Selatan
            # (miny, lat menaik dari meshgrid(lon_arr, lat_arr)). Flip vertikal
            # agar urutan flat C-order (row=lat menaik, col=lon) IDENTIK dengan
            # data HDF5 dan fallback Shapely. JANGAN transpose (.T menukar
            # sumbu lat/lon -> mask teracak).
            raster_flat = np.flipud(raster_array).ravel()
            # burn_values=[1] -> 1 di dalam polygon darat, 0 di laut.
            if MASK_MODE == "clip_outside":
                # Mode khusus: mask area di LUAR polygon (laut)
                land_mask = raster_flat == 0
            else:
                # Default (land_inside): mask sel DARAT di dalam polygon
                land_mask = raster_flat == 1

    except ImportError:
        log.warning("[LANDMASK] GDAL tidak tersedia, fallback ke Shapely")
        # Fallback ke metode Shapely asli
        if not gdf.empty:
            bbox_geom = shapely_box(minx, miny, maxx, maxy)
            gdf = gdf.copy()
            gdf["geometry"] = gdf["geometry"].intersection(bbox_geom)
            gdf = gdf[~gdf["geometry"].is_empty & gdf["geometry"].notna()]

        if gdf.empty:
            if MASK_MODE == "clip_outside":
                log.error("[LANDMASK] Shapefile kosong di BBOX — semua sel di-mask!")
                return np.ones(nx * ny, dtype=bool)
            return np.zeros(nx * ny, dtype=bool)

        try:
            land_union = gdf.geometry.union_all()
        except AttributeError:
            from shapely.ops import unary_union
            land_union = unary_union(gdf.geometry.values)

        # [REFACTOR v3.0] Buffer sudah diset ke 0, tidak perlu buffer lagi
        # if LAND_BUFFER_DEG != 0.0:
        #     land_union = land_union.buffer(LAND_BUFFER_DEG)

        lon_arr = np.linspace(minx, maxx, nx)
        lat_arr = np.linspace(miny, maxy, ny)
        lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
        lon_flat = lon_grid.ravel(order="C")
        lat_flat = lat_grid.ravel(order="C")

        shapely_ver = tuple(int(x) for x in shapely.__version__.split(".")[:2])
        if shapely_ver >= (2, 0):
            from shapely import contains_xy
            inside_mask = contains_xy(land_union, lon_flat, lat_flat)
        else:
            from shapely.vectorized import contains as vcontains
            inside_mask = vcontains(land_union, lon_flat, lat_flat)

        land_mask = ~inside_mask if MASK_MODE == "clip_outside" else inside_mask

    n_masked = int(land_mask.sum())
    n_total  = len(land_mask)
    log.info(f"[LANDMASK] Di-mask: {n_masked:,} | Valid: {n_total-n_masked:,} "
             f"({100*(n_total-n_masked)/n_total:.1f}%)")

    if n_masked == 0:
        log.warning("[LANDMASK] 0 sel di-mask — periksa MASK_MODE dan shapefile.")
    elif n_masked == n_total:
        log.error("[LANDMASK] SEMUA sel di-mask! Kemungkinan shapefile terbalik.")

    return land_mask


def apply_land_mask_to_h5(h5_path: str, land_mask: np.ndarray,
                           grid_properties: dict) -> dict:
    """[F18][F19] Terapkan land mask dengan validasi ukuran."""
    if land_mask is None or land_mask.sum() == 0:
        return {"n_groups": 0, "n_masked": 0, "errors": 0}

    n_masked       = int(land_mask.sum())
    expected_size  = grid_properties["nx"] * grid_properties["ny"]
    summary        = {"n_groups": 0, "n_masked": n_masked, "errors": 0}
    fname          = Path(h5_path).name
    SC_INSTANCE    = "SurfaceCurrent/SurfaceCurrent.01"

    if len(land_mask) != expected_size:
        log.error(f"[LANDMASK] Ukuran mask ({len(land_mask)}) ≠ nx*ny ({expected_size})")
        summary["errors"] += 1
        return summary

    with h5py.File(h5_path, "r+") as f:
        if SC_INSTANCE not in f:
            summary["errors"] += 1
            return summary

        inst = f[SC_INSTANCE]
        group_paths = sorted([
            f"{SC_INSTANCE}/{n}" for n in inst.keys()
            if n.startswith("Group_") and "values" in inst[n]
        ])
        if not group_paths:
            return summary

        speed_global_min =  np.inf
        speed_global_max = -np.inf

        for grp_path in group_paths:
            ds_path  = f"{grp_path}/values"
            grp_name = grp_path.split("/")[-1]
            if ds_path not in f:
                summary["errors"] += 1
                continue

            ds  = f[ds_path]
            arr = ds[:]
            orig_shape = arr.shape
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            if len(arr) != expected_size or arr.dtype.names is None:
                summary["errors"] += 1
                continue

            speed_field = arr.dtype.names[0]
            dir_field   = arr.dtype.names[1]
            # Fixed: Use np.where for structured array assignment to avoid broadcasting error
            arr[speed_field] = np.where(land_mask, float(NODATA_SPEED), arr[speed_field])
            arr[dir_field]   = np.where(land_mask, float(NODATA_DIR), arr[dir_field])
            ds[:] = arr.reshape(orig_shape)

            water_mask   = (~land_mask) & (arr[speed_field] > float(NODATA_SPEED) + 1.0)
            valid_speeds = arr[speed_field][water_mask]
            sp_min = float(valid_speeds.min()) if valid_speeds.size > 0 else float(NODATA_SPEED)
            sp_max = float(valid_speeds.max()) if valid_speeds.size > 0 else float(NODATA_SPEED)
            speed_global_min = min(speed_global_min, sp_min)
            speed_global_max = max(speed_global_max, sp_max)

            grp_obj = f[grp_path]
            for k in ["minimumCurrentSpeed", "minDatasetCurrentSpeed"]:
                if k in grp_obj.attrs: grp_obj.attrs[k] = np.float32(sp_min)
            for k in ["maximumCurrentSpeed", "maxDatasetCurrentSpeed"]:
                if k in grp_obj.attrs: grp_obj.attrs[k] = np.float32(sp_max)
            summary["n_groups"] += 1

        if speed_global_min < np.inf:
            for k in ["minDatasetCurrentSpeed", "minimumCurrentSpeed"]:
                if k in inst.attrs: inst.attrs[k] = np.float32(speed_global_min)
            for k in ["maxDatasetCurrentSpeed", "maximumCurrentSpeed"]:
                if k in inst.attrs: inst.attrs[k] = np.float32(speed_global_max)

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        f.attrs["landMaskApplied"]    = b"TRUE"
        f.attrs["landMaskSource"]     = Path(LAND_SHP).name.encode("utf-8")
        f.attrs["landMaskMode"]       = MASK_MODE.encode("utf-8")
        f.attrs["landMaskDate"]       = now.encode("utf-8")
        f.attrs["landMaskCellCount"]  = np.uint32(n_masked)
        f.attrs["landMaskBuffer_deg"] = np.float32(LAND_BUFFER_DEG)
        f.attrs["landMaskStandard"]   = b"S_111_Ed_2_0_0_fillValue_-9999.0"

        existing = f.attrs.get("history", b"")
        if isinstance(existing, bytes):
            existing = existing.decode("utf-8", errors="replace")
        f.attrs["history"] = (
            f"{existing}\n{now} — Land mask: {MASK_MODE}, "
            f"cells={n_masked:,}, groups={summary['n_groups']}."
        ).encode("utf-8")

    log.info(f"[LANDMASK] [{fname}] {summary['n_groups']} groups, {n_masked:,} sel")
    return summary


def apply_land_mask_to_h5_multi(h5_path: str, inst_idx: int,
                                  land_mask: np.ndarray,
                                  grid_properties: dict) -> dict:
    """
    [F35] Land mask untuk instance SurfaceCurrent.<NN> tertentu pada HDF5
    multi-instance. NN diformat 2-digit zero-padded (01, 02, ..., 21).
    """
    if land_mask is None or land_mask.sum() == 0:
        return {"n_groups": 0, "n_masked": 0, "errors": 0}

    n_masked      = int(land_mask.sum())
    expected_size = grid_properties["nx"] * grid_properties["ny"]
    summary       = {"n_groups": 0, "n_masked": n_masked, "errors": 0}
    fname         = Path(h5_path).name
    inst_name     = f"SurfaceCurrent.{inst_idx:02d}"
    SC_INSTANCE   = f"SurfaceCurrent/{inst_name}"

    if len(land_mask) != expected_size:
        log.error(f"[LANDMASK-MULTI] [{inst_name}] "
                  f"size mismatch ({len(land_mask)} ≠ {expected_size})")
        summary["errors"] += 1
        return summary

    with h5py.File(h5_path, "r+") as f:
        if SC_INSTANCE not in f:
            log.error(f"[LANDMASK-MULTI] [{inst_name}] tidak ada di {fname}")
            summary["errors"] += 1
            return summary

        inst = f[SC_INSTANCE]
        group_paths = sorted([
            f"{SC_INSTANCE}/{n}" for n in inst.keys()
            if n.startswith("Group_") and "values" in inst[n]
        ])
        if not group_paths:
            return summary

        speed_inst_min =  np.inf
        speed_inst_max = -np.inf

        for grp_path in group_paths:
            ds_path  = f"{grp_path}/values"
            if ds_path not in f:
                summary["errors"] += 1
                continue

            ds  = f[ds_path]
            arr = ds[:]
            orig_shape = arr.shape
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            if len(arr) != expected_size or arr.dtype.names is None:
                summary["errors"] += 1
                continue

            speed_field = arr.dtype.names[0]
            dir_field   = arr.dtype.names[1]
            # Fixed: Use np.where for structured array assignment to avoid broadcasting error
            arr[speed_field] = np.where(land_mask, float(NODATA_SPEED), arr[speed_field])
            arr[dir_field]   = np.where(land_mask, float(NODATA_DIR), arr[dir_field])
            ds[:] = arr.reshape(orig_shape)

            water_mask   = (~land_mask) & (arr[speed_field] > float(NODATA_SPEED) + 1.0)
            valid_speeds = arr[speed_field][water_mask]
            sp_min = float(valid_speeds.min()) if valid_speeds.size > 0 else float(NODATA_SPEED)
            sp_max = float(valid_speeds.max()) if valid_speeds.size > 0 else float(NODATA_SPEED)
            speed_inst_min = min(speed_inst_min, sp_min)
            speed_inst_max = max(speed_inst_max, sp_max)

            grp_obj = f[grp_path]
            for k in ["minimumCurrentSpeed", "minDatasetCurrentSpeed"]:
                if k in grp_obj.attrs: grp_obj.attrs[k] = np.float32(sp_min)
            for k in ["maximumCurrentSpeed", "maxDatasetCurrentSpeed"]:
                if k in grp_obj.attrs: grp_obj.attrs[k] = np.float32(sp_max)
            summary["n_groups"] += 1

        if speed_inst_min < np.inf:
            for k in ["minDatasetCurrentSpeed", "minimumCurrentSpeed"]:
                if k in inst.attrs: inst.attrs[k] = np.float32(speed_inst_min)
            for k in ["maxDatasetCurrentSpeed", "maximumCurrentSpeed"]:
                if k in inst.attrs: inst.attrs[k] = np.float32(speed_inst_max)

    log.info(f"[LANDMASK-MULTI] [{fname}/{inst_name}] "
             f"{summary['n_groups']} groups, {n_masked:,} sel")
    return summary


def validate_land_masked_h5(h5_path: str, land_mask: np.ndarray) -> bool:
    if land_mask is None or land_mask.sum() == 0:
        return True
    errors      = []
    fname       = Path(h5_path).name
    SC_INSTANCE = "SurfaceCurrent/SurfaceCurrent.01"

    with h5py.File(h5_path, "r") as f:
        if SC_INSTANCE not in f:
            return False
        inst = f[SC_INSTANCE]
        group_paths = sorted([
            f"{SC_INSTANCE}/{n}" for n in inst.keys()
            if n.startswith("Group_") and "values" in inst[n]
        ])
        check_groups = ([group_paths[0]] +
                        ([group_paths[-1]] if len(group_paths) > 1 else []))

        for grp_path in check_groups:
            ds_path = f"{grp_path}/values"
            if ds_path not in f:
                continue
            arr = f[ds_path][:]
            if arr.ndim != 1: arr = arr.reshape(-1)
            if len(arr) != len(land_mask):
                errors.append(f"{grp_path}: size mismatch")
                continue
            speeds = arr[arr.dtype.names[0]]
            wrong  = speeds[land_mask][speeds[land_mask] > float(NODATA_SPEED) + 1.0]
            if len(wrong) > 0:
                errors.append(f"{grp_path}: {len(wrong):,} sel masih valid")
            valid_w = speeds[~land_mask][speeds[~land_mask] > float(NODATA_SPEED) + 1.0]
            log.info(f"  [{fname}] {grp_path.split('/')[-1]}: "
                     f"valid_water={valid_w.size:,}")

        if "landMaskApplied" not in f.attrs:
            errors.append("attr landMaskApplied tidak ada")

    for e in errors: log.error(f"  VALIDATION: {e}")
    return len(errors) == 0


# ==============================================================================
# [REFACTOR v3.0] Fungsi classify_grid_by_land_ratio() dihapus
# Filosofi baru: SEMUA grid diproses, tidak ada klasifikasi
# ==============================================================================

# ==============================================================================
# [LMK-02] MASK EXTRAPOLATION BOUNDARY — BUG-U6
# ==============================================================================

def mask_extrapolation_boundary(u_grid: np.ndarray, v_grid: np.ndarray,
                                 grid_xx: np.ndarray, grid_yy: np.ndarray,
                                 elem_x_surf: np.ndarray,
                                 elem_y_surf: np.ndarray) -> tuple:
    """
    [BUG-U6] Set NODATA untuk sel grid yang terlalu jauh dari elemen DFSU
    terdekat. Ini mengatasi boundary extrapolation yang menghasilkan data
    arus palsu di area darat di luar jangkauan model.

    Args:
        u_grid, v_grid:  Grid 2D hasil regrid (float32)
        grid_xx, grid_yy: Meshgrid koordinat grid (2D)
        elem_x_surf, elem_y_surf: Koordinat elemen DFSU dalam BBOX

    Returns:
        (u_grid, v_grid) dengan boundary extrapolation di-mask NODATA
    """
    from scipy.spatial import KDTree

    elem_pts = np.column_stack([elem_x_surf.ravel(), elem_y_surf.ravel()])
    tree     = KDTree(elem_pts)
    flat_xx  = grid_xx.ravel()
    flat_yy  = grid_yy.ravel()
    query_pts = np.column_stack([flat_xx, flat_yy])

    dists, _ = tree.query(query_pts, k=1)
    dists    = dists.reshape(grid_xx.shape)

    far_mask = dists > EXTRAP_MAX_DIST_DEG
    n_far    = int(far_mask.sum())

    if n_far > 0:
        u_grid = np.where(far_mask, NODATA, u_grid).astype(np.float32)
        v_grid = np.where(far_mask, NODATA, v_grid).astype(np.float32)
        log.info(f"[EXTRAP-BOUNDARY] Mask {n_far:,} sel > {EXTRAP_MAX_DIST_DEG}° "
                 f"dari elemen DFSU terdekat")
    else:
        log.info("[EXTRAP-BOUNDARY] Tidak ada sel boundary yang perlu di-mask")

    return u_grid, v_grid


# ==============================================================================
# [LMK-03] GENERATE NEW GRID NAMES & MAPPING
# ==============================================================================

def generate_new_grid_names(valid_grids: list) -> dict:
    """
    Hasilkan mapping nama grid lama -> baru dengan urutan zero-padded.

    Args:
        valid_grids: List nama grid yang valid (sudah difilter, urut)

    Returns:
        Dict {nama_lama: nama_baru}, misal {"Grid1": "Grid01", "Grid5": "Grid02"}
    """
    n_digits = len(str(len(valid_grids)))
    mapping  = {}
    for i, old_name in enumerate(valid_grids, 1):
        new_name = f"Grid{i:0{n_digits}d}"
        mapping[old_name] = new_name
    return mapping


def save_grid_mapping(mapping: dict, output_path: str,
                      grids: dict = None):
    """
    Simpan mapping grid lama -> baru ke file CSV.

    Args:
        mapping:      Dict {old_name: new_name}
        output_path:  Path tujuan file CSV
        grids:        Dict opsional {grid_name: {"bbox": [...]}} untuk
                      menyertakan koordinat BBOX di CSV.

    Format: old_name,new_name,minx,miny,maxx,maxy
    """
    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["old_name", "new_name", "minx", "miny", "maxx", "maxy"])
        for old, new in sorted(mapping.items()):
            if grids and old in grids:
                b = grids[old].get("bbox", ["", "", "", ""])
                writer.writerow([old, new, b[0], b[1], b[2], b[3]])
            else:
                writer.writerow([old, new, "", "", "", ""])
    log.info(f"[MAPPING] Grid mapping saved: {output_path} ({len(mapping)} entries)")


def classify_and_filter_grids(grids_to_process: dict, output_dir: str) -> tuple:
    """
    [LMK-04] [F48] Generate rename mapping untuk SEMUA grid — TANPA filtering.

    Perbaikan v2.12:
    - Hapus filtering grid berdasarkan klasifikasi darat/laut
    - Semua grid yang memiliki data arus DFSU diproses
    - Landmask diterapkan per-grid untuk menghapus sel darat saja

    Args:
        grids_to_process: Dict {grid_name: {"bbox": [...], ...}}
        output_dir: Direktori output untuk menyimpan mapping file

    Returns:
        (grids_filtered, grid_rename_mapping, grid_classifications)
        - grids_filtered: Dict {grid_name: grid_cfg} SEMUA grid (no filtering)
        - grid_rename_mapping: Dict {old_name: new_name}
        - grid_classifications: Dict {grid_name: "land"|"sea"|"mixed"}
    """
    from tqdm import tqdm
    import sys

    log.info("")
    log.info("=" * 68)
    log.info("  [LMK-04] GENERATE RENAME MAPPING [F48 - NO FILTERING]")
    log.info("=" * 68)

    # [F48] Pertahankan SEMUA grid — tidak ada filtering
    grids_filtered = dict(grids_to_process)
    grid_classifications = {}

    # [REFACTOR v3.0] Tidak ada klasifikasi grid - semua grid diproses
    log.info("  [REFACTOR v3.0] SEMUA grid akan diproses (tidak ada klasifikasi)")

    # Generate rename mapping untuk SEMUA grid
    valid_grid_names = sorted(grids_filtered.keys())
    grid_rename_mapping = generate_new_grid_names(valid_grid_names)

    # Save mapping ke CSV
    mapping_path = os.path.join(output_dir, "grid_rename_mapping.csv")
    save_grid_mapping(grid_rename_mapping, mapping_path, grids=grids_filtered)

    # Ringkasan
    n_total = len(grids_to_process)
    n_filtered = len(grids_filtered)

    log.info("")
    log.info("  [REFACTOR v3.0] RINGKASAN")
    log.info(f"    Total grid    : {n_total}")
    log.info(f"    Grid diproses : {n_filtered} (SEMUA grid)")
    log.info(f"    Mapping file  : {mapping_path}")
    log.info("")

    return grids_filtered, grid_rename_mapping, {}


# ==============================================================================
# [RENUMBER v3.1] GEOGRAPHIC + LAND-AWARE GRID RENUMBER — port dari S-104 v2.5
# ==============================================================================

def load_land_union(shp_path: str, full_bbox: tuple, buffer_deg: float = LAND_BUFFER_DEG):
    """
    Baca shapefile daratan SEKALI (clipped ke full_bbox gabungan semua grid),
    lalu union_all() + buffer. Dipakai agar N grid tidak masing-masing
    membaca shapefile sendiri-sendiri (mahal untuk ratusan grid).

    Args:
        shp_path : path shapefile.
        full_bbox: (minlon, minlat, maxlon, maxlat) — bbox gabungan seluruh grid.
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


def classify_and_renumber_grids(grids_to_process: dict, land_union, elem_x=None, elem_y=None) -> tuple:
    """
    Klasifikasi tiap grid (PURE_LAND/MIXED/PURE_SEA/NO_DATA) dan penomoran
    ulang spasial (row=0 di selatan, col=0 di barat; urutan S->N lalu W->E).

    Grid yang PURE_LAND (land_frac >= LAND_SKIP_THRESHOLD) atau NO_DATA
    (tidak ada elemen DFSU di dalam bbox, bila elem_x/elem_y diberikan)
    dikeluarkan (keep=False) dari output S-111 — tapi tetap dilaporkan
    di CSV/JSON mapping & PNG catalog untuk keperluan evaluasi.

    Args:
        grids_to_process: dict {name: {"bbox": [minlon, minlat, maxlon, maxlat], ...}}
        (format S-111 parse_grid_file — bbox lon-first).
        land_union: shapely geometry (hasil load_land_union) atau None
        (-> land_frac=0.0 untuk semua grid, semua dianggap PURE_SEA).
        elem_x, elem_y: koordinat elemen DFSU (opsional). Jika None, n_elem
        tidak dihitung (None) dan tidak dipakai untuk exclude.

    Returns:
        (kept_ordered, mapping, classified)
        - kept_ordered: dict {old_name: cfg} — HANYA grid KEPT, urut new_num.
        - mapping: dict {old_name: new_name} — HANYA grid KEPT.
        - classified: list[dict] — semua grid (termasuk yang di-skip), keys:
        old_name, bbox, center_lon, center_lat, land_frac, n_elem,
        tile_class, keep, row, col, new_name, new_num.
    """
    log.info("=" * 62)
    log.info("  KLASIFIKASI & PENOMORAN ULANG GRID")
    log.info("=" * 62)

    classified = []

    for old_name, cfg in grids_to_process.items():
        minlon, minlat, maxlon, maxlat = cfg["bbox"]
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
            if shapely_ver >= (2, 0) and contains_xy is not None:
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

        if not ENABLE_GRID_RENUMBER:
            keep = True

        classified.append({
            "old_name": old_name,
            "bbox": [minlon, minlat, maxlon, maxlat],
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

    # --- ordering & penomoran: grid KEPT, urut (row asc, col asc) ---
    kept = [c for c in classified if c["keep"]]
    kept_sorted = sorted(kept, key=lambda c: (c["row"], c["col"]))

    for i, c in enumerate(kept_sorted, start=1):
        c["new_num"] = i
        c["new_name"] = (c["old_name"] if not ENABLE_GRID_RENUMBER
                         else GRID_NAME_FORMAT.format(n=i))

    # --- summary log ---
    n_total = len(classified)
    n_kept = len(kept)
    n_pure_land = sum(1 for c in classified if c["tile_class"] == "PURE_LAND")
    n_no_data = sum(1 for c in classified if c["tile_class"] == "NO_DATA")
    n_mixed = sum(1 for c in classified if c["tile_class"] == "MIXED")
    n_pure_sea = sum(1 for c in classified if c["tile_class"] == "PURE_SEA")

    log.info(f"  Total grid     : {n_total}")
    log.info(f"  Kept (output)  : {n_kept}")
    log.info(f"  Skip PURE_LAND : {n_pure_land}")
    log.info(f"  Skip NO_DATA   : {n_no_data}")
    log.info(f"  MIXED (kept)   : {n_mixed}")
    log.info(f"  PURE_SEA (kept): {n_pure_sea}")
    log.info("=" * 62)

    # --- build outputs: kept_ordered (dict urut new_num) + mapping ---
    kept_ordered = {}
    mapping = {}
    for c in sorted(kept, key=lambda c: c["new_num"]):
        old_name = c["old_name"]
        kept_ordered[old_name] = grids_to_process[old_name]
        mapping[old_name] = c["new_name"]

    return kept_ordered, mapping, classified


def save_grid_mapping_csv(classified: list, out_csv: str) -> str:
    """Tulis CSV mapping old_name -> new_name (+ row/col/class/land_frac/dll)."""
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
                c["old_name"], c["new_name"] or "", c["new_num"] or "",
                c["row"], c["col"], c["tile_class"], c["keep"],
                f"{c['land_frac']:.4f}",
                c["n_elem"] if c["n_elem"] is not None else "",
                f"{minlon:.6f}", f"{minlat:.6f}", f"{maxlon:.6f}", f"{maxlat:.6f}",
            ])

    log.info(f"  Grid mapping CSV ditulis: {out_csv} ({len(ordered)} baris)")
    return out_csv


def save_grid_mapping_json(mapping: dict, classified: list, json_path: str) -> str:
    """
    [RENUMBER v3.1] Simpan mapping lengkap old_name -> detail klasifikasi +
    new_name ke file JSON (mengaktifkan GRID_MAPPING_FILE yang sebelumnya
    didefinisikan tapi tidak pernah ditulis).
    """
    import json

    registry = {}
    for c in classified:
        registry[c["old_name"]] = {
            "new_name": c["new_name"],
            "new_num": c["new_num"],
            "tile_class": c["tile_class"],
            "keep": c["keep"],
            "bbox": c["bbox"],
            "land_frac": c["land_frac"],
            "n_elem": c["n_elem"],
        }

    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    log.info(f"  Grid mapping JSON (registry) ditulis: {json_path} ({len(registry)} entri)")
    return json_path


def plot_grid_catalog(classified: list, land_shp: str, out_png: str,
                       dpi: int = CATALOG_DPI) -> str:
    """
    Render peta katalog grid: poligon daratan sebagai latar, kotak grid
    berwarna per kelas, nomor baru untuk grid yang kept.
    """
    if not HAS_MPL:
        log.warning("  matplotlib tidak tersedia — plot_grid_catalog dilewati")
        return ""

    if not classified:
        log.warning("  Tidak ada grid untuk di-plot — dilewati")
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
    ax.set_title("S-111 Grid Catalog — renumbered (S->N, W->E), land grids excluded")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_aspect("equal", adjustable="box")

    legend_handles = [
        Rectangle((0, 0), 1, 1, edgecolor="#1f6feb", facecolor="none", label="PURE_SEA (kept)"),
        Rectangle((0, 0), 1, 1, edgecolor="#e07b00", facecolor="none", label="MIXED (kept)"),
        Rectangle((0, 0), 1, 1, edgecolor="#999999", facecolor="none",
                  linestyle="--", label="Excluded (land/no-data)"),
    ]
    # Legend di LUAR area peta (di bawah) agar tidak menutupi grid bernomor.
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(0.0, -0.05), ncol=3, fontsize=8, frameon=True)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    log.info(f"  Grid catalog PNG disimpan: {out_png}")
    return out_png


# ==============================================================================
# POST-PROCESSOR — KHOA VIEWER COMPATIBILITY
# ==============================================================================

def _convert_timepoint(tp_str: str) -> str:
    """Convert time string to ISO 8601 format for KHOA Viewer compatibility.

    s100py writes timePoint as ISO 8601 (e.g. '20260909T000000Z').
    If update_metadata passed space format (e.g. '2026-09-09 00:00:00Z'),
    convert it to ISO 8601. If already ISO 8601, leave unchanged.
    """
    # Space format -> ISO 8601
    m = re.match(r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})Z$", tp_str)
    if m:
        y, mo, d, h, mi, s = m.groups()
        return f"{y}{mo}{d}T{h}{mi}{s}Z"
    # Already ISO 8601 — leave unchanged
    if re.match(r"\d{8}T\d{6}Z$", tp_str):
        return tp_str
    return tp_str


def _postprocess_hdf5_khoa(h5_path: str, time_record_interval: int = 3600):
    def _fix_group_f(hf):
        if "Group_F/SurfaceCurrent" not in hf: return
        rows = [
            (b"surfaceCurrentSpeed",     b"Surface Current Speed",
             b"knot", b"-9999.0", b"H5T_FLOAT", b"0.0", b"99.00",  b"geSemiInterval"),
            (b"surfaceCurrentDirection", b"Surface Current Direction",
             b"degree", b"-9999.0", b"H5T_FLOAT", b"0.0", b"359.9", b"closedInterval"),
        ]
        dt = np.dtype([
            ("code","S1024"), ("name","S1024"), ("uom.name","S1024"),
            ("fillValue","S1024"), ("datatype","S1024"),
            ("lower","S1024"), ("upper","S1024"), ("closure","S1024"),
        ])
        del hf["Group_F/SurfaceCurrent"]
        hf["Group_F/SurfaceCurrent"] = np.array(rows, dtype=dt)

    def _fix_feature_code(hf):
        if "Group_F/featureCode" not in hf: return
        del hf["Group_F/featureCode"]
        ds = hf["Group_F"].create_dataset(
            "featureCode", (1,), dtype=h5py.string_dtype(encoding="ascii", length=64))
        ds[0] = b"SurfaceCurrent"

    def _fix_axis_names(hf):
        if "SurfaceCurrent/axisNames" not in hf: return
        del hf["SurfaceCurrent/axisNames"]
        hf["SurfaceCurrent"].create_dataset(
            "axisNames",
            data=np.array([[b"longitude"], [b"latitude"]], dtype="S64"),
            dtype=h5py.string_dtype(encoding="ascii", length=64))

    def _fix_root_attrs(hf):
        hf.attrs["horizontalDatumReference"] = np.bytes_("EPSG")
        hf.attrs["epoch"] = np.bytes_("G1762")
        if "metadata" not in hf.attrs:
            hf.attrs["metadata"] = np.bytes_("MD_BASE.xml")
        for key in ("productSpecification", "geographicIdentifier",
                    "issueDate", "issueTime", "datasetDeliveryInterval"):
            if key in hf.attrs:
                val = hf.attrs[key]
                # Handle bytes/np.bytes_ safely
                if isinstance(val, (bytes, np.bytes_)):
                    try:
                        val = val.decode("utf-8")
                    except (UnicodeDecodeError, AttributeError):
                        continue
                elif not isinstance(val, str):
                    continue
                # Replace unicode arrow with ASCII safe alternative
                val = val.replace('→', '->')
                hf.attrs[key] = np.bytes_(val)

    def _fix_sc_attrs(hf):
        sc = hf.get("SurfaceCurrent")
        if sc is None: return
        sc.attrs["sequencingRule.scanDirection"] = np.bytes_("Longitude,Latitude")
        if "methodCurrentsProduct" in sc.attrs:
            val = sc.attrs["methodCurrentsProduct"]
            # Handle bytes/np.bytes_ safely
            if isinstance(val, (bytes, np.bytes_)):
                try:
                    val = val.decode("utf-8")
                except (UnicodeDecodeError, AttributeError):
                    return
            elif not isinstance(val, str):
                return
            val = val.replace('→', '->')
            sc.attrs["methodCurrentsProduct"] = np.bytes_(val)

    def _fix_instance_attrs(inst, root):
        for key in ("dateTimeOfFirstRecord", "dateTimeOfLastRecord"):
            if key in inst.attrs:
                val = inst.attrs[key]
                if isinstance(val, (bytes, np.bytes_)): val = val.decode("utf-8")
                inst.attrs[key] = np.bytes_(_convert_timepoint(val))
        if "startSequence" in inst.attrs:
            val = inst.attrs["startSequence"]
            if isinstance(val, str): inst.attrs["startSequence"] = np.bytes_(val)
        if "timeRecordInterval" in inst.attrs:
            cv = int(inst.attrs["timeRecordInterval"])
            inst.attrs["timeRecordInterval"] = np.int32(cv if cv >= 1 else time_record_interval)
        else:
            inst.attrs["timeRecordInterval"] = np.int32(time_record_interval)
        for bb in ("westBoundLongitude", "eastBoundLongitude",
                   "southBoundLatitude", "northBoundLatitude"):
            if bb not in inst.attrs and bb in root.attrs:
                inst.attrs[bb] = root.attrs[bb]
        for src, dst in (("eastBoundLongitude", "eastboundLongitude"),
                         ("westBoundLongitude", "westboundLongitude")):
            if dst not in inst.attrs and src in root.attrs:
                inst.attrs[dst] = root.attrs[src]

    def _fix_group_timepoints(inst):
        for grp_name in sorted(inst.keys()):
            if not grp_name.startswith("Group_"): continue
            grp = inst[grp_name]
            if "timePoint" not in grp.attrs: continue
            tp = grp.attrs["timePoint"]
            if isinstance(tp, (bytes, np.bytes_)): tp = tp.decode("utf-8")
            grp.attrs["timePoint"] = np.bytes_(_convert_timepoint(tp))

    def _fix_container_attrs(hf):
        """Ensure numInstances and numGRP are set for KHOA Viewer."""
        sc = hf.get("SurfaceCurrent")
        if sc is None: return
        instances = [k for k in sc.keys() if k.startswith("SurfaceCurrent.")]
        n_inst = len(instances)
        sc.attrs["numInstances"] = np.int32(n_inst)
        for inst_name in instances:
            inst = sc[inst_name]
            groups = [k for k in inst.keys() if k.startswith("Group_")]
            inst.attrs["numGRP"] = np.int32(len(groups))
            if "numberOfGroups" not in inst.attrs:
                inst.attrs["numberOfGroups"] = np.int32(len(groups))
            if "numberOfTimes" not in inst.attrs:
                inst.attrs["numberOfTimes"] = np.int32(len(groups))

    try:
        with h5py.File(h5_path, "r+") as hf:
            _fix_group_f(hf); _fix_feature_code(hf); _fix_axis_names(hf)
            _fix_root_attrs(hf); _fix_sc_attrs(hf)
            sc = hf.get("SurfaceCurrent")
            if sc is None: return
            for inst_name in sorted(sc.keys()):
                if not inst_name.startswith("SurfaceCurrent."): continue
                _fix_instance_attrs(sc[inst_name], hf)
                _fix_group_timepoints(sc[inst_name])
            _fix_container_attrs(hf)
    except Exception as e:
        log.error(f"  [post] GAGAL: {os.path.basename(h5_path)}: {e}", exc_info=True)
        raise


def _postprocess_hdf5_khoa_multi(h5_path: str, time_record_interval: int = 3600,
                                  n_instances: int = None):
    """
    [F35] Post-processor KHOA Viewer untuk HDF5 multi-instance.
    Identik dengan _postprocess_hdf5_khoa, namun:
      - Memvalidasi semua instance SurfaceCurrent.NN tersedia.
      - Memastikan numInstances pada container = jumlah instance sebenarnya.
      - Mendukung axisNames per-instance bila ada.
    """
    def _fix_group_f(hf):
        if "Group_F/SurfaceCurrent" not in hf: return
        rows = [
            (b"surfaceCurrentSpeed",     b"Surface Current Speed",
             b"knot", b"-9999.0", b"H5T_FLOAT", b"0.0", b"99.00",  b"geSemiInterval"),
            (b"surfaceCurrentDirection", b"Surface Current Direction",
             b"degree", b"-9999.0", b"H5T_FLOAT", b"0.0", b"359.9", b"closedInterval"),
        ]
        dt = np.dtype([
            ("code","S1024"), ("name","S1024"), ("uom.name","S1024"),
            ("fillValue","S1024"), ("datatype","S1024"),
            ("lower","S1024"), ("upper","S1024"), ("closure","S1024"),
        ])
        del hf["Group_F/SurfaceCurrent"]
        hf["Group_F/SurfaceCurrent"] = np.array(rows, dtype=dt)

    def _fix_feature_code(hf):
        if "Group_F/featureCode" not in hf: return
        del hf["Group_F/featureCode"]
        ds = hf["Group_F"].create_dataset(
            "featureCode", (1,), dtype=h5py.string_dtype(encoding="ascii", length=64))
        ds[0] = b"SurfaceCurrent"

    def _fix_axis_names(hf):
        if "SurfaceCurrent/axisNames" not in hf: return
        del hf["SurfaceCurrent/axisNames"]
        hf["SurfaceCurrent"].create_dataset(
            "axisNames",
            data=np.array([[b"longitude"], [b"latitude"]], dtype="S64"),
            dtype=h5py.string_dtype(encoding="ascii", length=64))

    def _fix_root_attrs(hf):
        hf.attrs["horizontalDatumReference"] = np.bytes_("EPSG")
        hf.attrs["epoch"] = np.bytes_("G1762")
        if "metadata" not in hf.attrs:
            hf.attrs["metadata"] = np.bytes_("MD_BASE.xml")
        for key in ("productSpecification", "geographicIdentifier",
                    "issueDate", "issueTime", "datasetDeliveryInterval"):
            if key in hf.attrs:
                val = hf.attrs[key]
                # Handle bytes/np.bytes_ safely
                if isinstance(val, (bytes, np.bytes_)):
                    try:
                        val = val.decode("utf-8")
                    except (UnicodeDecodeError, AttributeError):
                        continue
                elif not isinstance(val, str):
                    continue
                # Replace unicode arrow with ASCII safe alternative
                val = val.replace('→', '->')
                hf.attrs[key] = np.bytes_(val)

    def _fix_sc_attrs(hf):
        sc = hf.get("SurfaceCurrent")
        if sc is None: return
        sc.attrs["sequencingRule.scanDirection"] = np.bytes_("Longitude,Latitude")
        if "methodCurrentsProduct" in sc.attrs:
            val = sc.attrs["methodCurrentsProduct"]
            # Handle bytes/np.bytes_ safely
            if isinstance(val, (bytes, np.bytes_)):
                try:
                    val = val.decode("utf-8")
                except (UnicodeDecodeError, AttributeError):
                    return
            elif not isinstance(val, str):
                return
            val = val.replace('→', '->')
            sc.attrs["methodCurrentsProduct"] = np.bytes_(val)

    def _fix_instance_attrs(inst, root):
        for key in ("dateTimeOfFirstRecord", "dateTimeOfLastRecord"):
            if key in inst.attrs:
                val = inst.attrs[key]
                if isinstance(val, (bytes, np.bytes_)): val = val.decode("utf-8")
                inst.attrs[key] = np.bytes_(_convert_timepoint(val))
        if "startSequence" in inst.attrs:
            val = inst.attrs["startSequence"]
            if isinstance(val, str): inst.attrs["startSequence"] = np.bytes_(val)
        if "timeRecordInterval" in inst.attrs:
            cv = int(inst.attrs["timeRecordInterval"])
            inst.attrs["timeRecordInterval"] = np.int32(cv if cv >= 1 else time_record_interval)
        else:
            inst.attrs["timeRecordInterval"] = np.int32(time_record_interval)
        for bb in ("westBoundLongitude", "eastBoundLongitude",
                   "southBoundLatitude", "northBoundLatitude"):
            if bb not in inst.attrs and bb in root.attrs:
                inst.attrs[bb] = root.attrs[bb]
        for src, dst in (("eastBoundLongitude", "eastboundLongitude"),
                         ("westBoundLongitude", "westboundLongitude")):
            if dst not in inst.attrs and src in inst.attrs:
                inst.attrs[dst] = inst.attrs[src]
            elif dst not in inst.attrs and src in root.attrs:
                inst.attrs[dst] = root.attrs[src]

    def _fix_group_timepoints(inst):
        for grp_name in sorted(inst.keys()):
            if not grp_name.startswith("Group_"): continue
            grp = inst[grp_name]
            if "timePoint" not in grp.attrs: continue
            tp = grp.attrs["timePoint"]
            if isinstance(tp, (bytes, np.bytes_)): tp = tp.decode("utf-8")
            grp.attrs["timePoint"] = np.bytes_(_convert_timepoint(tp))

    def _fix_container_attrs(hf):
        sc = hf.get("SurfaceCurrent")
        if sc is None: return
        instances = [k for k in sc.keys() if k.startswith("SurfaceCurrent.")]
        n_inst = len(instances)
        sc.attrs["numInstances"] = np.int32(n_inst)
        for inst_name in instances:
            inst = sc[inst_name]
            groups = [k for k in inst.keys() if k.startswith("Group_")]
            inst.attrs["numGRP"] = np.int32(len(groups))
            if "numberOfGroups" not in inst.attrs:
                inst.attrs["numberOfGroups"] = np.int32(len(groups))
            if "numberOfTimes" not in inst.attrs:
                inst.attrs["numberOfTimes"] = np.int32(len(groups))

    try:
        with h5py.File(h5_path, "r+") as hf:
            _fix_group_f(hf); _fix_feature_code(hf); _fix_axis_names(hf)
            _fix_root_attrs(hf); _fix_sc_attrs(hf)
            sc = hf.get("SurfaceCurrent")
            if sc is None: return
            instances = sorted([k for k in sc.keys()
                                if k.startswith("SurfaceCurrent.")])
            actual_n = len(instances)
            if n_instances is not None and actual_n != n_instances:
                log.warning(f"  [post-multi] Jumlah instance {actual_n} ≠ "
                            f"ekspektasi {n_instances}")
            for inst_name in instances:
                _fix_instance_attrs(sc[inst_name], hf)
                _fix_group_timepoints(sc[inst_name])
            _fix_container_attrs(hf)
    except Exception as e:
        log.error(f"  [post-multi] GAGAL: {os.path.basename(h5_path)}: {e}",
                  exc_info=True)
        raise

# ==============================================================================
# SAFE CLOSE + CREATE S111
# ==============================================================================

def safe_close_and_create_s111(output_h5: str, dcf: int = 2):
    gc.collect()
    # NOTE: s100py v2.0.1 uses positional list indexing, not class-level
    # counters. Group numbering (Group_001, Group_002, ...) is determined
    # by enumerate(self) in S1xxCollection.write(), so each new S111File
    # starts fresh. No counter reset needed.

    try:
        open_ids = h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE)
        for fid in open_ids:
            try:
                fname = h5py.h5f.get_name(fid).decode("utf-8")
                if (os.path.normcase(os.path.abspath(fname)) ==
                        os.path.normcase(os.path.abspath(output_h5))):
                    h5py.h5f.close(fid)
            except Exception:
                pass
    except Exception as e:
        log.warning(f"  Tidak dapat iterasi open HDF5 IDs: {e}")

    if os.path.exists(output_h5):
        try:
            os.remove(output_h5)
        except PermissionError as e:
            log.error(f"  File masih dikunci: {e}"); raise

    data_file = s111.utils.create_s111(output_h5, dcf)
    log.info(f"  S111_70_85_smart_grid_Output_File: {output_h5}")
    return data_file

def safe_close_and_delete(output_h5: str):
    """Close HDF5 file handle and delete file if no valid data written."""
    gc.collect()
    try:
        open_ids = h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE)
        for fid in open_ids:
            try:
                fname = h5py.h5f.get_name(fid).decode("utf-8")
                if (os.path.normcase(os.path.abspath(fname)) ==
                        os.path.normcase(os.path.abspath(output_h5))):
                    h5py.h5f.close(fid)
            except Exception:
                pass
    except Exception as e:
        log.warning(f"  Tidak dapat iterasi open HDF5 IDs: {e}")

    if os.path.exists(output_h5):
        try:
            os.remove(output_h5)
            log.info(f"  File invalid dihapus: {output_h5}")
        except PermissionError as e:
            log.error(f"  File masih dikunci: {e}")

# ==============================================================================
# DFSU OPEN & DETECT
# ==============================================================================

# [MEMFIX] Sentinel: diset True oleh process_one_timestep() saat ds_open.read()
# gagal (mis. OOM saat itu mem-poison state internal reader mikeio, memicu
# kaskade DFS error code 2019 di timestep berikutnya). Loop pemanggil per-hari
# memeriksa sentinel ini dan memanggil reopen_dfsu() untuk memulihkan handle.
_DFSU_REOPEN_REQUESTED = False

def open_and_detect_dfsu(dfsu_path: str):
    try:
        ds_open = mikeio.open(dfsu_path)
        log.info(f"DFSU    : {dfsu_path}")
        log.info(f"Items   : {[i.name for i in ds_open.items]}")
        log.info(f"Waktu   : {ds_open.time[0]} -> {ds_open.time[-1]} ({len(ds_open.time)} ts)")
    except Exception as e:
        log.error(f"Gagal buka DFSU: {e}"); raise

    item_names = [i.name for i in ds_open.items]
    for item in [U_ITEM, V_ITEM, WD_ITEM]:
        if item not in item_names:
            raise ValueError(f"Item '{item}' tidak ada. Tersedia: {item_names}")

    geom     = ds_open.geometry
    elem_x   = geom.element_coordinates[:, 0]
    elem_y   = geom.element_coordinates[:, 1]
    n_layers = getattr(geom, "n_layers", None)

    if n_layers is None or n_layers <= 1:
        log.info(f"Tipe: 2D | {len(elem_x):,} elemen")
        return ds_open, "2D", 1, None, None, elem_x, elem_y, None, None

    try:
        layer_ids  = geom.layer_ids
        bbox_z_col = geom.element_coordinates[:, 2]
    except AttributeError as e:
        log.warning(f"layer_ids tidak tersedia ({e}) -> 2D")
        return ds_open, "2D", 1, None, None, elem_x, elem_y, None, None

    unique_lids = np.unique(layer_ids)
    z_unique    = np.array([bbox_z_col[layer_ids == lid].mean() for lid in unique_lids])
    log.info(f"Tipe: 3D | {n_layers} layer | {np.abs(z_unique).round(2).tolist()} m")
    return (ds_open, "3D", n_layers, z_unique, unique_lids,
            elem_x, elem_y, layer_ids, bbox_z_col)

def reopen_dfsu(old_ds_open=None, dfsu_path: str = DFSU_PATH):
    """
    [MEMFIX] Buka ulang handle DFSU setelah ds_open.read() gagal (biasanya
    dipicu tekanan memori) dan mem-poison state internal reader mikeio,
    menyebabkan kaskade DFS error code 2019 pada timestep berikutnya.

    Hanya handle dataset yang di-refresh — geometri (elem_x/elem_y/layer_ids)
    tidak berubah sehingga tidak perlu dihitung ulang oleh caller.
    """
    if old_ds_open is not None:
        try:
            old_ds_open.close()
        except Exception as e:
            log.warning(f"  Gagal menutup handle DFSU lama sebelum reopen: {e}")
    try:
        new_ds_open = mikeio.open(dfsu_path)
        log.warning(f"  DFSU handle di-reopen setelah kegagalan read: {dfsu_path}")
        return new_ds_open
    except Exception as e:
        log.error(f"  Gagal reopen DFSU: {e}"); raise

# ==============================================================================
# BBOX MASK
# ==============================================================================

def build_bbox_mask(elem_x, elem_y, layer_ids, unique_lids, bbox, dfsu_type):
    minx, miny, maxx, maxy = bbox
    geo_mask = ((elem_x >= minx) & (elem_x <= maxx) &
                (elem_y >= miny) & (elem_y <= maxy))

    if dfsu_type == "2D" or layer_ids is None:
        if geo_mask.sum() == 0:
            raise ValueError(f"Tidak ada elemen dalam BBOX {bbox}.")
        log.info(f"BBOX mask 2D: {geo_mask.sum():,}/{len(elem_x):,} elemen")
        return geo_mask, None, elem_x[geo_mask], elem_y[geo_mask]

    top_layer_id   = unique_lids[-1]
    top_layer_mask = (layer_ids == top_layer_id)
    combined_surf  = geo_mask & top_layer_mask
    if combined_surf.sum() == 0:
        raise ValueError(f"Tidak ada elemen surface dalam BBOX {bbox}.")
    surface_in_bbox_pos = np.where(combined_surf[top_layer_mask])[0]
    log.info(f"BBOX mask 3D: {combined_surf.sum():,} elemen surface")
    return (surface_in_bbox_pos, geo_mask,
            elem_x[combined_surf], elem_y[combined_surf])

# ==============================================================================
# U/V -> SPEED & DIRECTION (OCEANOGRAPHIC CONVENTION)
# ==============================================================================

def uv_to_speed_direction(u_grid: np.ndarray,
                           v_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = ((u_grid > NODATA + 1.0) & (v_grid > NODATA + 1.0) &
             np.isfinite(u_grid) & np.isfinite(v_grid))
    speed     = np.full(u_grid.shape, NODATA, dtype=np.float32)
    direction = np.full(u_grid.shape, NODATA, dtype=np.float32)
    u_v = u_grid[valid].astype(np.float64)
    v_v = v_grid[valid].astype(np.float64)
    speed[valid]     = np.sqrt(u_v**2 + v_v**2).astype(np.float32)
    direction[valid] = ((np.degrees(np.arctan2(-v_v, -u_v)) + 360.0)
                        % 360.0).astype(np.float32)
    vs = speed[valid]
    if vs.size > 0:
        log.info(f"  Speed [{vs.min():.4f}, {vs.max():.4f}] m/s | {valid.sum():,} valid")
    return speed, direction

# ==============================================================================
# GRID & PROPERTIES
# ==============================================================================

def build_grid_and_properties(bbox, dx, dy):
    lon_arr = np.arange(bbox[0], bbox[2], dx)
    lat_arr = np.arange(bbox[1], bbox[3], dy)
    if len(lon_arr) == 0 or len(lat_arr) == 0:
        raise ValueError(f"Grid kosong! BBOX={bbox}, DX={dx}, DY={dy}")
    grid_xx, grid_yy = np.meshgrid(lon_arr, lat_arr)
    nx, ny = len(lon_arr), len(lat_arr)
    props = {
        "minx":       float(lon_arr[0]),  "maxx": float(lon_arr[-1]),
        "miny":       float(lat_arr[0]),  "maxy": float(lat_arr[-1]),
        "cellsize_x": float(dx),          "cellsize_y": float(dy),
        "nx":         int(nx),            "ny":  int(ny),
    }
    log.info(f"Grid S-111: {nx}×{ny} = {nx*ny:,} sel | "
             f"{dx*111000:.1f}m × {dy*110500:.1f}m")
    return grid_xx, grid_yy, props

# ==============================================================================
# METADATA
# ==============================================================================

def build_s111_metadata(datetime_issuance, geographic_id="Teluk Jakarta, Indonesia"):
    return {
        "horizontalCRS":                 4326,
        "horizontalDatumReference":      "EPSG",
        "epoch":                         "G1762",
        "geographicIdentifier":          geographic_id,
        "speedUncertainty":             -1.0,
        "directionUncertainty":         -1.0,
        "verticalUncertainty":          -1.0,
        "horizontalPositionUncertainty":-1.0,
        "surfaceCurrentDepth":           0.0,
        "depthTypeIndex":                1,
        "commonPointRule":               2,
        "interpolationType":             1,
        "dataDynamicity":                5,
        "methodCurrentsProduct":         "MIKE_FM_Hydrodynamic_Model_Forecast",
        "verticalCS":                    6498,
        "verticalCoordinateBase":        2,
        "verticalDatumReference":        1,
        "verticalDatum":                 3,
        "datasetDeliveryInterval":       "PT1H",
        "issueDate":                     datetime_issuance,
        "issueTime":                     datetime_issuance,
    }

# ==============================================================================
# CORE: PROSES SATU TIMESTEP — [F25] pakai regrid_uv_to_s111
# ==============================================================================

def process_one_timestep(t, ds_open, dfsu_type, bbox_mask_2d,
                         elem_x_surf, elem_y_surf, grid_xx, grid_yy):
    # [MEMFIX] pre-declare agar finally-cleanup di bawah tidak perlu locals()/hasattr
    # guard dan tetap aman walau exit lewat return paling awal.
    ds_slice = ds_bbox = raw_u = raw_v = raw_wd = dry_mask = u_grid = v_grid = idx = None
    try:
        try:
            ds_slice = ds_open.read(items=[U_ITEM, V_ITEM, WD_ITEM], time=[t])
        except Exception as e:
            global _DFSU_REOPEN_REQUESTED
            _DFSU_REOPEN_REQUESTED = True
            log.error(f"  Gagal read {t}: {e} — reopen DFSU akan diminta"); return None, None, None

        try:
            idx     = (np.where(bbox_mask_2d)[0].tolist()
                       if dfsu_type == "2D" else bbox_mask_2d.tolist())
            ds_bbox = ds_slice.isel(element=idx)
        except IndexError as e:
            log.error(f"  IndexError isel {t}: {e}"); return None, None, None

        raw_u  = ds_bbox[U_ITEM].to_numpy().flatten().astype(np.float64)
        raw_v  = ds_bbox[V_ITEM].to_numpy().flatten().astype(np.float64)
        raw_wd = ds_bbox[WD_ITEM].to_numpy().flatten().astype(np.float64)

        dry_mask = raw_wd < DRY_CELL_THRESHOLD
        raw_u    = np.where(dry_mask, np.nan, raw_u)
        raw_v    = np.where(dry_mask, np.nan, raw_v)

        n_wet = int((~dry_mask).sum())
        log.info(f"  wet={n_wet:,}/{len(raw_u):,} | dry={int(dry_mask.sum()):,}")
        if n_wet < 3:
            log.warning(f"  {t}: {n_wet} wet < 3 — skip"); return None, None, None

        u_grid, v_grid = regrid_uv_to_s111(
            elem_x_surf, elem_y_surf, raw_u, raw_v, grid_xx, grid_yy
        )

        # [BUG-U6] Mask boundary extrapolation yang menghasilkan data palsu
        # di area darat di luar jangkauan model
        if MASK_EXTRAP_BOUNDARY:
            u_grid, v_grid = mask_extrapolation_boundary(
                u_grid, v_grid, grid_xx, grid_yy, elem_x_surf, elem_y_surf
            )

        speed, direction = uv_to_speed_direction(u_grid, v_grid)

        if int((speed > NODATA + 1.0).sum()) == 0:
            log.warning(f"  {t}: 0 sel valid setelah regrid — skip")
            return None, None, None

        if hasattr(t, "to_pydatetime"):
            t_dt = t.to_pydatetime()
        elif isinstance(t, np.datetime64):
            t_dt = t.astype("M8[ms]").astype(datetime.datetime)
        else:
            t_dt = t

        return speed, direction, t_dt
    finally:
        # [MEMFIX] lepas buffer per-elemen (~36k elemen: U/V float64 + WD int32)
        # segera setelah dipakai di setiap exit path, jangan tunggu function return
        # (mengurangi steady-state memory ceiling lintas ribuan timestep).
        ds_slice = ds_bbox = raw_u = raw_v = raw_wd = dry_mask = u_grid = v_grid = idx = None

# ==============================================================================
# DAY BLOCK GROUPING — daily (1 hari/file) atau 3day (3 hari/file)
# ==============================================================================

def build_day_blocks(sorted_days: list, mode: str) -> list:
    """
    Kelompokkan daftar hari menjadi blok sesuai OUTPUT_MODE.

    - "daily" -> 1 hari per blok  : [[d1], [d2], ...]
    - "3day"  -> BLOCK_DAYS per blok: [[d1,d2,d3], [d4,d5,d6], ...]

    Blok terakhir boleh berisi < BLOCK_DAYS hari (sisa). Setiap blok
    menghasilkan SATU file HDF5 (semua timestep hari-hari dalam blok
    ditulis sebagai Group_001..Group_NNN berurutan).
    """
    step = BLOCK_DAYS if mode == "3day" else 1
    return [sorted_days[i:i + step] for i in range(0, len(sorted_days), step)]


# ==============================================================================
# MODE DAILY / 3DAY (satu file per blok hari)
# ==============================================================================

def process_daily(date_key, day_timesteps, ds_open, dfsu_type,
                  bbox_mask_2d, elem_x_surf, elem_y_surf,
                  grid_xx, grid_yy, grid_properties,
                  metadata, output_dir, prefix=FILE_PREFIX,
                  land_mask=None, date_key_end=None):
    # date_key = hari pertama blok; date_key_end = hari terakhir blok (opsional).
    # Untuk blok multi-hari (3day), nama file mencakup rentang tanggal.
    if date_key_end is not None and date_key_end != date_key:
        fname = f"{prefix}_{date_key.strftime('%Y%m%d')}_{date_key_end.strftime('%Y%m%d')}.h5"
    else:
        fname = f"{prefix}_{date_key.strftime('%Y%m%d')}.h5"
    output_h5 = os.path.join(output_dir, fname)
    n_times   = len(day_timesteps)
    log.info(f"  [DAILY] {date_key} | {n_times} ts -> {fname}")

    data_file = safe_close_and_create_s111(output_h5, dcf=2)
    s111.utils.add_metadata(metadata, data_file)
    s111.utils.add_surface_current_instance(data_file)

    t_first_str = t_last_str = None
    n_written   = 0

    for i, t in enumerate(day_timesteps):
        log.info(f"    [{i+1:03d}/{n_times}] {t}")
        speed, direction, t_dt = process_one_timestep(
            t, ds_open, dfsu_type, bbox_mask_2d,
            elem_x_surf, elem_y_surf, grid_xx, grid_yy
        )
        if speed is None:
            # [MEMFIX] Pulihkan handle DFSU yang mungkin "poisoned" akibat
            # kegagalan read (biasanya dipicu tekanan memori), agar timestep
            # berikutnya tidak ikut gagal dengan DFS error code 2019.
            global _DFSU_REOPEN_REQUESTED
            if _DFSU_REOPEN_REQUESTED:
                ds_open = reopen_dfsu(ds_open, DFSU_PATH)
                _DFSU_REOPEN_REQUESTED = False
            continue
        try:
            s111.utils.add_data_from_arrays(
                speed, direction, data_file, grid_properties, t_dt, 2
            )
            n_written += 1
            t_meta = t_dt.strftime("%Y%m%dT%H%M%SZ")
            if t_first_str is None: t_first_str = t_meta
            t_last_str = t_meta
        except Exception as e:
            log.error(f"    Gagal write {t}: {e}")

    # Hanya lakukan post-process jika ada data yang ditulis
    if n_written == 0:
        log.warning(f"  [SKIP] Tidak ada data valid untuk {date_key} — tidak membuat file")
        safe_close_and_delete(output_h5)
        return None, n_written

    s111.utils.update_metadata(data_file, grid_properties, {
        "dateTimeOfFirstRecord": t_first_str or "",
        "dateTimeOfLastRecord":  t_last_str  or "",
        "numberOfGroups": n_written, "numberOfTimes": n_written,
        "timeRecordInterval": 3600, "num_instances": 1, "dataDynamicity": 5,
    })
    try:
        s111.utils.write_data_file(data_file)
    except Exception as e:
        log.error(f"  [WRITE] Gagal menulis {fname}: {e} — file dihapus")
        safe_close_and_delete(output_h5)
        return None, n_written

    # [MEMFIX] Tutup handle secara eksplisit SEBELUM land-mask/post-process
    # membuka file yang sama (mode r+). Jangan andalkan GC __dealloc__ untuk
    # flush — di bawah tekanan memori itu bisa gagal (Errno 22) dan
    # merusak/memotong file output.
    try:
        data_file.flush()
        data_file.close()
    except Exception as e:
        log.error(f"  [WRITE] Gagal flush/close {fname}: {e}")
    del data_file
    gc.collect()

    if land_mask is not None and land_mask.sum() > 0:
        try:
            apply_land_mask_to_h5(output_h5, land_mask, grid_properties)
            if not validate_land_masked_h5(output_h5, land_mask):
                log.error(f"  [LANDMASK] VALIDASI GAGAL: masih ada sel darat "
                          f"berisi arus di {fname}")
        except Exception as e:
            log.error(f"  [LANDMASK] GAGAL: {e}")

    try:
        _postprocess_hdf5_khoa(output_h5)
    except Exception as e:
        log.error(f"  Post-process GAGAL: {e}")

    log.info(f"  OK {fname} ({n_written}/{n_times})")
    return output_h5, n_written

# ==============================================================================
# MODE HOURLY
# ==============================================================================

def process_hourly(t, ds_open, dfsu_type, bbox_mask_2d,
                   elem_x_surf, elem_y_surf,
                   grid_xx, grid_yy, grid_properties,
                   metadata, output_dir, prefix=FILE_PREFIX,
                   land_mask=None):
    speed, direction, t_dt = process_one_timestep(
        t, ds_open, dfsu_type, bbox_mask_2d,
        elem_x_surf, elem_y_surf, grid_xx, grid_yy
    )
    if speed is None: return None

    t_fname   = t_dt.strftime("%Y%m%dT%H%MZ")
    t_meta    = t_dt.strftime("%Y%m%dT%H%M%SZ")
    fname     = f"{prefix}_{t_fname}.h5"
    output_h5 = os.path.join(output_dir, fname)
    log.info(f"  [HOURLY] -> {fname}")

    data_file = safe_close_and_create_s111(output_h5, dcf=2)
    s111.utils.add_metadata(metadata, data_file)
    s111.utils.add_surface_current_instance(data_file)

    try:
        s111.utils.add_data_from_arrays(
            speed, direction, data_file, grid_properties, t_dt, 2
        )
    except Exception as e:
        log.error(f"  Gagal write {t}: {e}"); return None

    s111.utils.update_metadata(data_file, grid_properties, {
        "dateTimeOfFirstRecord": t_meta, "dateTimeOfLastRecord": t_meta,
        "numberOfGroups": 1, "numberOfTimes": 1, "timeRecordInterval": 3600,
        "num_instances": 1, "dataDynamicity": 5,
    })
    try:
        s111.utils.write_data_file(data_file)
    except Exception as e:
        log.error(f"  [WRITE] Gagal menulis {fname}: {e} — file dihapus")
        safe_close_and_delete(output_h5)
        return None

    # [MEMFIX] Tutup handle secara eksplisit SEBELUM land-mask/post-process
    # membuka file yang sama (mode r+). Jangan andalkan GC __dealloc__ untuk
    # flush — di bawah tekanan memori itu bisa gagal (Errno 22) dan
    # merusak/memotong file output.
    try:
        data_file.flush()
        data_file.close()
    except Exception as e:
        log.error(f"  [WRITE] Gagal flush/close {fname}: {e}")
    del data_file
    gc.collect()

    if land_mask is not None and land_mask.sum() > 0:
        try:
            apply_land_mask_to_h5(output_h5, land_mask, grid_properties)
            if not validate_land_masked_h5(output_h5, land_mask):
                log.error(f"  [LANDMASK] VALIDASI GAGAL: masih ada sel darat "
                          f"berisi arus di {fname}")
        except Exception as e:
            log.error(f"  [LANDMASK] GAGAL: {e}")

    try:
        _postprocess_hdf5_khoa(output_h5)
    except Exception as e:
        log.error(f"  Post-process GAGAL: {e}")

    log.info(f"  OK {fname}")
    return output_h5

# ==============================================================================
# [F33][F34] PREPARE GRID CONTEXTS — siapkan semua grid sekaligus
# ==============================================================================

def prepare_grid_context(grid_name, grid_cfg, ds_open, dfsu_type,
                          elem_x, elem_y, layer_ids, unique_lids,
                          land_union=None):
    """
    [F33] Siapkan konteks lengkap untuk satu grid TANPA memproses timestep.
    Mengembalikan dict dengan: bbox, bbox_mask_2d, elem_x_surf, elem_y_surf,
    grid_xx, grid_yy, grid_properties, land_mask, status.

    Dipakai oleh mode multi-instance (semua grid dievaluasi dulu, lalu
    instance dibuat satu per satu di HDF5 yang sama).
    """
    bbox = grid_cfg["bbox"]
    ctx = {"grid_name": grid_name, "bbox": bbox, "status": "ok"}

    # ── 1. BBOX MASK ────────────────────────────────────────────────────────
    try:
        (bbox_mask_2d, _bbox_full,
         elem_x_surf, elem_y_surf) = build_bbox_mask(
            elem_x, elem_y, layer_ids, unique_lids, bbox, dfsu_type)
    except ValueError as e:
        ctx["status"] = f"bbox_skip: {e}"
        return ctx
    if len(elem_x_surf) == 0:
        ctx["status"] = "bbox_empty"
        return ctx

    ctx["bbox_mask_2d"]  = bbox_mask_2d
    ctx["elem_x_surf"]   = elem_x_surf
    ctx["elem_y_surf"]   = elem_y_surf

    # ── 2. DFSU RESOLUTION ──────────────────────────────────────────────────
    dfsu_res_m = detect_dfsu_resolution(elem_x_surf, elem_y_surf)
    ctx["dfsu_res_m"] = dfsu_res_m

    # ── 3. ADAPTIVE REGRID ──────────────────────────────────────────────────
    n_groups_est = (72 if OUTPUT_MODE == "3day"
                    else 1 if OUTPUT_MODE == "hourly" else 24)
    dx_new, dy_new, _nx_est, _ny_est, _est_size, _dx_m_final = adaptive_grid_resolution(
        bbox, GRID_FILE_DX, GRID_FILE_DY, n_groups_est, MAX_FILE_SIZE_MB, dfsu_res_m
    )

    # ── 4. BUILD GRID ───────────────────────────────────────────────────────
    try:
        grid_xx, grid_yy, grid_properties = build_grid_and_properties(bbox, dx_new, dy_new)
    except ValueError as e:
        ctx["status"] = f"build_grid_skip: {e}"
        return ctx

    ctx["grid_xx"]         = grid_xx
    ctx["grid_yy"]         = grid_yy
    ctx["grid_properties"] = grid_properties

    # ── 5. LAND MASK ────────────────────────────────────────────────────────
    land_mask = None
    if APPLY_LAND_MASK:
        try:
            land_mask = build_land_mask(LAND_SHP, grid_properties, land_union=land_union)
        except Exception as e:
            log.error(f"  [{grid_name}] [LANDMASK] GAGAL: {e}")
    ctx["land_mask"] = land_mask

    return ctx


# ==============================================================================
# [F33][F34] MULTI-INSTANCE DCF2 — semua grid dalam satu HDF5 per hari
# ==============================================================================

def process_daily_multi_instance(date_key, day_timesteps, ds_open, dfsu_type,
                                  grid_contexts, metadata, output_dir,
                                  prefix=FILE_PREFIX):
    """
    [F33] Tulis SEMUA grid sebagai instance SurfaceCurrent.01 .. SurfaceCurrent.N
    dalam satu HDF5 (mode DCF2 multi-instance sesuai S-111 spec).

    Setiap grid -> add_surface_current_instance (instance baru) lalu untuk setiap
    timestep di hari tersebut -> add_data_from_arrays (group baru pada instance
    terakhir).

    Args:
        date_key:        datetime.date untuk filename & logging
        day_timesteps:   list timestep pada hari tersebut
        ds_open:         mikeio.Dataset
        dfsu_type:       "2D" / "3D"
        grid_contexts:   list of dict dari prepare_grid_context()
        metadata:        dict metadata S-111 (root)
        output_dir:      direktori output utama
        prefix:          prefix filename

    Returns:
        (output_h5, n_grid_written, n_groups_written)
    """
    fname     = f"{prefix}{date_key.strftime('%Y%m%d')}.h5"
    output_h5 = os.path.join(output_dir, fname)
    n_times   = len(day_timesteps)
    valid_grids = [c for c in grid_contexts if c.get("status") == "ok"]
    n_grids   = len(valid_grids)
    log.info(f"  [MULTI] {date_key} | {n_times} ts × {n_grids} grid -> {fname}")

    if n_grids == 0:
        log.error(f"  [MULTI] Tidak ada grid valid untuk {date_key} — skip")
        return None, 0, 0

    data_file = safe_close_and_create_s111(output_h5, dcf=2)
    s111.utils.add_metadata(metadata, data_file)

    # ── ITERATE GRID (instance) ──────────────────────────────────────────────
    # [F34] Setiap grid -> 1 instance, lalu N_timestep Group ditambahkan ke instance tsb.
    # S1xxCollection.write() menomori Group_XXX fresh per instance via enumerate(self).
    written_summary = []  # [(inst_idx, grid_name, n_groups, t_first, t_last, land_mask, grid_props)]
    speed_global_min =  np.inf
    speed_global_max = -np.inf

    for inst_idx, ctx in enumerate(valid_grids, 1):
        grid_name = ctx["grid_name"]
        bbox      = ctx["bbox"]
        grid_xx   = ctx["grid_xx"]
        grid_yy   = ctx["grid_yy"]
        grid_props = ctx["grid_properties"]
        bbox_mask_2d = ctx["bbox_mask_2d"]
        elem_x_surf  = ctx["elem_x_surf"]
        elem_y_surf  = ctx["elem_y_surf"]
        land_mask    = ctx["land_mask"]

        log.info(f"    [MULTI] instance {inst_idx:02d}/{n_grids} | grid={grid_name}")

        # Tambah instance baru (SurfaceCurrent.NN)
        s111.utils.add_surface_current_instance(data_file)

        t_first_str = t_last_str = None
        n_written   = 0

        for i, t in enumerate(day_timesteps):
            speed, direction, t_dt = process_one_timestep(
                t, ds_open, dfsu_type, bbox_mask_2d,
                elem_x_surf, elem_y_surf, grid_xx, grid_yy
            )
            if speed is None:
                # [MEMFIX] Pulihkan handle DFSU jika ditandai poisoned oleh
                # kegagalan read pada process_one_timestep().
                global _DFSU_REOPEN_REQUESTED
                if _DFSU_REOPEN_REQUESTED:
                    ds_open = reopen_dfsu(ds_open, DFSU_PATH)
                    _DFSU_REOPEN_REQUESTED = False
                continue
            try:
                s111.utils.add_data_from_arrays(
                    speed, direction, data_file, grid_props, t_dt, 2
                )
                n_written += 1
                t_meta = t_dt.strftime("%Y%m%dT%H%M%SZ")
                if t_first_str is None: t_first_str = t_meta
                t_last_str = t_meta
            except Exception as e:
                log.error(f"      [{grid_name}] Gagal write {t}: {e}")

        # Instance-level metadata update (langsung ke S1xxObject via atribut)
        # Snake_case names come from s100py get_standard_properties().
        try:
            sc_feature = data_file.root.surface_current
            inst_obj   = sc_feature.surface_current[inst_idx - 1]
            if t_first_str:
                inst_obj.date_time_of_first_record = t_first_str
            if t_last_str:
                inst_obj.date_time_of_last_record  = t_last_str
            inst_obj.number_of_times      = np.int32(n_written)
            inst_obj.time_record_interval = np.int32(3600)
            inst_obj.num_grp              = np.int32(n_written)
            inst_obj.west_bound_longitude  = np.float64(bbox[0])
            inst_obj.east_bound_longitude  = np.float64(bbox[2])
            inst_obj.south_bound_latitude  = np.float64(bbox[1])
            inst_obj.north_bound_latitude  = np.float64(bbox[3])
            inst_obj.start_sequence        = "0,0"
        except Exception as e:
            log.warning(f"      [{grid_name}] instance attrs update lemah: {e}")

        # Update root metadata ( Sekali untuk seluruh file )
        try:
            s111.utils.update_metadata(data_file, grid_props, {
                "num_instances": n_grids,
                "dataDynamicity": 5,
                "numberOfGroups": n_written,
                "numberOfTimes":  n_written,
                "timeRecordInterval": 3600,
            })
        except Exception:
            pass

        written_summary.append({
            "inst_idx":     inst_idx,
            "grid_name":    grid_name,
            "n_groups":     n_written,
            "t_first":      t_first_str or "",
            "t_last":       t_last_str  or "",
            "land_mask":    land_mask,
            "grid_props":   grid_props,
        })
        log.info(f"      [{grid_name}] {n_written}/{n_times} group tertulis")

    # ── VALIDASI: Pastikan ada data yang ditulis sebelum write file ─────────
    total_written = sum(ws["n_groups"] for ws in written_summary)
    if total_written == 0:
        log.warning(f"  [MULTI] Tidak ada data valid untuk {date_key} dari semua grid — tidak membuat file")
        safe_close_and_delete(output_h5)
        return None, n_grids, 0

    # Tulis seluruh struktur ke HDF5
    try:
        s111.utils.write_data_file(data_file)
    except Exception as e:
        log.error(f"  [WRITE] Gagal menulis {fname}: {e} — file dihapus")
        safe_close_and_delete(output_h5)
        return None, n_grids, total_written

    # [MEMFIX] Tutup handle secara eksplisit SEBELUM land-mask/post-process
    # membuka file yang sama (mode r+). Jangan andalkan GC __dealloc__ untuk
    # flush — di bawah tekanan memori itu bisa gagal (Errno 22) dan
    # merusak/memotong file output.
    try:
        data_file.flush()
        data_file.close()
    except Exception as e:
        log.error(f"  [WRITE] Gagal flush/close {fname}: {e}")
    del data_file
    gc.collect()

    # ── POST-WRITE: land mask + post-process (handle multi-instance) ────────
    for ws in written_summary:
        if ws["land_mask"] is not None and ws["land_mask"].sum() > 0:
            try:
                apply_land_mask_to_h5_multi(
                    output_h5, ws["inst_idx"], ws["land_mask"], ws["grid_props"]
                )
            except Exception as e:
                log.error(f"  [LANDMASK] inst {ws['inst_idx']} ({ws['grid_name']}) GAGAL: {e}")

    try:
        _postprocess_hdf5_khoa_multi(output_h5, n_instances=n_grids)
    except Exception as e:
        log.error(f"  Post-process GAGAL: {e}")

    log.info(f"  OK {fname} | {n_grids} instance × ~{n_times} ts")
    return output_h5, n_grids, sum(ws["n_groups"] for ws in written_summary)




def process_single_grid(grid_name, grid_cfg, ds_open, dfsu_type,
                        elem_x, elem_y, layer_ids, unique_lids,
                        daily_groups, sorted_days, output_dir_base,
                        grid_rename_mapping=None, land_union=None):
    """
    [F30] Pipeline lengkap untuk memproses satu grid:
      1. BBOX mask
      2. Deteksi resolusi DFSU
      3. Adaptive regrid
      4. Build grid
      5. Build land mask
      6. Proses semua timestep (daily/hourly)

    Args:
        grid_name:      Nama grid (untuk logging & output naming)
        grid_cfg:       Dict {"bbox": [...]} atau object serupa
        ds_open:        Dataset MIKE IO yang sudah terbuka
        dfsu_type:      "2D" atau "3D"
        elem_x, elem_y: Koordinat elemen DFSU
        layer_ids:      Layer IDs (None untuk 2D)
        unique_lids:    Unique layer IDs (None untuk 2D)
        daily_groups:   Dict {date: [timesteps]}
        sorted_days:    List date yang sudah difilter
        output_dir_base: Direktori output utama

    Returns:
        List path file HDF5 yang dihasilkan
    """
    bbox = grid_cfg["bbox"]
    log.info("")
    log.info("=" * 68)
    log.info(f"  [GRID] {grid_name} | BBOX={bbox}")
    log.info("=" * 68)

    # ── 1. BBOX MASK ────────────────────────────────────────────────────────
    try:
        (bbox_mask_2d, bbox_mask_full,
         elem_x_surf, elem_y_surf) = build_bbox_mask(
            elem_x, elem_y, layer_ids, unique_lids, bbox, dfsu_type)
    except ValueError as e:
        log.error(f"  [GRID] {grid_name} SKIP — BBOX mask gagal: {e}")
        return []

    if len(elem_x_surf) == 0:
        log.error(f"  [GRID] {grid_name} SKIP — 0 elemen dalam BBOX")
        return []

    # ── 2. DFSU RESOLUTION ──────────────────────────────────────────────────
    dfsu_res_m = detect_dfsu_resolution(elem_x_surf, elem_y_surf)

    # ── 3. ADAPTIVE REGRID ──────────────────────────────────────────────────
    n_groups_est = (72 if OUTPUT_MODE == "3day"
                    else 1 if OUTPUT_MODE == "hourly" else 24)
    dx = GRID_FILE_DX
    dy = GRID_FILE_DY
    dx_new, dy_new, nx_est, ny_est, est_size, dx_m_final = adaptive_grid_resolution(
        bbox, dx, dy, n_groups_est, MAX_FILE_SIZE_MB, dfsu_res_m
    )

    # ── 4. BUILD GRID ───────────────────────────────────────────────────────
    try:
        grid_xx, grid_yy, grid_properties = build_grid_and_properties(bbox, dx_new, dy_new)
    except ValueError as e:
        log.error(f"  [GRID] {grid_name} SKIP — Build grid gagal: {e}")
        return []

    # ── 5. LAND MASK (per-grid, menggunakan grid final) ─────────────────────
    land_mask = None
    if APPLY_LAND_MASK:
        try:
            land_mask = build_land_mask(LAND_SHP, grid_properties, land_union=land_union)
        except Exception as e:
            log.error(f"  [GRID] {grid_name} [LANDMASK] GAGAL: {e}")

    # ── 6. METADATA (geographicIdentifier per-grid) ─────────────────────────
    first_t = ds_open.time[0]
    if hasattr(first_t, "to_pydatetime"):
        dt_issuance = first_t.to_pydatetime()
    elif isinstance(first_t, np.datetime64):
        dt_issuance = first_t.astype("M8[ms]").astype(datetime.datetime)
    else:
        dt_issuance = first_t

    # [LMK-05] Gunakan nama grid baru jika mapping tersedia
    display_name = grid_name
    output_name = grid_name
    if grid_rename_mapping and grid_name in grid_rename_mapping:
        output_name = grid_rename_mapping[grid_name]
        log.info(f"  [RENAME] {grid_name} -> {output_name}")
        display_name = f"{grid_name} ({output_name})"

    geo_id = f"{display_name} | {bbox[0]:.4f},{bbox[1]:.4f} -> {bbox[2]:.4f},{bbox[3]:.4f}"
    metadata = build_s111_metadata(dt_issuance, geographic_id=geo_id)

    # ── 7. OUTPUT DIRECTORY (grid-specific) ─────────────────────────────────
    # [F28] Subdirektori per grid: "Output/Grid1/", "Output/Grid2/", dst.
    # [LMK-05] Gunakan nama baru untuk output directory
    grid_output_dir = os.path.join(output_dir_base, output_name)
    os.makedirs(grid_output_dir, exist_ok=True)
    log.info(f"  [GRID] Output dir: {grid_output_dir}")

    # Prefix file: nama grid dimasukkan ke filename
    grid_prefix = f"{FILE_PREFIX}{output_name}_"

    # ── 8. PROSES SEMUA TIMESTEP ────────────────────────────────────────────
    output_files = []
    if OUTPUT_MODE == "hourly":
        # Satu file per timestep
        for day_idx, day_key in enumerate(sorted_days):
            log.info(f"  [{day_idx+1:04d}/{len(sorted_days)}] {day_key} | {grid_name}")
            for t in daily_groups[day_key]:
                try:
                    h5_path = process_hourly(
                        t=t, ds_open=ds_open, dfsu_type=dfsu_type,
                        bbox_mask_2d=bbox_mask_2d, elem_x_surf=elem_x_surf,
                        elem_y_surf=elem_y_surf, grid_xx=grid_xx, grid_yy=grid_yy,
                        grid_properties=grid_properties, metadata=metadata,
                        output_dir=grid_output_dir, prefix=grid_prefix, land_mask=land_mask,
                    )
                    if h5_path:
                        output_files.append(h5_path)
                    # [MEMFIX] Pulihkan handle DFSU jika process_one_timestep()
                    # menandai kegagalan read (kemungkinan poisoned reader state).
                    global _DFSU_REOPEN_REQUESTED
                    if _DFSU_REOPEN_REQUESTED:
                        ds_open = reopen_dfsu(ds_open, DFSU_PATH)
                        _DFSU_REOPEN_REQUESTED = False
                except Exception as e:
                    log.error(f"  [GRID] {grid_name} hourly error: {e}", exc_info=True)
                # [MEMFIX] gc.collect() per-file (bukan per-timestep) untuk
                # membebaskan data_file yang sudah ditutup + buffer timestep.
                gc.collect()
    else:
        # daily (1 hari/file) atau 3day (BLOCK_DAYS hari/file)
        blocks = build_day_blocks(sorted_days, OUTPUT_MODE)
        for block_idx, block_days in enumerate(blocks):
            block_ts = []
            for d in block_days:
                block_ts.extend(daily_groups[d])
            log.info(f"  [BLOCK {block_idx+1:04d}/{len(blocks)}] "
                     f"{block_days[0]}..{block_days[-1]} "
                     f"({len(block_days)} hari, {len(block_ts)} ts) | {grid_name}")
            try:
                h5_path, _ = process_daily(
                    date_key=block_days[0], date_key_end=block_days[-1],
                    day_timesteps=block_ts,
                    ds_open=ds_open, dfsu_type=dfsu_type,
                    bbox_mask_2d=bbox_mask_2d, elem_x_surf=elem_x_surf,
                    elem_y_surf=elem_y_surf, grid_xx=grid_xx, grid_yy=grid_yy,
                    grid_properties=grid_properties, metadata=metadata,
                    output_dir=grid_output_dir, prefix=grid_prefix, land_mask=land_mask,
                )
                if h5_path:
                    output_files.append(h5_path)
            except Exception as e:
                log.error(f"  [GRID] {grid_name} block error: {e}", exc_info=True)
            # [MEMFIX] gc.collect() per-file (bukan per-timestep) untuk
            # membebaskan data_file yang sudah ditutup + buffer timestep.
            gc.collect()

    log.info(f"  [GRID] {grid_name} SELESAI — {len(output_files)} file")
    return output_files

# ==============================================================================
# MAIN — v2.9 [F27][F28][F29][F30][F32][F33][F34][F35]
# ==============================================================================

def main():
    if OUTPUT_MODE not in ("daily", "hourly", "3day"):
        raise ValueError(f"OUTPUT_MODE tidak valid: '{OUTPUT_MODE}'")

    # [F33] Tentukan mode multi-instance: aktif hanya bila:
    #   - AREA_MODE == "gridfile"
    #   - MULTI_INSTANCE_PER_FILE = True
    #   - OUTPUT_MODE == "daily"  (multi-instance per-file untuk hourly
    #     tidak diimplementasikan; akan fallback ke per-grid)
    use_multi_instance = (
        AREA_MODE == "gridfile"
        and MULTI_INSTANCE_PER_FILE
        and OUTPUT_MODE == "daily"
    )

    log.info("=" * 68)
    log.info(f"  PIPELINE DFSU -> S-111 HDF5 | S-111 Ed. 2.0.0 | v2.9 MULTI-INSTANCE")
    log.info(f"  OUTPUT_MODE           = {OUTPUT_MODE.upper()}")
    log.info(f"  AREA_MODE             = {AREA_MODE}")
    log.info(f"  SELECTED_GRID         = {SELECTED_GRID if SELECTED_GRID else '<ALL>'}")
    log.info(f"  MULTI_INSTANCE        = {use_multi_instance}")
    log.info(f"  MAX_FILE_SIZE_MB      = {MAX_FILE_SIZE_MB} MB")
    log.info(f"  REGRID_MIN_RES_M      = {REGRID_MIN_RES_M} m")
    log.info(f"  REGRID_MAX_SKIP_FACTOR= {REGRID_MAX_SKIP_FACTOR}×")
    log.info(f"  REGRID_STRATEGY       = {REGRID_ADAPTIVE_STRATEGY}")
    log.info(f"  MASK_MODE             = {MASK_MODE}")
    log.info("=" * 68)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── BUKA DFSU SEKALI ────────────────────────────────────────────────────
    (ds_open, dfsu_type, n_layers, z_unique, unique_lids,
     elem_x, elem_y, layer_ids, bbox_z_col) = open_and_detect_dfsu(DFSU_PATH)

    # ── AREA CONFIGURATION ──────────────────────────────────────────────────
    if AREA_MODE == "list":
        BBOX, DX, DY = configure_area_from_list(SELECTED_AREA)
        grids_to_process = {"__single__": {"bbox": BBOX}}
    elif AREA_MODE == "manual":
        BBOX, DX, DY = configure_area_manual(BBOX_MANUAL, DX_MANUAL, DY_MANUAL)
        grids_to_process = {"__single__": {"bbox": BBOX}}
    elif AREA_MODE == "auto":
        BBOX, DX, DY = configure_area_auto(ds_open)
        grids_to_process = {"__single__": {"bbox": BBOX}}
    elif AREA_MODE == "gridfile":
        grids_to_process, _, _ = configure_area_from_gridfile(
            GRID_FILE_PATH, SELECTED_GRID, GRID_FILE_DX, GRID_FILE_DY)
    else:
        raise ValueError(f"AREA_MODE tidak valid: '{AREA_MODE}'")

    total_grids_raw = len(grids_to_process)
    log.info(f"[MAIN] Total grid terdeteksi: {total_grids_raw}")

    # ── [RENUMBER v3.1] Geografis + land-aware, port dari S-104 ──────────────
    land_union = None
    _bl = [cfg["bbox"] for cfg in grids_to_process.values()]
    master_bbox = (min(b[0] for b in _bl), min(b[1] for b in _bl),
                   max(b[2] for b in _bl), max(b[3] for b in _bl))
    if APPLY_LAND_MASK and HAS_GEO:
        try:
            land_union = load_land_union(LAND_SHP, master_bbox)
        except Exception as e:
            log.warning(f"[RENUMBER] load_land_union gagal: {e} — lanjut tanpa land_frac")

    if ENABLE_GRID_RENUMBER:
        grids_to_process, grid_rename_mapping, grid_classifications = \
            classify_and_renumber_grids(grids_to_process, land_union, elem_x, elem_y)
        save_grid_mapping_csv(grid_classifications, GRID_MAPPING_CSV)
        save_grid_mapping_json(grid_rename_mapping, grid_classifications, GRID_MAPPING_FILE)
        if ENABLE_CATALOG_PNG and HAS_MPL:
            try:
                plot_grid_catalog(grid_classifications, LAND_SHP, CATALOG_PNG_PATH)
            except Exception as e:
                log.warning(f"[CATALOG] gagal buat PNG: {e}")
    else:
        grid_rename_mapping = generate_new_grid_names(sorted(grids_to_process.keys()))

    total_grids = len(grids_to_process)
    log.info(f"[MAIN] Total grid akan diproses: {total_grids} "
             f"(dibuang: {total_grids_raw - total_grids})")

    # ── PREPARE TIMESTEP GROUPS (sekali, digunakan semua grid) ─────────────
    daily_groups = defaultdict(list)
    for t in ds_open.time:
        day_key = (t.date() if hasattr(t, "date")
                   else t.astype("M8[D]").astype(datetime.date))
        daily_groups[day_key].append(t)

    all_days    = sorted(daily_groups.keys())
    sorted_days = all_days.copy()

    if DATE_START is not None or DATE_END is not None:
        d_start = DATE_START if DATE_START is not None else all_days[0]
        d_end   = DATE_END   if DATE_END   is not None else all_days[-1]
        if d_start > d_end:
            raise ValueError(f"DATE_START > DATE_END")
        sorted_days = [d for d in all_days if d_start <= d <= d_end]
        if not sorted_days:
            raise ValueError(
                f"Tidak ada hari dalam {d_start}->{d_end}. "
                f"Domain: {all_days[0]}->{all_days[-1]}")
        log.info(f"Filter: {d_start}->{d_end} | {len(sorted_days)} hari")

    all_output_files = []

    if total_grids == 0:
        log.error("TIDAK ADA grid valid untuk diproses setelah klasifikasi.")
        return []

    # ── [F33] BRANCH: MULTI-INSTANCE PATH ───────────────────────────────────
    if use_multi_instance:
        log.info("")
        log.info("#" * 68)
        log.info(f"# MULTI-INSTANCE DCF2 | {total_grids} grid -> 1 HDF5/hari")
        log.info("#" * 68)

        # [F34] Siapkan semua konteks grid terlebih dahulu (1× untuk semua hari)
        grid_contexts = []
        for grid_idx, (grid_name, grid_cfg) in enumerate(grids_to_process.items(), 1):
            log.info(f"[PREP {grid_idx}/{total_grids}] {grid_name}")
            try:
                ctx = prepare_grid_context(
                    grid_name=grid_name, grid_cfg=grid_cfg,
                    ds_open=ds_open, dfsu_type=dfsu_type,
                    elem_x=elem_x, elem_y=elem_y,
                    layer_ids=layer_ids, unique_lids=unique_lids,
                    land_union=land_union,
                )
            except Exception as e:
                ctx = {"grid_name": grid_name, "bbox": grid_cfg["bbox"],
                       "status": f"prep_error: {e}"}
                log.error(f"[PREP] {grid_name} error: {e}", exc_info=True)
            grid_contexts.append(ctx)

        n_valid = sum(1 for c in grid_contexts if c.get("status") == "ok")
        log.info(f"[PREP] Konteks siap: {n_valid}/{total_grids} grid valid")

        # Metadata root (1× per file)
        first_t = ds_open.time[0]
        if hasattr(first_t, "to_pydatetime"):
            dt_issuance = first_t.to_pydatetime()
        elif isinstance(first_t, np.datetime64):
            dt_issuance = first_t.astype("M8[ms]").astype(datetime.datetime)
        else:
            dt_issuance = first_t
        metadata = build_s111_metadata(
            dt_issuance,
            geographic_id=f"Teluk Jakarta Multi-Instance ({n_valid} grids)",
        )

        # Iterasi hari — tiap hari = 1 HDF5 berisi semua grid
        for day_idx, day_key in enumerate(sorted_days):
            log.info(f"  [DAY {day_idx+1:04d}/{len(sorted_days)}] {day_key}")
            try:
                h5_path, n_inst, n_grp = process_daily_multi_instance(
                    date_key=day_key,
                    day_timesteps=daily_groups[day_key],
                    ds_open=ds_open,
                    dfsu_type=dfsu_type,
                    grid_contexts=grid_contexts,
                    metadata=metadata,
                    output_dir=OUTPUT_DIR,
                    prefix=FILE_PREFIX,
                )
                if h5_path:
                    all_output_files.append(h5_path)
            except Exception as e:
                log.error(f"  [MULTI] {day_key} GAGAL: {e}", exc_info=True)
            # [MEMFIX] gc.collect() per-file (bukan per-timestep) untuk
            # membebaskan data_file yang sudah ditutup + buffer timestep.
            gc.collect()

    # ── [F30] FALLBACK: PER-GRID PATH (mode lama) ───────────────────────────
    else:
        for grid_idx, (grid_name, grid_cfg) in enumerate(grids_to_process.items(), 1):
            new_name = grid_rename_mapping.get(grid_name, grid_name)
            log.info("")
            log.info("#" * 68)
            log.info(f"# [{grid_idx}/{total_grids}] MEMULAI: {grid_name} -> {new_name}")
            log.info("#" * 68)

            try:
                grid_files = process_single_grid(
                    grid_name=grid_name,
                    grid_cfg=grid_cfg,
                    ds_open=ds_open,
                    dfsu_type=dfsu_type,
                    elem_x=elem_x,
                    elem_y=elem_y,
                    layer_ids=layer_ids,
                    unique_lids=unique_lids,
                    daily_groups=daily_groups,
                    sorted_days=sorted_days,
                    output_dir_base=OUTPUT_DIR,
                    grid_rename_mapping=grid_rename_mapping,
                    land_union=land_union,
                )
                all_output_files.extend(grid_files)
                log.info(f"# [{grid_idx}/{total_grids}] SUKSES: {grid_name} -> {new_name} — {len(grid_files)} file")
            except Exception as e:
                log.error(f"# [{grid_idx}/{total_grids}] GAGAL: {grid_name} — {e}", exc_info=True)

            gc.collect()
            log.info("#" * 68)

    # ── RINGKASAN AKHIR ─────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 68)
    log.info(f"  PIPELINE SELESAI")
    log.info(f"  Mode            : {'MULTI-INSTANCE' if use_multi_instance else 'PER-GRID'}")
    log.info(f"  Grid diproses   : {total_grids}")
    log.info(f"  Total file HDF5 : {len(all_output_files)}")
    log.info(f"  Output utama    : {OUTPUT_DIR}")
    log.info("=" * 68)

    return all_output_files


if __name__ == "__main__":
    main()

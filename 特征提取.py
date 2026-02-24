# ============================================================
# Pelvic RT Plan Feature Extraction (CT folders + flat DICOM files)
# Add-ons (as requested):
#   1) Coverage: PTV_V95%, PTV_V100%, PTV_V105% (relative to Rx)
#   2) Gradient/Spill: PIV50, GI=PIV50/PIV100, R50=PIV50/PTVvol
#   3) RTPLAN basic complexity: TotalMU, NumBeams, TotalControlPoints, AvgMUperBeam
#
# Folder structure you described:
#   CT_ROOT/<patient_id>/(CT dicoms...)
#   RS_ROOT/*.dcm  (flat)
#   RD_ROOT/*.dcm  (flat)
#   RP_ROOT/*.dcm  (flat, optional but recommended)
#
# Output: one CSV row per patient.
#
# Dependencies:
#   pip install pydicom SimpleITK opencv-python numpy pandas scipy
# ============================================================

import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, binary_erosion


# =========================
# Paths (edit to your local)
# =========================
CT_ROOT = Path(r"C:\zq\pytorchlearn\Radiotherapy files\CT")
RS_ROOT = Path(r"C:\zq\pytorchlearn\Radiotherapy files\Structure")
RD_ROOT = Path(r"C:\zq\pytorchlearn\Radiotherapy files\Dose")
RP_ROOT = Path(r"C:\zq\pytorchlearn\Radiotherapy files\Plan")

OUT_CSV = Path(r"C:\zq\pytorchlearn\Radiotherapy files\features_full_coverage_GI_R50_plancomplexity.csv")

# Optional: per-patient prescription map (recommended)
# CSV columns: patient_id,prescription_gy
PRESCRIPTION_MAP_CSV = Path(r"C:\zq\pytorchlearn\Radiotherapy files\prescription_map.csv")
PRESCRIPTION_GY_DEFAULT = 45.0  # fallback if map missing or patient not found


# =========================
# Feature configs
# =========================
TARGET_STRUCT_ALIASES = {
    "PTV": ["PTV", "PTV1", "PCTV","PCTV1"],
    "Bladder": ["Bladder"],
    "BowelBag": ["BOWEL", "BOWELBAG", "BOWEL_BAG", "SMALLBOWEL", "SB"],
    "Rectum": ["Rectum"],
    "FemoralHead_L": ["Femoral Head L"],
    "FemoralHead_R": ["Femoral Head R"],
    "Body": ["BODY"]
}

# OAR dose endpoints
Vx_LIST_GY = [40, 45, 50]

# Ring shells around PTV (mm)
RINGS_MM = [(0, 5), (5, 10), (10, 20)]


# =========================
# Utilities
# =========================
def list_patient_ids_from_ct(ct_root: Path) -> List[str]:
    return sorted([p.name for p in ct_root.iterdir() if p.is_dir()])


def normalize_name(s: str) -> str:
    s = str(s).upper()
    s = re.sub(r"[\s\-]+", "_", s)
    return s


def match_structure_name(all_names: List[str], target_key: str) -> Optional[str]:
    aliases = TARGET_STRUCT_ALIASES.get(target_key, [target_key])
    aliases = [normalize_name(a) for a in aliases]

    # exact match
    for n in all_names:
        nn = normalize_name(n)
        if any(a == nn for a in aliases):
            return n

    # substring match
    for n in all_names:
        nn = normalize_name(n)
        if any(a in nn for a in aliases):
            return n
    return None


def pick_flat_dcm_by_patient_id(root: Path, patient_id: str, modality: str) -> Optional[Path]:
    """
    root is flat: 232101.dcm / 232101_RTDOSE.dcm / etc.
    Prefer filename contains patient_id; verify Modality via pydicom header.
    """
    if not root.exists():
        return None

    # fast candidates by filename
    candidates = []
    for fp in root.glob("*"):
        if fp.is_file() and (patient_id in fp.stem):
            candidates.append(fp)

    # fallback (slower) scan all .dcm if none matched by name
    if not candidates:
        candidates = [fp for fp in root.glob("*.dcm") if fp.is_file()]

    modality = modality.upper()
    for fp in candidates:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            if str(getattr(ds, "Modality", "")).upper() == modality:
                return fp
        except Exception:
            continue
    return None


# =========================
# Prescription map
# =========================
def load_prescription_map(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path)
    if "patient_id" not in df.columns or "prescription_gy" not in df.columns:
        return {}
    mp = {}
    for _, r in df.iterrows():
        pid = str(r["patient_id"]).strip()
        try:
            mp[pid] = float(r["prescription_gy"])
        except Exception:
            continue
    return mp


# =========================
# SimpleITK IO
# =========================
def read_ct_image(ct_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(ct_dir))
    if not series_ids:
        raise RuntimeError(f"No CT series found in {ct_dir}")
    series_files = reader.GetGDCMSeriesFileNames(str(ct_dir), series_ids[0])
    reader.SetFileNames(series_files)
    return reader.Execute()


def read_rtdose_image(rtdose_fp: Path) -> sitk.Image:
    ds = pydicom.dcmread(str(rtdose_fp), force=True)
    dose_scaling = float(getattr(ds, "DoseGridScaling", 1.0))
    dose_img = sitk.ReadImage(str(rtdose_fp))
    dose_img = sitk.Cast(dose_img, sitk.sitkFloat32) * dose_scaling
    return dose_img


def resample_to_reference(moving: sitk.Image, reference: sitk.Image, is_label: bool = False) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving)


# =========================
# RTSTRUCT -> masks (needs opencv)
# =========================
def rtstruct_to_label_image(rtstruct_fp: Path, reference_ct: sitk.Image) -> Dict[str, sitk.Image]:
    import cv2  # required
    ds = pydicom.dcmread(str(rtstruct_fp), force=True)

    roi_names = {}
    for roi in getattr(ds, "StructureSetROISequence", []):
        roi_names[int(roi.ROINumber)] = str(roi.ROIName)

    size = reference_ct.GetSize()      # (x,y,z)
    spacing = reference_ct.GetSpacing()
    origin = reference_ct.GetOrigin()
    direction = reference_ct.GetDirection()
    inv_dir = np.linalg.inv(np.array(direction).reshape(3, 3))

    def phys_to_index(pt_xyz):
        v = np.array(pt_xyz) - np.array(origin)
        v = inv_dir.dot(v)
        return v / np.array(spacing)

    masks = {}
    roi_contours = getattr(ds, "ROIContourSequence", None)
    if roi_contours is None:
        return masks

    for roi_contour in roi_contours:
        roi_num = int(roi_contour.ReferencedROINumber)
        name = roi_names.get(roi_num, f"ROI_{roi_num}")

        mask_np = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)  # z,y,x
        cont_seq = getattr(roi_contour, "ContourSequence", None)
        if cont_seq is None:
            continue

        for cont in cont_seq:
            if not hasattr(cont, "ContourData"):
                continue
            pts = np.array(cont.ContourData, dtype=float).reshape(-1, 3)
            idx_pts = np.array([phys_to_index(p) for p in pts])

            z_index = int(round(np.mean(idx_pts[:, 2])))
            if z_index < 0 or z_index >= size[2]:
                continue

            poly = idx_pts[:, :2]
            poly[:, 0] = np.clip(poly[:, 0], 0, size[0] - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, size[1] - 1)

            poly_int = np.round(poly).astype(np.int32).reshape((-1, 1, 2))  # (x,y)
            slice_img = mask_np[z_index]
            cv2.fillPoly(slice_img, [poly_int], 1)
            mask_np[z_index] = slice_img

        mask_img = sitk.GetImageFromArray(mask_np.astype(np.uint8))
        mask_img.CopyInformation(reference_ct)
        masks[name] = mask_img

    return masks


# =========================
# Feature functions
# =========================
def mask_volume_cc(mask_arr: np.ndarray, spacing_xyz) -> float:
    sx, sy, sz = spacing_xyz
    voxel_cc = (sx * sy * sz) / 1000.0
    return float(mask_arr.sum() * voxel_cc)


def dose_stats_in_mask(dose_arr: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if mask.sum() == 0:
        return {"Dmean": np.nan, "Dmax": np.nan, "D98": np.nan, "D95": np.nan, "D50": np.nan, "D2": np.nan}
    vals = dose_arr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"Dmean": np.nan, "Dmax": np.nan, "D98": np.nan, "D95": np.nan, "D50": np.nan, "D2": np.nan}
    return {
        "Dmean": float(np.mean(vals)),
        "Dmax": float(np.max(vals)),
        # Note: D98 is 2nd percentile; D2 is 98th percentile (common DVH convention)
        "D98": float(np.percentile(vals, 2)),
        "D95": float(np.percentile(vals, 5)),
        "D50": float(np.percentile(vals, 50)),
        "D2": float(np.percentile(vals, 98)),
    }


def Vx_percent(dose_arr: np.ndarray, mask: np.ndarray, x_gy: float) -> float:
    if mask.sum() == 0:
        return np.nan
    vals = dose_arr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(100.0 * np.mean(vals >= x_gy))


def Vrel_percent(dose_arr: np.ndarray, mask: np.ndarray, rel: float, rx_gy: float) -> float:
    """Volume % receiving >= rel*Rx"""
    if (rx_gy is None) or (not np.isfinite(rx_gy)) or rx_gy <= 0:
        return np.nan
    return Vx_percent(dose_arr, mask, rel * rx_gy)


def min_mean_surface_distance_mm(a: np.ndarray, b: np.ndarray, spacing_xyz) -> Dict[str, float]:
    if a.sum() == 0 or b.sum() == 0:
        return {"minDist_mm": np.nan, "meanSurfDist_mm": np.nan}

    sx, sy, sz = spacing_xyz
    sampling = (sz, sy, sx)  # z,y,x for scipy
    dist_to_b = distance_transform_edt(~b, sampling=sampling)

    a_er = binary_erosion(a, iterations=1)
    a_boundary = np.logical_and(a, ~a_er)
    dists = dist_to_b[a_boundary]
    if dists.size == 0:
        dists = dist_to_b[a]
    return {"minDist_mm": float(np.min(dists)), "meanSurfDist_mm": float(np.mean(dists))}


def ring_masks_from_ptv(ptv_arr: np.ndarray, spacing_xyz, rings_mm: List[Tuple[int, int]]) -> Dict[str, np.ndarray]:
    sx, sy, sz = spacing_xyz
    sampling = (sz, sy, sx)
    dist_from_ptv = distance_transform_edt(~ptv_arr, sampling=sampling)
    out = {}
    for a, b in rings_mm:
        out[f"ring_{a}_{b}mm"] = np.logical_and(dist_from_ptv > a, dist_from_ptv <= b)
    return out


def compute_HI(ptv_stats: Dict[str, float]) -> float:
    # HI = (D2 - D98) / D50
    D2, D98, D50 = ptv_stats["D2"], ptv_stats["D98"], ptv_stats["D50"]
    if not np.isfinite(D2) or not np.isfinite(D98) or not np.isfinite(D50) or D50 == 0:
        return np.nan
    return float((D2 - D98) / D50)


def compute_CI_paddick(dose_arr: np.ndarray, ptv_mask: np.ndarray, spacing_xyz, rx_gy: float) -> Dict[str, float]:
    """
    Paddick CI = (TV_PIV^2) / (TV * PIV), where PIV = V(dose>=Rx).
    """
    if rx_gy is None or (not np.isfinite(rx_gy)) or rx_gy <= 0:
        return {"CI_Paddick": np.nan, "PIV100_cc": np.nan, "TV_PIV100_cc": np.nan}

    piv = dose_arr >= float(rx_gy)
    tv_cc = mask_volume_cc(ptv_mask, spacing_xyz)
    piv_cc = mask_volume_cc(piv, spacing_xyz)
    tv_piv_cc = mask_volume_cc(np.logical_and(ptv_mask, piv), spacing_xyz)

    if tv_cc <= 0 or piv_cc <= 0:
        return {"CI_Paddick": np.nan, "PIV100_cc": float(piv_cc), "TV_PIV100_cc": float(tv_piv_cc)}

    ci = (tv_piv_cc ** 2) / (tv_cc * piv_cc)
    return {"CI_Paddick": float(ci), "PIV100_cc": float(piv_cc), "TV_PIV100_cc": float(tv_piv_cc)}


def compute_GI_R50(dose_arr: np.ndarray, ptv_mask: np.ndarray, spacing_xyz, rx_gy: float, piv100_cc: float) -> Dict[str, float]:
    """
    GI = PIV50 / PIV100
    R50 = PIV50 / TV (PTV volume)
    """
    if rx_gy is None or (not np.isfinite(rx_gy)) or rx_gy <= 0:
        return {"PIV50_cc": np.nan, "GI": np.nan, "R50": np.nan}

    piv50 = dose_arr >= float(0.5 * rx_gy)
    piv50_cc = mask_volume_cc(piv50, spacing_xyz)
    tv_cc = mask_volume_cc(ptv_mask, spacing_xyz)

    gi = np.nan
    if piv100_cc is not None and np.isfinite(piv100_cc) and piv100_cc > 0:
        gi = float(piv50_cc / piv100_cc)

    r50 = np.nan
    if tv_cc > 0:
        r50 = float(piv50_cc / tv_cc)

    return {"PIV50_cc": float(piv50_cc), "GI": gi, "R50": r50}


# =========================
# RTPLAN basic complexity
# =========================
def parse_rtplan_basic_complexity(rtplan_fp: Path) -> Dict[str, float]:
    """
    Basic, robust metrics that typically exist in RTPLAN:
      - NumBeams (excluding setup beams)
      - TotalMU (sum BeamMeterset or FractionGroupSequence)
      - AvgMUperBeam
      - TotalControlPoints (sum over beams)
    """
    if rtplan_fp is None or (not Path(rtplan_fp).exists()):
        return {
            "Plan_present": 0,
            "NumBeams": np.nan,
            "TotalMU": np.nan,
            "AvgMUperBeam": np.nan,
            "TotalControlPoints": np.nan
        }

    try:
        ds = pydicom.dcmread(str(rtplan_fp), stop_before_pixels=True, force=True)
    except Exception:
        return {
            "Plan_present": 0,
            "NumBeams": np.nan,
            "TotalMU": np.nan,
            "AvgMUperBeam": np.nan,
            "TotalControlPoints": np.nan
        }

    beams = getattr(ds, "BeamSequence", [])
    # filter setup beams if possible
    treatment_beams = []
    for b in beams:
        bt = str(getattr(b, "BeamType", "")).upper()  # "TREATMENT" / "SETUP"
        if bt == "" or bt == "TREATMENT":
            treatment_beams.append(b)

    num_beams = len(treatment_beams) if treatment_beams else len(beams)

    # Total control points
    total_cps = 0
    for b in (treatment_beams if treatment_beams else beams):
        cps = getattr(b, "ControlPointSequence", None)
        if cps is not None:
            total_cps += len(cps)

    # Total MU: prefer FractionGroupSequence BeamMeterset if present
    total_mu = None
    try:
        fgs = getattr(ds, "FractionGroupSequence", None)
        if fgs is not None and len(fgs) > 0:
            fg0 = fgs[0]
            ref_beams = getattr(fg0, "ReferencedBeamSequence", None)
            if ref_beams is not None:
                mus = []
                for rb in ref_beams:
                    if hasattr(rb, "BeamMeterset"):
                        mus.append(float(rb.BeamMeterset))
                if mus:
                    total_mu = float(np.sum(mus))
    except Exception:
        total_mu = None

    # fallback: sum BeamMeterset in BeamSequence if present
    if total_mu is None:
        mus = []
        for b in (treatment_beams if treatment_beams else beams):
            if hasattr(b, "BeamMeterset"):
                try:
                    mus.append(float(b.BeamMeterset))
                except Exception:
                    pass
        if mus:
            total_mu = float(np.sum(mus))

    avg_mu = np.nan
    if total_mu is not None and np.isfinite(total_mu) and num_beams and num_beams > 0:
        avg_mu = float(total_mu / num_beams)

    return {
        "Plan_present": 1,
        "NumBeams": float(num_beams) if num_beams else np.nan,
        "TotalMU": float(total_mu) if total_mu is not None else np.nan,
        "AvgMUperBeam": avg_mu,
        "TotalControlPoints": float(total_cps) if total_cps else np.nan
    }


# =========================
# Extraction core
# =========================
def extract_one(patient_id: str, rx_map: Dict[str, float]) -> Optional[Dict]:
    ct_dir = CT_ROOT / patient_id
    if not ct_dir.exists():
        print(f"[Skip] CT folder missing: {patient_id}")
        return None

    rs_fp = pick_flat_dcm_by_patient_id(RS_ROOT, patient_id, "RTSTRUCT")
    rd_fp = pick_flat_dcm_by_patient_id(RD_ROOT, patient_id, "RTDOSE")
    rp_fp = pick_flat_dcm_by_patient_id(RP_ROOT, patient_id, "RTPLAN")  # recommended, may be None

    if rs_fp is None or rd_fp is None:
        print(f"[Skip] missing RTSTRUCT/RTDOSE file: {patient_id} | RS:{rs_fp} RD:{rd_fp}")
        return None

    rx_gy = float(rx_map.get(patient_id, PRESCRIPTION_GY_DEFAULT))

    try:
        ct_img = read_ct_image(ct_dir)
        dose_img = read_rtdose_image(rd_fp)
        dose_on_ct = resample_to_reference(dose_img, ct_img, is_label=False)
        dose_arr = sitk.GetArrayFromImage(dose_on_ct).astype(np.float32)

        masks = rtstruct_to_label_image(rs_fp, ct_img)
        names = list(masks.keys())

        ptv_name = match_structure_name(names, "PTV")
        if ptv_name is None:
            print(f"[Skip] no PTV in RTSTRUCT: {patient_id}")
            return None

        spacing = ct_img.GetSpacing()  # (x,y,z)
        ptv_arr = (sitk.GetArrayFromImage(masks[ptv_name]) > 0)

        row = {
            "patient_id": patient_id,
            "CT_dir": str(ct_dir),
            "RTSTRUCT": str(rs_fp),
            "RTDOSE": str(rd_fp),
            "RTPLAN": str(rp_fp) if rp_fp else "",
            "prescription_gy": rx_gy,
            "PTV_name": ptv_name,
        }

        # --- PTV DVH points
        ptv_stats = dose_stats_in_mask(dose_arr, ptv_arr)
        row.update({f"PTV_{k}_Gy": v for k, v in ptv_stats.items()})
        row["PTV_volume_cc"] = mask_volume_cc(ptv_arr, spacing)

        # --- HI
        row["HI_D2D98_over_D50"] = compute_HI(ptv_stats)

        # --- Coverage relative to Rx
        row["PTV_V95pctRx_pct"] = Vrel_percent(dose_arr, ptv_arr, 0.95, rx_gy)
        row["PTV_V100pctRx_pct"] = Vrel_percent(dose_arr, ptv_arr, 1.00, rx_gy)
        row["PTV_V105pctRx_pct"] = Vrel_percent(dose_arr, ptv_arr, 1.05, rx_gy)

        # --- CI (Paddick) + PIV100
        ci_pack = compute_CI_paddick(dose_arr, ptv_arr, spacing, rx_gy)
        row.update(ci_pack)

        # --- GI / R50 + PIV50
        gi_pack = compute_GI_R50(dose_arr, ptv_arr, spacing, rx_gy, piv100_cc=ci_pack.get("PIV100_cc", np.nan))
        row.update(gi_pack)

        # --- OARs (present-dependent)
        for oar in ["Bladder", "BowelBag", "Rectum", "FemoralHead_L", "FemoralHead_R", "Body"]:
            oname = match_structure_name(names, oar)
            if oname is None:
                row[f"{oar}_present"] = 0
                continue

            row[f"{oar}_present"] = 1
            row[f"{oar}_name"] = oname

            oar_arr = (sitk.GetArrayFromImage(masks[oname]) > 0)
            row[f"{oar}_volume_cc"] = mask_volume_cc(oar_arr, spacing)
            row[f"{oar}_overlap_cc"] = mask_volume_cc(np.logical_and(ptv_arr, oar_arr), spacing)

            dist = min_mean_surface_distance_mm(ptv_arr, oar_arr, spacing)
            row[f"{oar}_PTV_minDist_mm"] = dist["minDist_mm"]
            row[f"{oar}_PTV_meanSurfDist_mm"] = dist["meanSurfDist_mm"]

            st = dose_stats_in_mask(dose_arr, oar_arr)
            row[f"{oar}_Dmean_Gy"] = st["Dmean"]
            row[f"{oar}_Dmax_Gy"] = st["Dmax"]
            row[f"{oar}_D2_Gy"] = st["D2"]
            for vx in Vx_LIST_GY:
                row[f"{oar}_V{vx}Gy_pct"] = Vx_percent(dose_arr, oar_arr, vx)

        # --- rings (spill proxies)
        rings = ring_masks_from_ptv(ptv_arr, spacing, RINGS_MM)
        for rname, rmask in rings.items():
            st = dose_stats_in_mask(dose_arr, rmask)
            row[f"{rname}_Dmean_Gy"] = st["Dmean"]
            row[f"{rname}_D95_Gy"] = st["D95"]
            row[f"{rname}_Dmax_Gy"] = st["Dmax"]

        # --- RTPLAN basic complexity
        row.update(parse_rtplan_basic_complexity(rp_fp))

        return row

    except Exception as e:
        print(f"[Fail] {patient_id}: {e}")
        return None


def main():
    # hard dependency check
    try:
        import cv2  # noqa
    except Exception:
        raise RuntimeError("Missing dependency: opencv-python. Please run: pip install opencv-python")

    rx_map = load_prescription_map(PRESCRIPTION_MAP_CSV)
    if rx_map:
        print(f"[Rx map] loaded: {len(rx_map)} patients from {PRESCRIPTION_MAP_CSV}")
    else:
        print(f"[Rx map] not found or invalid. Using default Rx={PRESCRIPTION_GY_DEFAULT} Gy for all.")

    pids = list_patient_ids_from_ct(CT_ROOT)
    print(f"[Patients] {len(pids)}")

    rows = []
    kept = 0
    for i, pid in enumerate(pids, 1):
        r = extract_one(pid, rx_map)
        if r is not None:
            rows.append(r)
            kept += 1
        if i % 20 == 0:
            print(f"Processed {i}/{len(pids)} | kept {kept}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[Saved] {OUT_CSV}")
    print("Shape:", df.shape)
    print(df.head(3).T)


if __name__ == "__main__":
    main()

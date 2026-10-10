"""Step 0: ingest the MIAS database into the workspace (raw layer of the platform layout).
  python bc.py ingest --src ../inbox/all-mias_tar.gz        (also accepts archive.zip or an extracted folder)
Writes  workspace/raw/tenant=is360/source=mias/images/<patient>/<image_id>/original.pgm
        workspace/raw/.../licence/Licence.txt, source_info.json
        workspace/manifests/ingest_manifest.csv       (one row per image)
        workspace/processed/.../annotations/ann_v1/annotations.csv   (one row per annotation, coord_frame recorded)
"""
import os, sys, json, shutil, tarfile, zipfile, hashlib, argparse, uuid
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import config as C
from mias_parse import parse_info, image_table

EXPECTED_IMAGES, EXPECTED_ROWS = 322, 330

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def resolve_source(src):
    """-> directory containing Info.txt + mdb*.pgm (+ optional path to all_mias_scans.h5)."""
    p = Path(src); work = Path(C.WS) / "inbox_extracted"; work.mkdir(parents=True, exist_ok=True); h5 = None
    if p.is_dir():
        for cand in [p, p / "all-mias"]:
            if (cand / "Info.txt").exists(): return cand, (p / "all_mias_scans.h5" if (p / "all_mias_scans.h5").exists() else None)
        raise FileNotFoundError("Info.txt not found in folder")
    if p.is_file() and p.suffix != ".zip" and tarfile.is_tarfile(p):
        with tarfile.open(p) as t:
            try: t.extractall(work / "all-mias", filter="data")
            except TypeError: t.extractall(work / "all-mias")
        return work / "all-mias", None
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if n.startswith("all-mias/") and not n.endswith("/"): z.extract(n, work)
                if n.endswith(".h5"): z.extract(n, work); h5 = work / n
        return work / "all-mias", h5
    raise ValueError("unsupported --src")

def ingest(src, verify_h5=True):
    C.ensure_dirs(); d, h5 = resolve_source(src)
    ann = parse_info((d / "Info.txt").read_text(encoding="latin-1")); imgs = image_table(ann)
    assert len(imgs) == EXPECTED_IMAGES and len(ann) == EXPECTED_ROWS, f"expected {EXPECTED_IMAGES} images / {EXPECTED_ROWS} rows, got {len(imgs)}/{len(ann)}"
    rows = []
    for r in imgs.itertuples():
        f = d / f"{r.refnum}.pgm"
        if not f.exists(): raise FileNotFoundError(f)
        a = np.array(Image.open(f)); assert a.shape == (1024, 1024) and a.dtype == np.uint8, f"unexpected image {f}"
        iid = "img_" + sha256(f)[:12]; dst = Path(C.image_dir("raw", r.patient_id, iid)); dst.mkdir(parents=True, exist_ok=True)
        if not (dst / "original.pgm").exists(): shutil.copy2(f, dst / "original.pgm")
        sid = uuid.uuid5(uuid.NAMESPACE_URL, f"is360:mias:pair:{r.patient_id}").hex[:12]
        rows.append({**r._asdict(), "image_id": iid, "study_id": "st_" + sid, "study_label": "IS360-BR-MIAS-" + r.patient_id[-3:],
                     "laterality_source": "file_rule", "view_source": "assumed", "sha256": sha256(f), "pixel_spacing_mm": C.PIXEL_SPACING_MM, "view": "MLO",
                     "view_note": "assumed MLO from visual inspection; verify", "raw_path": os.path.relpath(dst / "original.pgm", C.WS)})
    man = pd.DataFrame(rows).drop(columns=["Index"], errors="ignore")
    assert man.image_id.is_unique, "duplicate image ids (identical files)"
    man.to_csv(os.path.join(C.MAN, "ingest_manifest.csv"), index=False)
    ann = ann.merge(man[["refnum", "image_id", "patient_id"]], on="refnum")
    ad = os.path.join(C.PROC, "annotations", C.ANNOT_VERSION); os.makedirs(ad, exist_ok=True); ann.to_csv(os.path.join(ad, "annotations.csv"), index=False)
    ld = os.path.join(C.RAW, "licence"); os.makedirs(ld, exist_ok=True)
    for n in ["Licence.txt", "Info.txt"]:
        if (d / n).exists(): shutil.copy2(d / n, os.path.join(ld, n))
    json.dump(dict(source="MIAS MiniMammographic Database", licence="research_only (see Licence.txt: no renting, leasing or selling; cite in publications)",
                   citation="Suckling J et al. (1994) The Mammographic Image Analysis Society digital mammogram database. Excerpta Medica. International Congress Series 1069, pp 375-378.",
                   pixel_spacing_mm=C.PIXEL_SPACING_MM, n_images=len(man), n_patients=int(man.patient_id.nunique()), coord_frame="mias_bottom_left_1024"),
              open(os.path.join(C.RAW, "source_info.json"), "w"), indent=2)
    msg = ""
    if verify_h5 and h5 and h5.exists():
        try:
            import h5py
            with h5py.File(h5, "r") as f:
                refs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["REFNUM"][:]]; scans = f["scan"]
                first = {}; [first.setdefault(r, i) for i, r in enumerate(refs)]
                bad = [r for r, i in first.items() if not np.array_equal(scans[i], np.array(Image.open(d / f"{r}.pgm")))]
            msg = f"h5 cross-check: {len(first) - len(bad)}/{len(first)} images identical to PGM" + (f" (MISMATCH: {bad[:5]})" if bad else "")
        except Exception as e: msg = f"h5 cross-check skipped ({e})"
    return man, ann, msg

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--no-h5-check", action="store_true"); a = ap.parse_args()
    man, ann, msg = ingest(a.src, not a.no_h5_check)
    print(f"images: {len(man)}  patients: {man.patient_id.nunique()}  annotation rows: {len(ann)}")
    print("image class:", man.image_class.value_counts().to_dict()); print("density:", man.bg.value_counts().to_dict())
    print("lesion rows with ROI:", int(ann.has_roi.sum()), "of", int((ann.cls != 'NORM').sum())); print(msg)

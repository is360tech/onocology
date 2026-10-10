"""Parse the MIAS Info.txt annotation list into clean tables.
Info.txt columns: refnum, background (F/G/D), class (CALC/CIRC/SPIC/MISC/ARCH/ASYM/NORM), severity (B/M), x, y, radius.
Origin of (x, y) is the BOTTOM-LEFT corner of the 1024x1024 image. Even file numbers = left breast, odd = right breast.
Pairs of consecutive numbers (mdb001+mdb002) are the two breasts of ONE patient."""
import re, math
import pandas as pd

ROW = re.compile(r"^(mdb\d{3})\s+([FGD])\s+(CALC|CIRC|SPIC|MISC|ARCH|ASYM|NORM)\s*([BM])?\s*(?:(\d+)\s+(\d+)\s+(\d+))?\s*(?:\*NOTE\s*\d+\*)?\s*$")
COORD_FRAME = "mias_bottom_left_1024"

def parse_info(text):
    rows = []
    for line in text.splitlines():
        m = ROW.match(line.strip())
        if not m: continue
        ref, bg, cls, sev, x, y, r = m.groups()
        rows.append(dict(refnum=ref, bg=bg, cls=cls, severity=sev if sev else None,
                         x=float(x) if x else None, y=float(y) if y else None, radius=float(r) if r else None))
    df = pd.DataFrame(rows)
    if df.empty: raise ValueError("no annotation rows found in Info.txt")
    df["coord_frame"] = COORD_FRAME
    df["has_roi"] = df.x.notna()
    df["lesion_id"] = df.groupby("refnum").cumcount().astype(str)
    df["lesion_id"] = df.refnum + "_L" + df.lesion_id
    return df

def image_table(ann):
    """One row per image."""
    out = []
    for ref, g in ann.groupby("refnum", sort=True):
        n = int(ref[3:]); pair = math.ceil(n / 2)
        sev = set(g.severity.dropna())
        image_class = "malignant" if "M" in sev else ("benign" if "B" in sev else "normal")
        out.append(dict(refnum=ref, image_num=n, patient_id=f"mias_p{pair:03d}", laterality="L" if n % 2 == 0 else "R",
                        bg=g.bg.iloc[0], image_class=image_class, abnormal=int(image_class != "normal"),
                        n_lesions=int((g.cls != "NORM").sum()), lesion_types=";".join(sorted(set(g.cls) - {"NORM"}))))
    return pd.DataFrame(out)

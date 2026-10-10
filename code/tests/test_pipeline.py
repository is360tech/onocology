"""Self-test on synthetic data (no MIAS needed, ~1 min). Runs in a throw-away HOME.   python bc.py test"""
import os, sys, json, tempfile
_HOME = tempfile.mkdtemp(prefix="bc_test_home_"); os.environ["BC_HOME"] = _HOME
import numpy as np, pandas as pd
import config as C, mias_parse, preprocess_lib as L, feature_lib as F, partition as PT, provenance
from models import NeuroSyncSLDA, paz_gate
from evalkit import cv_repeat, metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
rng = np.random.default_rng(0)

def test_parse():
    txt = "mdb001 G CIRC B 535 425 197\nmdb002 G NORM\nmdb003 D CALC M *NOTE 3*\nmdb004 D NORM \nmdb005 F CIRC B 477 133 30\nmdb005 F CIRC B 500 168 26\n"
    a = mias_parse.parse_info(txt); im = mias_parse.image_table(a)
    assert len(a) == 6 and a.has_roi.sum() == 3 and len(im) == 5
    r3 = im[im.refnum == "mdb003"].iloc[0]; assert r3.image_class == "malignant" and r3.laterality == "R" and r3.patient_id == "mias_p002"
    assert im[im.refnum == "mdb002"].iloc[0].laterality == "L" and im[im.refnum == "mdb005"].iloc[0].n_lesions == 2; print("parse ok")

def phantom(w=1024, chest="right"):
    img = np.zeros((w, w), np.uint8); yy, xx = np.mgrid[0:w, 0:w]
    breast = ((xx - 900) / 480.0) ** 2 + ((yy - 520) / 380.0) ** 2 < 1; breast &= xx <= 900 if False else breast
    img[breast] = 90; img[xx > 900] = 0 if chest == "right" else img[xx > 900]
    img[(np.hypot(xx - 560, yy - 480) < 30) & breast] = 230            # bright "lesion" at x=560, y_top=480
    return img

def test_transform():
    img = phantom(); cx, cy = 560.0, 480.0
    for v in ["P0", "P1", "P2", "P3", "P4"]:
        res = L.run_variant(img, v, "right", 512); tf = res["transform"]; xm, ym = tf.map_point(cx, cy)
        assert 0 <= xm < 512 and 0 <= ym < 512, (v, xm, ym)
        win = res["image"][int(ym) - 6:int(ym) + 7, int(xm) - 6:int(xm) + 7]; ref = np.median(res["image"][res["breast"]])
        assert win.mean() > ref * 1.15, f"{v}: mapped point is not on the lesion ({win.mean():.0f} vs {ref:.0f})"
        x2, y2 = tf.inverse_point(xm, ym); assert abs(x2 - cx) < 1e-6 and abs(y2 - cy) < 1e-6
    print("transform ok (lesion stays under the mapped point for P0..P4, inverse exact)")

def synth_manifest(n=90):
    rows = []
    for i in range(n):
        w = ["normal", "benign", "malignant"][i % 3]; pid = f"mias_p{i:03d}"
        for lat in "LR": rows.append(dict(patient_id=pid, image_id=f"img_{pid}{lat}", image_class=w if lat == "L" else "normal", refnum=f"mdb{2*i+1:03d}", bg="FGD"[i % 3], laterality=lat))
    return pd.DataFrame(rows)

def test_partition():
    C.ensure_dirs(); m = synth_manifest(); m.to_csv(os.path.join(C.MAN, "ingest_manifest.csv"), index=False)
    a = PT.make_split(m); b = PT.make_split(m); assert a.equals(b), "split not deterministic"
    assert a.partition.value_counts().to_dict() == {"MS": 63, "DEV": 27}, a.partition.value_counts().to_dict()
    for w, g in a.groupby("worst"): assert abs((g.partition == "DEV").sum() - round(0.3 * len(g))) == 0
    info = PT.freeze(); ids = PT.load(); assert set(ids[ids.partition == "DEV"].patient_id).isdisjoint(ids[ids.partition == "MS"].patient_id)
    sel = PT.select(m, "DEV"); assert set(sel.patient_id) == set(ids[ids.partition == "DEV"].patient_id) and len(sel) == 54
    open(os.path.join(C.DATASET_ROOT, "partitions", "MS.csv"), "a").write("tamper\n")
    try: PT.load(); raise AssertionError("tampering not detected")
    except SystemExit: pass
    print("partition ok (deterministic, patient-level, stratified, tamper-evident)")

def test_features_models():
    x = (rng.random((96, 96)) * 200).astype(np.uint8); m = np.ones_like(x, bool); f = F.extract(x, m, with_hist=True)
    assert len(f) > 60 and all(np.isfinite(v) for v in f.values())
    X = rng.normal(size=(200, 6)); y = np.where(X[:, 0] + 0.3 * rng.normal(size=200) > 0, "a", "b"); Xs = (X - X.mean(0)) / X.std(0)
    ours = NeuroSyncSLDA(lam=1e-6, uniform_priors=True).fit(Xs, y).predict(Xs); ref = LDA(solver="lsqr", priors=[.5, .5]).fit(Xs, y).predict(Xs); assert (ours == ref).mean() > 0.97
    mg, amb = paz_gate(np.array([[.55, .45], [.95, .05]]), 0.15); assert amb.tolist() == [True, False]; print("features / SLDA / PAZ ok")

def test_cv_no_leakage():
    n_p = 60; groups = np.repeat([f"p{i}" for i in range(n_p)], 2); yp = np.array(["M", "B"] * (n_p // 2)); y = np.repeat(yp, 2)
    X = rng.normal(size=(len(y), 8)); X[:, 0] += (y == "M") * 1.5
    P, _ = cv_repeat(X, y, groups, "slda", 1); auc = metrics(y, P, np.unique(y), "M")["auc"]; assert auc > 0.75, auc
    yp2 = rng.permutation(y); P2, _ = cv_repeat(X, yp2, groups, "slda", 1); assert abs(metrics(yp2, P2, np.unique(yp2), "M")["auc"] - 0.5) < 0.2; print(f"grouped CV ok (AUC {auc:.2f}; permuted ~0.5; leakage assertion active)")

def test_governance():
    import s4_experiments as S
    assert S.run_kind("DEV") == "development"
    try: S.run_kind("MS"); raise AssertionError("MS run allowed without frozen protocol")
    except SystemExit: pass
    os.makedirs(S.EXP_ROOT, exist_ok=True); open(C.PROTOCOL_FILE, "w").write("primary: test\n")
    assert S.run_kind("MS") == "confirmatory"; S._ledger("MS", "confirmatory", "exp_x"); assert S.run_kind("MS") == "exploratory"; print("governance ok (first MS run = confirmatory, later = exploratory, logged)")

if __name__ == "__main__":
    for t in [test_parse, test_transform, test_partition, test_features_models, test_cv_no_leakage, test_governance]: t()
    print("\nALL TESTS PASSED")

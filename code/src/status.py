"""Phase 1 stand-in for the platform state machine: one row per image in manifests/status.csv (state, updated_at)."""
import os, datetime
import pandas as pd
import config as C

ORDER = ["INGESTED", "STANDARDISED", "QC_DONE", "PREPROCESSED", "FEATURES_DONE"]

def mark(state, image_ids):
    p = os.path.join(C.MAN, "status.csv"); now = datetime.datetime.now().isoformat(timespec="seconds")
    cur = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=["image_id", "state", "updated_at"])
    cur = cur[~cur.image_id.isin(image_ids)]
    pd.concat([cur, pd.DataFrame(dict(image_id=list(image_ids), state=state, updated_at=now))]).to_csv(p, index=False)

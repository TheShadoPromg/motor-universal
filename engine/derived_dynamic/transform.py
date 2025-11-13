# engine/derived_dynamic/transform.py
from __future__ import annotations
import os, argparse, time
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np

# add at top with other imports
from engine._utils.schema import normalize_events_df

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_OUT = REPO_ROOT / "data" / "outputs"
EVENTS_CSV = DATA_RAW / "eventos_numericos.csv"
DERIVED_PATH = DATA_OUT / "derived_dynamic.parquet"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

ESPEJO_MAP = {f"{i:02d}": f"{99-i:02d}" for i in range(100)}
COMPLEMENTO_MAP = {f"{i:02d}": f"{(100-i)%100:02d}" for i in range(100)}

def log(m): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)
def to_nn(x)->str:
    try: return f"{int(x):02d}"
    except: return "00"

def load_events()->pd.DataFrame:
    if not EVENTS_CSV.exists():
        raise SystemExit(f"[ERROR] Missing {EVENTS_CSV}.")
    df_raw = pd.read_csv(EVENTS_CSV)
    df = normalize_events_df(df_raw)
    # sanity check: min 100 numbers per date
    min_unique = df.groupby("date")["number"].nunique().min()
    if pd.notna(min_unique) and min_unique < 100:
        log(f"[WARN] Some dates have <100 numbers (min={min_unique}).")
    return df


def calc_relation(ev: pd.DataFrame, kind: str, k_vals: List[int], lags: List[int]) -> pd.DataFrame:
    tgt = ev[["date","number","pos1","pos2","pos3"]].copy()
    tgt["appears"] = ((tgt["pos1"]+tgt["pos2"]+tgt["pos3"])>0).astype(int)
    dates = sorted(ev["date"].unique())
    nums  = [f"{i:02d}" for i in range(100)]
    ev_by = {d: ev.loc[ev["date"]==d].set_index("number") for d in dates}
    tg_by = {d: tgt.loc[tgt["date"]==d].set_index("number") for d in dates}
    rows=[]
    for idx, d in enumerate(dates):
        for lag in lags:
            j = idx - lag
            if j < 0: continue
            dprev = dates[j]
            for n in nums:
                now = int(tg_by[d].loc[n,"appears"])
                if kind=="mirror":
                    src = ESPEJO_MAP[n]
                    op = int(ev_by[dprev].loc[src,["pos1","pos2","pos3"]].sum()>0)
                    rows.append(dict(date=d, number=n, relation=kind, k=np.nan, lag=lag,
                                     opportunities=op, activations=int(op and now)))
                elif kind=="complement":
                    src = COMPLEMENTO_MAP[n]
                    op = int(ev_by[dprev].loc[src,["pos1","pos2","pos3"]].sum()>0)
                    rows.append(dict(date=d, number=n, relation=kind, k=np.nan, lag=lag,
                                     opportunities=op, activations=int(op and now)))
                elif kind=="sequence":
                    n_int=int(n)
                    left=f"{(n_int-1)%100:02d}"; right=f"{(n_int+1)%100:02d}"
                    op = int(ev_by[dprev].loc[left,["pos1","pos2","pos3"]].sum()>0) or \
                         int(ev_by[dprev].loc[right,["pos1","pos2","pos3"]].sum()>0)
                    rows.append(dict(date=d, number=n, relation=kind, k=np.nan, lag=lag,
                                     opportunities=int(op>0), activations=int(op and now)))
                elif kind=="sum_mod":
                    for k in (k_vals or [1,2,5,10,50]):
                        found=0
                        for m_int in range(100):
                            m=f"{m_int:02d}"
                            if (m_int+int(k))%100==int(n):
                                if int(ev_by[dprev].loc[m,["pos1","pos2","pos3"]].sum()>0):
                                    found=1; break
                        rows.append(dict(date=d, number=n, relation=kind, k=int(k), lag=lag,
                                         opportunities=found, activations=int(found and now)))
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["date","number","relation","k","lag","opportunities","activations","consistency"])
    out["opportunities"]=out["opportunities"].fillna(0).astype(int)
    out["activations"]=out["activations"].fillna(0).astype(int)
    out["consistency"]=np.where(out["opportunities"]>0, out["activations"]/out["opportunities"], 0.0)
    out["date"]=pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["number"]=out["number"].map(to_nn); out["lag"]=out["lag"].astype(int)
    return out.sort_values(["date","number","relation","lag","k"], na_position="last").reset_index(drop=True)

def run_gx(derived_path: Path)->Dict:
    try:
        import great_expectations as gx
        ctx = gx.get_context()
        res = ctx.run_checkpoint(checkpoint_name="derived_dynamic")
        return {"status":"passed" if res["success"] else "failed"}
    except Exception as e:
        log(f"[WARN] GE skipped: {e}"); return {"status":"skipped"}

def log_mlflow(params: Dict, metrics: Dict, artifacts: list[Path]):
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("derived_dynamic")
        with mlflow.start_run(run_name="build_derived_dynamic"):
            for k,v in params.items(): mlflow.log_param(k,v)
            for k,v in metrics.items(): mlflow.log_metric(k,float(v))
            for p in artifacts:
                if Path(p).exists(): mlflow.log_artifact(str(p), artifact_path="outputs")
    except Exception as e:
        log(f"[WARN] MLflow skipped: {e}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--lags", default="1,2,3,7,14,30")
    p.add_argument("--k",    default="1,2,5,10,50")
    p.add_argument("--relations", default="mirror,sum_mod,sequence,complement")
    p.add_argument("--skip-validation", action="store_true")
    a=p.parse_args()

    lags=[int(x) for x in a.lags.split(",") if x.strip()]
    ks  =[int(x) for x in a.k.split(",") if x.strip()]
    rels=[x.strip() for x in a.relations.split(",") if x.strip()]
    log(f"Params => lags={lags} k={ks} relations={rels}")

    ev = load_events()
    frames=[calc_relation(ev, r, ks, lags) for r in rels]
    derived = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    DERIVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    derived.to_parquet(DERIVED_PATH, index=False)
    log(f"[OK] wrote {DERIVED_PATH} ({len(derived):,} rows)")

    status={"status":"skipped"} if a.skip_validation else run_gx(DERIVED_PATH)
    tot_o=int(derived["opportunities"].sum()) if not derived.empty else 0
    tot_a=int(derived["activations"].sum()) if not derived.empty else 0
    ratio=(tot_a/tot_o) if tot_o>0 else 0.0
    log_mlflow(
        params=dict(lags=",".join(map(str,lags)), k=",".join(map(str,ks)),
                    relations=",".join(rels),
                    events_path=str(EVENTS_CSV), derived_path=str(DERIVED_PATH),
                    gx_status=status["status"]),
        metrics=dict(rows=len(derived), opportunities=tot_o, activations=tot_a, ratio=ratio),
        artifacts=[DERIVED_PATH],
    )
    if status["status"]=="failed":
        log("[ERROR] GE validation failed."); return 2
    log("[DONE] Derived Dynamic Engine completed."); return 0

if __name__=="__main__":
    from pathlib import Path
    raise SystemExit(main())

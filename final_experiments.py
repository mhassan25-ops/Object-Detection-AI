"""
final_experiments.py
Final experiment run comparing Hit-Counter vs Bayesian layer.

The Bayesian counter is redesigned here with correct state machine logic:
  - Belief tracks P(object currently in scene)
  - An object is confirmed when belief first RISES above confirm_threshold
  - The counter then WAITS for belief to DROP below exit_threshold before
    it can confirm another object (prevents re-counting the same object)
  - This correctly handles persistent objects staying in scene

Parameters derived from inspect_detections.py:
  box : λ_in=1.12, p_arrive=0.029, p_nonzero=37.1%
  sack: λ_in=1.02, p_arrive=0.023, p_nonzero=18.3%
"""
import csv, math, random, time
from collections import defaultdict
from pathlib import Path
import cv2
from ultralytics import YOLO

random.seed(42)

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt; HAS_MPL=True
except ImportError:
    HAS_MPL=False

VIDEO   = r"C:\Users\pc\Downloads\sack-detection-interface-main (1)\video.mp4"
WEIGHTS = r"C:\Users\pc\Downloads\sack-detection-interface-main (1)\sack-detection-yolov8-main\runs\detect\sack_box_detect\v1\weights\best.pt"
OUT_DIR = Path(r"C:\Users\pc\Downloads\sack-detection-interface-main (1)\sack-detection-interface-main\outputs")
(OUT_DIR/"results").mkdir(parents=True, exist_ok=True)
(OUT_DIR/"figures").mkdir(parents=True, exist_ok=True)

GT={"sack":3,"box":6}; GT_TOTAL=9; STEP=5

CONFIGS=[
    {"label":"A","conf":0.30,"min_hits":3},
    {"label":"B","conf":0.40,"min_hits":3},
    {"label":"C","conf":0.55,"min_hits":5},
]

# Data-driven per-class params
BAYES={
    "box" :{"p_arrive":0.029,"p_leave":0.08,"lam_in":1.12,"lam_out":0.05,
            "confirm":0.75,"exit":0.25},
    "sack":{"p_arrive":0.023,"p_leave":0.08,"lam_in":1.02,"lam_out":0.05,
            "confirm":0.75,"exit":0.25},
}

def _pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 1e-12
    try:
        return max(math.exp(k * math.log(lam) - lam - math.lgamma(k + 1)), 1e-12)
    except (ValueError, OverflowError):
        return 1e-12

def run_bayes(raw_dets_per_frame, class_names):
    """
    Two-state HMM with confirm/exit hysteresis.
    State machine per class:
      WAITING  → belief rises above confirm → COUNTING (count+=1)
      COUNTING → belief drops below exit   → WAITING  (ready for next object)
    """
    p       = BAYES
    belief  = {c: p.get(c,{}).get("p_arrive",0.02) for c in class_names}
    counts  = {c: 0 for c in class_names}
    in_obj  = {c: False for c in class_names}  # currently tracking an object

    for fd in raw_dets_per_frame:
        for cls in class_names:
            bp  = p.get(cls,{"p_arrive":0.02,"p_leave":0.08,
                              "lam_in":1.0,"lam_out":0.05,"confirm":0.75,"exit":0.25})
            obs = fd.get(cls,0)
            # predict
            pred = belief[cls]*(1-bp["p_leave"]) + (1-belief[cls])*bp["p_arrive"]
            # update
            pin  = _pmf(obs,bp["lam_in"])
            pout = _pmf(obs,bp["lam_out"])
            den  = pin*pred + pout*(1-pred)
            post = (pin*pred/den) if den>1e-15 else pred
            belief[cls] = post

            if not in_obj[cls] and post >= bp["confirm"]:
                counts[cls] += 1
                in_obj[cls]  = True
            elif in_obj[cls] and post < bp["exit"]:
                in_obj[cls]  = False   # object left; ready for next one
    return counts

rows=[]
for cfg in CONFIGS:
    lbl=cfg["label"]
    print(f"\nConfig {lbl}: conf={cfg['conf']}  min_hits={cfg['min_hits']}")
    model=YOLO(WEIGHTS); cnames=list(model.names.values())
    hc={c:defaultdict(int) for c in cnames}
    conf_ids=defaultdict(set); hc_counts=defaultdict(int)
    total_dets=0; raw_frames=[]
    cap=cv2.VideoCapture(VIDEO); t0=time.time(); af=0; samp=0

    while True:
        ret,frame=cap.read()
        if not ret: break
        af+=1
        if af%STEP!=0: continue
        samp+=1
        try:
            res=model.track(frame,persist=True,conf=cfg["conf"],
                            tracker="botsort.yaml",device="cpu",verbose=False)
        except Exception:
            continue
        fd=defaultdict(int)
        if res and res[0].boxes is not None:
            total_dets+=len(res[0].boxes)
            for box in res[0].boxes:
                cn=model.names[int(box.cls.item())]; fd[cn]+=1
                if box.id is None: continue
                tid=int(box.id.item()); hc[cn][tid]+=1
                if hc[cn][tid]>=cfg["min_hits"] and tid not in conf_ids[cn]:
                    conf_ids[cn].add(tid); hc_counts[cn]+=1
        raw_frames.append(dict(fd))

    cap.release(); elapsed=time.time()-t0
    bc=run_bayes(raw_frames,cnames)
    noise=sum(1 for ch in hc.values() for h in ch.values() if h<cfg["min_hits"])
    fps=samp/elapsed if elapsed>0 else 0
    hct=sum(hc_counts.values()); byt=sum(bc.values())

    print(f"  HC    : sack={hc_counts.get('sack',0)}  box={hc_counts.get('box',0)}  total={hct}  |err|={abs(hct-GT_TOTAL)}")
    print(f"  Bayes : sack={bc.get('sack',0)}  box={bc.get('box',0)}  total={byt}  |err|={abs(byt-GT_TOTAL)}")
    print(f"  GT    : sack=3  box=6  total=9 | noise={noise} fps={fps:.2f}")

    rows.append({
        "config_label":lbl,"conf_threshold":cfg["conf"],"min_hits":cfg["min_hits"],
        "frame_step":STEP,"frames_sampled":samp,"frames_total":af,
        "processing_time_s":round(elapsed,1),"fps":round(fps,2),
        "raw_detections":total_dets,"noise_tracks_discarded":noise,
        "hc_sack":hc_counts.get("sack",0),"hc_box":hc_counts.get("box",0),"hc_total":hct,
        "hc_err_sack":abs(hc_counts.get("sack",0)-GT["sack"]),
        "hc_err_box":abs(hc_counts.get("box",0)-GT["box"]),
        "hc_err_total":abs(hct-GT_TOTAL),
        "bayes_sack":bc.get("sack",0),"bayes_box":bc.get("box",0),"bayes_total":byt,
        "bayes_err_sack":abs(bc.get("sack",0)-GT["sack"]),
        "bayes_err_box":abs(bc.get("box",0)-GT["box"]),
        "bayes_err_total":abs(byt-GT_TOTAL),
        "gt_sack":3,"gt_box":6,"gt_total":9,
    })

csv_p=OUT_DIR/"results"/"experiment_results.csv"
with open(csv_p,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"\nCSV: {csv_p}")
for r in rows:
    print(f"  Config {r['config_label']}: HC={r['hc_total']} err={r['hc_err_total']} | Bayes={r['bayes_total']} err={r['bayes_err_total']}")

if HAS_MPL and rows:
    lbls=[f"Config {r['config_label']}\nconf={r['conf_threshold']} mh={r['min_hits']}" for r in rows]
    x,bw=range(len(rows)),0.18

    fig,ax=plt.subplots(figsize=(9,5))
    for (vals,col,nm),off in zip([
        ([r["hc_sack"]    for r in rows],"#4a7fc1","HC — sack"),
        ([r["hc_box"]     for r in rows],"#74c0d4","HC — box"),
        ([r["bayes_sack"] for r in rows],"#d4704a","Bayes — sack"),
        ([r["bayes_box"]  for r in rows],"#d4a44a","Bayes — box"),
    ],[-1.5,-0.5,0.5,1.5]):
        ax.bar([i+off*bw for i in x],vals,bw,label=nm,color=col)
    ax.axhline(3,color="#4a7fc1",ls="--",lw=1.3,label="GT sack=3")
    ax.axhline(6,color="#74c0d4",ls=":",lw=1.3,label="GT box=6")
    ax.set_xticks(list(x)); ax.set_xticklabels(lbls,fontsize=9)
    ax.set_ylabel("Object Count"); ax.set_title("Fig 1 — Predicted Counts vs Ground Truth")
    ax.legend(fontsize=8,ncol=2); ax.set_ylim(0,12)
    plt.tight_layout(); plt.savefig(OUT_DIR/"figures"/"fig1_counts_vs_gt.png",dpi=150); plt.close()

    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(lbls,[r["hc_err_total"]    for r in rows],"o-", color="#4a7fc1",lw=2,ms=9,label="Hit-Counter |error|")
    ax.plot(lbls,[r["bayes_err_total"] for r in rows],"s--",color="#d4704a",lw=2,ms=9,label="Bayesian |error|")
    for i,r in enumerate(rows):
        ax.annotate(str(r["hc_err_total"]),   (i,r["hc_err_total"]),   xytext=(0,9),  textcoords="offset points",ha="center",fontsize=10,color="#4a7fc1")
        ax.annotate(str(r["bayes_err_total"]),(i,r["bayes_err_total"]),xytext=(0,-15),textcoords="offset points",ha="center",fontsize=10,color="#d4704a")
    ax.set_ylabel("|predicted − GT|"); ax.set_title("Fig 2 — Counting Error: Hit-Counter vs Bayesian")
    ax.legend(fontsize=10); ax.set_ylim(-1,6)
    plt.tight_layout(); plt.savefig(OUT_DIR/"figures"/"fig2_abs_error.png",dpi=150); plt.close()

    fig,ax1=plt.subplots(figsize=(8,5))
    c1,c2="#9b59b6","#27ae60"
    ax1.bar(lbls,[r["noise_tracks_discarded"] for r in rows],color=c1,alpha=0.75,label="Noise discarded")
    ax1.set_ylabel("Noise Tracks Discarded",color=c1); ax1.tick_params(axis="y",labelcolor=c1)
    ax2=ax1.twinx()
    ax2.plot(lbls,[r["fps"] for r in rows],"D-",color=c2,lw=2,ms=9,label="FPS")
    ax2.set_ylabel("Processing Speed (fps)",color=c2); ax2.tick_params(axis="y",labelcolor=c2)
    ax1.set_title("Fig 3 — Noise Rejection vs Processing Speed")
    h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax1.legend(h1+h2,l1+l2,fontsize=9)
    plt.tight_layout(); plt.savefig(OUT_DIR/"figures"/"fig3_noise_vs_fps.png",dpi=150); plt.close()
    print("Charts saved.")
print("\n✓ Done.")

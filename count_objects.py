#!/usr/bin/env python3
"""
Count distinct box/sack objects appearing anywhere in a video, using
BoT-SORT tracking with ReID to survive brief occlusions (e.g. one worker
briefly blocking another's view). Prioritizes accuracy over speed.

A track is only counted once it has been consistently detected for
`min_hits` frames, to avoid a single noisy false-positive detection
inflating the count.

Usage:
    python3 count_objects.py --source video.mp4
"""
import argparse
import csv
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

try:
    import psutil
except ImportError:
    psutil = None


class ResourceMonitor:
    """
    Samples CPU%, system RAM, GPU utilization%, and GPU memory in a background
    thread while inference runs, so the CSV log reflects actual resource load
    during the run rather than a single snapshot.
    """
    def __init__(self, interval=1.0):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self.cpu_samples = []
        self.mem_samples_mb = []
        self.gpu_util_samples = []
        self.gpu_mem_samples_mb = []

    def _sample_gpu(self):
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                # handles multi-GPU by taking the first line/GPU
                first_line = result.stdout.strip().splitlines()[0]
                util_str, mem_str = [x.strip() for x in first_line.split(",")]
                return float(util_str), float(mem_str)
        except Exception:
            pass
        return None, None

    def _run(self):
        while not self._stop_event.is_set():
            if psutil is not None:
                self.cpu_samples.append(psutil.cpu_percent(interval=None))
                self.mem_samples_mb.append(psutil.virtual_memory().used / (1024 ** 2))
            gpu_util, gpu_mem = self._sample_gpu()
            if gpu_util is not None:
                self.gpu_util_samples.append(gpu_util)
                self.gpu_mem_samples_mb.append(gpu_mem)
            time.sleep(self.interval)

    def start(self):
        if psutil is not None:
            psutil.cpu_percent(interval=None)  # prime the first reading (first call is always 0.0)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1)

    def summary(self):
        def avg(lst):
            return round(sum(lst) / len(lst), 1) if lst else 0
        def mx(lst):
            return round(max(lst), 1) if lst else 0
        return {
            "cpu_util_avg_pct": avg(self.cpu_samples),
            "cpu_util_max_pct": mx(self.cpu_samples),
            "ram_used_avg_mb": avg(self.mem_samples_mb),
            "ram_used_max_mb": mx(self.mem_samples_mb),
            "gpu_util_avg_pct": avg(self.gpu_util_samples),
            "gpu_util_max_pct": mx(self.gpu_util_samples),
            "gpu_mem_used_avg_mb": avg(self.gpu_mem_samples_mb),
            "gpu_mem_used_max_mb": mx(self.gpu_mem_samples_mb),
        }


def parse_per_class_hits(value, class_names):
    """
    Accepts either a single int ("15") applied to all classes, or a
    per-class mapping like "box=20,sack=8".
    """
    if "=" not in value:
        n = int(value)
        return {name: n for name in class_names}
    result = {}
    for pair in value.split(","):
        k, v = pair.split("=")
        result[k.strip()] = int(v.strip())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/sack_box_detect/v1/weights/best.pt")
    parser.add_argument("--source", required=True, help="Path to input video")
    parser.add_argument("--out", default="video_counting", help="Output project folder")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tracker", default="botsort_accurate.yaml",
                         help="Path to tracker config (accuracy-tuned BoT-SORT + ReID by default)")
    parser.add_argument("--min-hits", type=str, default="15",
                         help="Minimum frames a track must appear in to be confirmed. "
                              "Single value (e.g. '15') applies to all classes, or per-class "
                              "e.g. 'box=15,sack=8' if classes have different typical visibility durations.")
    parser.add_argument("--log-csv", type=str, default="counting_log.csv",
                         help="Path to CSV file to append run results to")
    args = parser.parse_args()

    model = YOLO(args.weights)
    class_names = list(model.names.values())
    min_hits_map = parse_per_class_hits(args.min_hits, class_names)

    track_hit_counts = defaultdict(lambda: defaultdict(int))
    track_class = {}

    monitor = ResourceMonitor(interval=1.0)
    monitor.start()
    start_time = time.time()

    results = model.track(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        tracker=args.tracker,
        persist=True,
        save=True,
        project=args.out,
        name="predict",
        exist_ok=True,
        stream=True,
        show_labels=True,
        show_conf=True,
        line_width=2,
        verbose=False,
    )

    frame_count = 0
    for r in results:
        frame_count += 1
        if r.boxes is None or r.boxes.id is None:
            continue

        ids = r.boxes.id.int().tolist()
        classes = r.boxes.cls.int().tolist()

        for track_id, cls_idx in zip(ids, classes):
            cls_name = model.names[cls_idx]
            track_class[track_id] = cls_name
            track_hit_counts[cls_name][track_id] += 1

    elapsed = time.time() - start_time
    monitor.stop()
    resource_stats = monitor.summary()

    final_counts = {}
    rejected_counts = {}
    avg_hits_confirmed = {}
    for cls_name, id_hits in track_hit_counts.items():
        threshold = min_hits_map.get(cls_name, 15)
        confirmed_hits = [hits for tid, hits in id_hits.items() if hits >= threshold]
        rejected_hits = [hits for tid, hits in id_hits.items() if hits < threshold]
        final_counts[cls_name] = len(confirmed_hits)
        rejected_counts[cls_name] = len(rejected_hits)
        avg_hits_confirmed[cls_name] = round(sum(confirmed_hits) / len(confirmed_hits), 1) if confirmed_hits else 0

    print("\n" + "=" * 60)
    print("UNIQUE OBJECT COUNT (whole video, whole frame)")
    print("=" * 60)
    print(f"Frames processed: {frame_count}")
    print(f"Min-hits per class: {min_hits_map}\n")
    for cls_name in sorted(final_counts):
        print(f"  {cls_name}: {final_counts[cls_name]} confirmed unique objects "
              f"({rejected_counts[cls_name]} short-lived tracks discarded as likely noise), "
              f"avg confirmed track length: {avg_hits_confirmed[cls_name]} frames")

    total_confirmed = sum(final_counts.values())
    total_discarded = sum(rejected_counts.values())
    print(f"\nTotal confirmed objects: {total_confirmed}")
    print(f"Total discarded (noise) tracks: {total_discarded}")
    print(f"Processing time: {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
    print(f"CPU util: avg {resource_stats['cpu_util_avg_pct']}% / max {resource_stats['cpu_util_max_pct']}%")
    print(f"RAM used: avg {resource_stats['ram_used_avg_mb']} MB / max {resource_stats['ram_used_max_mb']} MB")
    print(f"GPU util: avg {resource_stats['gpu_util_avg_pct']}% / max {resource_stats['gpu_util_max_pct']}%")
    print(f"GPU mem used: avg {resource_stats['gpu_mem_used_avg_mb']} MB / max {resource_stats['gpu_mem_used_max_mb']} MB")

    # Convert avi -> mp4 if needed
    predict_dir = Path(args.out) / "predict"
    for avi_path in predict_dir.glob("*.avi"):
        mp4_path = avi_path.with_suffix(".mp4")
        print(f"\nConverting {avi_path.name} -> {mp4_path.name} via ffmpeg...")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(avi_path), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-c:a", "aac", str(mp4_path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"Converted: {mp4_path}")
        else:
            print(f"ffmpeg conversion failed:\n{result.stderr[-500:]}")

    print(f"\nAnnotated video with track IDs saved under: {predict_dir}/")

    # ---- CSV logging ----
    log_path = Path(args.log_csv)
    file_exists = log_path.exists()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "video_file": Path(args.source).name,
        "frames_processed": frame_count,
        "processing_time_sec": round(elapsed, 1),
        "fps": round(frame_count / elapsed, 2) if elapsed > 0 else 0,
        "conf_threshold": args.conf,
        "min_hits_config": str(min_hits_map),
        "tracker_config": args.tracker,
        "box_confirmed": final_counts.get("box", 0),
        "box_discarded": rejected_counts.get("box", 0),
        "box_avg_confirmed_track_frames": avg_hits_confirmed.get("box", 0),
        "sack_confirmed": final_counts.get("sack", 0),
        "sack_discarded": rejected_counts.get("sack", 0),
        "sack_avg_confirmed_track_frames": avg_hits_confirmed.get("sack", 0),
        "total_confirmed": total_confirmed,
        "total_discarded": total_discarded,
        "cpu_util_avg_pct": resource_stats["cpu_util_avg_pct"],
        "cpu_util_max_pct": resource_stats["cpu_util_max_pct"],
        "ram_used_avg_mb": resource_stats["ram_used_avg_mb"],
        "ram_used_max_mb": resource_stats["ram_used_max_mb"],
        "gpu_util_avg_pct": resource_stats["gpu_util_avg_pct"],
        "gpu_util_max_pct": resource_stats["gpu_util_max_pct"],
        "gpu_mem_used_avg_mb": resource_stats["gpu_mem_used_avg_mb"],
        "gpu_mem_used_max_mb": resource_stats["gpu_mem_used_max_mb"],
        "output_dir": str(predict_dir),
    }

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nRun logged to: {log_path.resolve()}")


if __name__ == "__main__":
    main()

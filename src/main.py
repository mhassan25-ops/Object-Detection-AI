"""
main.py — CLI entry point for the standalone sack/box detection pipeline.

Runs SackDetectorAgent over a video file and prints both
hit-counter and Bayesian counts at the end.

Usage
-----
    python src/main.py --source path/to/video.mp4
    python src/main.py --source path/to/video.mp4 --conf 0.4 --min-hits 15
"""

import argparse
import random
import time

from environment import DetectionEnvironment
from agent import SackDetectorAgent

WEIGHTS = r"C:\Users\pc\Downloads\sack-detection-interface-main (1)\sack-detection-yolov8-main\runs\detect\sack_box_detect\v1\weights\best.pt"


def main():
    parser = argparse.ArgumentParser(description="Sack/Box object counter")
    parser.add_argument("--source",   required=True, help="Path to video file")
    parser.add_argument("--weights",  default=WEIGHTS)
    parser.add_argument("--conf",     type=float, default=0.40)
    parser.add_argument("--min-hits", type=int,   default=15)
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    env   = DetectionEnvironment(args.source, seed=args.seed)
    agent = SackDetectorAgent(
        weights_path=args.weights,
        conf=args.conf,
        min_hits=args.min_hits,
        device=args.device,
        seed=args.seed,
    )

    print(f"Environment : {env}")
    print(f"Agent       : {agent}")
    print(f"Seed        : {args.seed}\n")

    t0 = time.time()
    frame_count = 0

    with env:
        for frame, idx, done in env:
            agent.perceive(frame)
            frame_count += 1
            if idx % 300 == 0:
                hc = agent.hit_counter_counts
                by = agent.bayesian_counts
                print(f"  Frame {idx:4d} | HC: {dict(hc)} | Bayes: {dict(by)}")

    elapsed = time.time() - t0

    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)
    print(f"Frames processed : {frame_count}")
    print(f"Time             : {elapsed:.1f}s  ({frame_count/elapsed:.1f} fps)")
    print(f"\nHit-Counter method  : {agent.hit_counter_counts}")
    print(f"Bayesian layer      : {agent.bayesian_counts}")
    print("=" * 55)


if __name__ == "__main__":
    main()

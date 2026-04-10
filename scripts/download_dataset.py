"""
Download a Roboflow drone detection dataset for fine-tuning.

Requires a Roboflow API key (free tier). Set it via environment variable:
    export ROBOFLOW_API_KEY="your_key_here"

Or pass it as the first argument:
    python3 download_dataset.py YOUR_KEY

Downloads to: datasets/drones_new/
"""

import os
import sys

from roboflow import Roboflow


def main():
    # Get API key from arg or environment
    api_key = None
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        print("ERROR: Provide your Roboflow API key.")
        print("  Option 1:  python3 download_dataset.py YOUR_KEY")
        print("  Option 2:  export ROBOFLOW_API_KEY=YOUR_KEY && python3 download_dataset.py")
        print("\nGet a free key at: https://app.roboflow.com/settings/api")
        sys.exit(1)

    print("[DL] Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)

    # The "drones_new" dataset by tracker-qjlj1
    # Has images of people holding drones — ideal for our demo scenario
    print("[DL] Fetching project: tracker-qjlj1/drones_new")
    project = rf.workspace("tracker-qjlj1").project("drones_new")

    # Get the latest version
    versions = project.versions()
    if not versions:
        print("ERROR: No versions available for this project")
        sys.exit(1)

    latest = versions[-1]
    print(f"[DL] Using version {latest.version}")

    print("[DL] Downloading in YOLOv8 format...")
    dataset = latest.download("yolov8", location="datasets/drones_new")

    print(f"\n[DL] Dataset ready at: {dataset.location}")
    print("[DL] Ready for fine-tuning. Run: python3 train.py")


if __name__ == "__main__":
    main()

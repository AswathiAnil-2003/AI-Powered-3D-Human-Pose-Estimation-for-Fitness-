import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pose_module.pose_estimator import PoseEstimator


input_folder = "data/examples/trimmed"
output_folder = "data/processed/"
output_norm_folder = os.path.join(output_folder, "normalized")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_norm_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        input_path = os.path.join(input_folder, filename)
        base_name = filename.replace(".mp4", "")
        output_csv = os.path.join(output_folder, base_name + "_keypoints.csv")
        output_norm_csv = os.path.join(output_norm_folder, base_name + "_normalized.csv")

        # ✅ Skip if both files already exist
        if os.path.exists(output_csv) and os.path.exists(output_norm_csv):
            print(f"Skipping already processed: {filename}")
            continue

        print(f"Processing: {filename}")
        estimator = PoseEstimator(
            video_path=input_path,
            output_path=output_csv,
            output_norm_path=output_norm_csv,
            draw_skeleton=False
        )
        estimator.extract_keypoints()


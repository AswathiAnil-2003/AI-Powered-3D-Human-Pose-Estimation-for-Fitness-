from pose_module.pose_estimator import PoseEstimator

def main():
    video_path = "data/examples/Sarthak_Trikonasana.mp4"
    output_raw = "data/processed/Sarthak_Trikonasana_raw.csv"
    output_norm = "data/processed/normalized/Sarthak_Trikonasana_norm.csv"

    estimator = PoseEstimator(video_path, output_raw, output_norm, draw_skeleton=True)
    estimator.extract_keypoints()

if __name__ == "__main__":
    main()

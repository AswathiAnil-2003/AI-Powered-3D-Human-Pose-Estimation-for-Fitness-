import cv2
import mediapipe as mp
import csv
import os
from pose_module.utils import normalize_keypoints

class PoseEstimator:
    def __init__(self, video_path, output_path=None, output_norm_path=None, draw_skeleton=True):
        self.video_path = video_path
        self.output_path = output_path  # CSV file path to save raw keypoints
        self.output_norm_path = output_norm_path  # CSV path to save normalized keypoints
        self.draw_skeleton = draw_skeleton

        # Initialize MediaPipe pose solution
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      model_complexity=1,
                                      enable_segmentation=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_keypoints(self):
        print(f"Opening video file: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {self.video_path}")
            return

        frame_num = 0

        # Prepare CSV writers if paths are given
        raw_writer, norm_writer = None, None
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            raw_csv = open(self.output_path, mode='w', newline='')
            raw_writer = csv.writer(raw_csv)
            raw_header = ['frame'] + [f'{c}{i}' for i in range(33) for c in ['x', 'y', 'z']]
            raw_writer.writerow(raw_header)

        if self.output_norm_path:
            os.makedirs(os.path.dirname(self.output_norm_path), exist_ok=True)
            norm_csv = open(self.output_norm_path, mode='w', newline='')
            norm_writer = csv.writer(norm_csv)
            norm_header = ['frame'] + [f'norm_{c}{i}' for i in range(33) for c in ['x', 'y', 'z']]
            norm_writer.writerow(norm_header)

        print("Starting frame processing...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video or failed to read frame.")
                break

            print(f"Processing frame {frame_num}")
            # Convert frame color for MediaPipe processing
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                # Extract raw keypoints as x,y,z for each landmark in order
                raw_kp = []
                for lm in results.pose_landmarks.landmark:
                    raw_kp.extend([lm.x, lm.y, lm.z])

                # Draw skeleton if requested
                if self.draw_skeleton:
                    self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                # Write raw keypoints
                if raw_writer:
                    raw_writer.writerow([frame_num] + raw_kp)

                # Normalize and write normalized keypoints
                if norm_writer:
                    norm_kp = normalize_keypoints(raw_kp)
                    norm_writer.writerow([frame_num] + norm_kp)
            else:
                # No landmarks detected: write zeros
                zeros = [0] * (33 * 3)
                if raw_writer:
                    raw_writer.writerow([frame_num] + zeros)
                if norm_writer:
                    norm_writer.writerow([frame_num] + zeros)

            # Show frame with skeleton
            cv2.imshow('Pose Estimation', frame)
            if cv2.waitKey(30) & 0xFF == 27:  # ESC key to exit early
                print("ESC pressed. Exiting early.")
                break

            frame_num += 1

        cap.release()
        cv2.destroyAllWindows()
        if raw_writer:
            raw_csv.close()
        if norm_writer:
            norm_csv.close()
        print("Video processing completed.")

if __name__ == "__main__":
    # Example usage
    video_file = "data/examples/Santosh_Bhujangasana.mp4"  # Adjust relative path if needed
    output_raw_csv = "data/processed/Santosh_Bhujangasana.csv"
    output_norm_csv = "data/processed/normalized/Santosh_Bhujangasana.csv"
    estimator = PoseEstimator(video_file, output_raw_csv, output_norm_csv, draw_skeleton=True)
    estimator.extract_keypoints()

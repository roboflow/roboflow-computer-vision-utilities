import os
import cv2
import json
import argparse
from roboflow import Roboflow

def process_frame(frame_path, model):
    predictions = model.predict(frame_path).json()["predictions"]
    
    return predictions

def main(
    video_path: str,
    api_key: str,
    project_name: str,
    model_version: int,
    confidence: int,
    overlap: int,
    active_learning: bool,
    raw_data_location: str = ".",
    raw_data_extension: str = ".jpg",
    upload_destination: str = "",
    conditionals: dict = "{}",
    fps: int = 1,
):
    rf = Roboflow(api_key=api_key)
    inference_project = rf.workspace().project(project_name)
    model = inference_project.version(model_version).model

    model.confidence = confidence
    model.overlap = overlap

    video = cv2.VideoCapture(video_path)
    frame_number = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = int(video.get(cv2.CAP_PROP_FPS) // fps)
    sampled_frames = []

    while frame_number < total_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()

        if not ret:
            break

        if active_learning:
            active_learning_frames = os.path.join(raw_data_location + "/sampled_frames")
            if os.path.exists(active_learning_frames) is False:
                os.mkdir(active_learning_frames)
        
        frame_path = os.path.abspath(active_learning_frames + f"/frame_{frame_number:04d}{raw_data_extension}")
        sampled_frames.append(frame_path)
        print(frame_path)
        if os.path.exists(frame_path) is False:
            cv2.imwrite(frame_path, frame)

        predictions = process_frame(frame_path, model)
        frame_number += skip_frames

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    if active_learning:
        workspace = rf.workspace()

        if "minimum_size_requirement" in conditionals:
            conditionals["minimum_size_requirement"] = float(
                conditionals["minimum_size_requirement"]) if not isinstance(
                conditionals["minimum_size_requirement"], float) else conditionals["minimum_size_requirement"]
        if "maximum_size_requirement" in conditionals:
            conditionals["maximum_size_requirement"] = float(
                conditionals["maximum_size_requirement"]) if not isinstance(
                conditionals["maximum_size_requirement"], float) else conditionals["maximum_size_requirement"]
        for i in range(len(sampled_frames)):
            workspace.active_learning(
                raw_data_location=sampled_frames[i],
                raw_data_extension=raw_data_extension,
                inference_endpoint=[project_name, model_version],
                upload_destination=upload_destination,
                conditionals=conditionals
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video object detection with Roboflow and optional Active Learning.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("--api_key", type=str, help="Your Roboflow API key.")
    parser.add_argument("--project_name", type=str, help="Your Roboflow project name.")
    parser.add_argument("--model_version", type=int, help="Model version number.")
    parser.add_argument("--confidence", default=40, type=int, help="Confidence threshold.")
    parser.add_argument("--overlap", default=30, type=int, help="Overlap threshold.")
    parser.add_argument("--active_learning", default=True, action="store_true", help="Enable Active Learning.")
    parser.add_argument("--raw_data_location", type=str, default=f"{os.curdir}", help="Location to save frames for Active Learning.")
    parser.add_argument("--raw_data_extension", type=str, default=".jpg", help="Image extension for saved frames.")
    parser.add_argument("--upload_destination", type=str, help="Upload destination (model ID). e.g) project_name (str)")
    parser.add_argument("--conditionals", type=str, default="{}", help="Conditionals for Active Learning (JSON string).")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to sample from the video.")

    args = parser.parse_args()

    conditionals = json.loads(args.conditionals)

    main(
        args.video_path,
        args.api_key,
        args.project_name,
        args.model_version,
        args.confidence,
        args.overlap,
        args.active_learning,
        args.raw_data_location,
        args.raw_data_extension,
        args.upload_destination,
        conditionals,
        args.fps,
    )
## Example below for how to run the file (remove the comment
## from each line below, prior to copy/pasting to your Terminal)
# python3 video.py --video_path="/test-1920x1080.mp4"
# --api_key="PRIVATE_API_KEY" \
# --project_name="face-detection-mik1i" \
# --model_version=18 \
# --raw_data_location="/active_learning_infer" \
# --upload_destination="face-detection-mik1i" \
# --conditionals='{"required_objects_count": 1, \
# "required_class_count": 1, "target_classes": ["face"], \
# "confidence_interval": [1,75], "minimum_size_requirement": \
# "float('-inf')", "maximum_size_requirement": "float('inf')"}'
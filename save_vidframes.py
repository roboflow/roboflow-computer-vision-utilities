import os
import cv2
import argparse

def main(
    video_path: str,
    active_learning: bool,
    raw_data_location: str = ".",
    raw_data_extension: str = ".jpg",
    fps: int = 1,
):
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

        frame_number += skip_frames

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sampling videos by a specified frames per second of video.")
    parser.add_argument("--video_path", type=str, help="Path to the video file.")
    parser.add_argument("--active_learning", type=str, help="Path to the video file.")
    parser.add_argument("--raw_data_location", type=str, default=f"{os.curdir}", help="Location to save frames.")
    parser.add_argument("--raw_data_extension", type=str, default=".jpg", help="Image extension for saved frames.")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to sample from the video.")

    args = parser.parse_args()

    main(
        args.video_path,
        args.active_learning,
        args.raw_data_location,
        args.raw_data_extension,
        args.fps,
    )
## Example below for how to run the file (remove the comment from each line below, prior to copy/paste to your Terminal)
# python3 save_vidframes.py --video_path="/path/to/video.mp4" \
# --active_learning=True \
# --raw_data_location="/path/to/save/video/frames" \
# --raw_data_extension=".jpg" \
# --fps=5

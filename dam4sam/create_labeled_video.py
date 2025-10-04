import cv2
import os
import glob
import argparse
import sys
import random

def draw_labels_on_video(video_path, labels_folder):
    """
    Reads a video and a folder of labels, and creates a new video with
    bounding boxes and IDs drawn on each frame.
    """
    # --- 1. Validate Inputs ---
    print("🔎 Validating inputs...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    label_files = sorted(glob.glob(os.path.join(labels_folder, '*.txt')))
    num_label_files = len(label_files)

    # --- FIX STARTS HERE: Changed the strict check to a warning ---
    if total_frames != num_label_files:
        print(f"⚠️ Warning: Frame count mismatch detected.")
        print(f"   - Video has {total_frames} frames.")
        print(f"   - Labels folder has {num_label_files} files.")
        print("   - Proceeding to generate video. Frames without a label file will be left blank.")
    else:
        print(f"✅ Validation successful. Video and labels both have {total_frames} frames.")
    # --- FIX ENDS HERE ---

    # --- 2. Setup Output Video ---
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_name, video_ext = os.path.splitext(os.path.basename(video_path))
    output_path = os.path.join(os.path.dirname(video_path), f"{video_name}_labeled.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"🎥 Will save labeled video to: {output_path}")

    # --- 3. Process and Draw Frame by Frame ---
    frame_idx = 0
    colors = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        label_file_path = os.path.join(labels_folder, f"{frame_idx:08d}.txt")

        # This logic correctly handles missing files on a per-frame basis
        if os.path.exists(label_file_path):
            with open(label_file_path, 'r') as f:
                for line in f:
                    try:
                        parts = line.strip().split()
                        norm_x1, norm_y1, norm_x2, norm_y2 = map(float, parts[:4])
                        obj_id = int(parts[4])

                        # Rescale normalized coordinates
                        pixel_x1 = int(norm_x1 * width)
                        pixel_y1 = int(norm_y1 * height)
                        pixel_x2 = int(norm_x2 * width)
                        pixel_y2 = int(norm_y2 * height)

                        if obj_id not in colors:
                            colors[obj_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                        color = colors[obj_id]

                        pt1 = (pixel_x1, pixel_y1)
                        pt2 = (pixel_x2, pixel_y2)
                        cv2.rectangle(frame, pt1, pt2, color, 2)

                        label = f"ID: {obj_id}"
                        text_pos = (pt1[0], pt1[1] - 10)
                        cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    except (ValueError, IndexError):
                        print(f"\n⚠️ Warning: Skipping malformed line in {os.path.basename(label_file_path)}: '{line.strip()}'")

        out.write(frame)

        progress = (frame_idx + 1) / total_frames
        sys.stdout.write(f"\r🎨 Processing: [{'=' * int(20 * progress):<20}] {int(100 * progress)}% - Frame {frame_idx+1}/{total_frames}")
        sys.stdout.flush()

        frame_idx += 1

    # --- 4. Cleanup ---
    print("\n✨ Processing complete!")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description="A script to draw bounding box labels from text files onto a video.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("video_path",
                        help="Path to the input video file.")

    parser.add_argument("labels_folder",
                        help="Path to the folder containing the corresponding .txt label files.")

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found at '{args.video_path}'")
        return

    if not os.path.isdir(args.labels_folder):
        print(f"❌ Error: Labels folder not found at '{args.labels_folder}'")
        return

    draw_labels_on_video(args.video_path, args.labels_folder)

if __name__ == "__main__":
    main()
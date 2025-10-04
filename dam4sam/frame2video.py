import os
import glob
import argparse
import cv2
from tqdm import tqdm

def create_video_from_frames(frames_dir, file_extension, ouput_file, fps):
    """
    Creates a video from a sequence of image frames.

    Args:
        frames_dir (str): Path to the directory containing the image frames.
        file_extension (str): The file extension of the frames (e.g., 'jpg', 'png').
        ouput_file (str): The full path for the output video file.
        fps (int): The frames per second for the output video.
    """
    # 1. Get and sort the list of frame files
    image_files = sorted(glob.glob(os.path.join(frames_dir, f'*.{file_extension}')))

    if not image_files:
        print(f'Error: No frames with extension ".{file_extension}" found in "{frames_dir}"')
        return

    # 2. Read the first frame to determine video dimensions
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"Error: Could not read the first frame: {image_files[0]}")
        return
    height, width, layers = first_frame.shape
    frame_size = (width, height)

    # 3. Initialize the VideoWriter object
    # Using 'mp4v' codec which is widely supported for .mp4 files.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(ouput_file, fourcc, fps, frame_size)
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for path '{ouput_file}'")
        return

    # --- Informative Printouts ---
    print("-" * 50)
    print("Starting Video Creation")
    print(f"  -> Input Directory:     {os.path.abspath(frames_dir)}")
    print(f"  -> Output Video Path:   {os.path.abspath(ouput_file)}")
    print(f"  -> Frame Count:         {len(image_files)}")
    print(f"  -> Video Dimensions:    {width}x{height}")
    print(f"  -> Frames Per Second:   {fps}")
    print("-" * 50)

    # 4. Loop through all frames and write them to the video
    for filename in tqdm(image_files, desc="Compiling Video", unit="frame"):
        img = cv2.imread(filename)
        out.write(img)

    # 5. Release the VideoWriter and clean up
    out.release()
    print("\nVideo creation complete.")
    print(f"Video saved successfully to: {ouput_file}")

def main():
    parser = argparse.ArgumentParser(description='Create a video from a directory of sequential frames.')
    parser.add_argument('--dir', type=str, required=True, 
                        help='Path to the directory containing the frames (e.g., "output/annotated_frames").')
    parser.add_argument('--ext', type=str, default='jpg', 
                        help='Image file extension (e.g., jpg, png). Default: jpg.')
    parser.add_argument('--output', type=str, default='output.mp4', 
                        help='Name and path for the output video file. Default: output.mp4.')
    parser.add_argument('--fps', type=int, default=30, 
                        help='Frames per second for the output video. Default: 30.')
    
    args = parser.parse_args()

    create_video_from_frames(args.dir, args.ext, args.output, args.fps)

if __name__ == "__main__":
    main()

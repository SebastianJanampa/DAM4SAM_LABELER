import cv2
import os, shutil
import argparse
from tqdm import tqdm
import multiprocessing

def process_frame_task(args):
    """
    Worker function: Processes a single frame task (compressing and saving).
    This function is designed to be called by a multiprocessing pool.
    """
    frame, output_path, max_size_kb = args
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    
    while True:
        result, buffer = cv2.imencode('.jpg', frame, encode_param)
        if not result:
            return False # Failed to encode
            
        file_size_kb = len(buffer) / 1024
        if file_size_kb <= max_size_kb or encode_param[1] <= 10:
            break
        # Decrease quality for the next attempt
        encode_param[1] -= 5

    with open(output_path, 'wb') as f:
        f.write(buffer)
    return True

def extract_and_resize_frames(video_path, output_folder, frame_skip, max_size_kb, num_workers):
    """
    Extracts frames from a video in parallel using a multiprocessing pool.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    # Create the output directory based on the video's name
    video_filename = os.path.basename(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- Step 1: Scan video and collect all tasks to be processed ---
    tasks = []
    print("Scanning video to prepare frame data...")
    for frame_num in tqdm(range(total_frames), desc="Scanning Video"):
        success, frame = cap.read()
        if not success:
            break

        # Always keep track of the latest frame read from the video
        last_frame_data = (frame, frame_num)
        
        # Check if this frame should be processed based on the skip value
        if frame_num % (frame_skip + 1) == 0:
            # Use the actual frame number for the filename, zero-padded
            frame_filename = f"{frame_num:08d}.jpg"
            output_path = os.path.join(output_folder, frame_filename)
            # Add the frame data, output path, and max size to our list of tasks
            tasks.append((frame, output_path, max_size_kb))
    
    cap.release()
    
    if last_frame_data:
        last_frame, last_frame_num = last_frame_data
        # Check if the last frame's index would have been skipped
        if last_frame_num % (frame_skip + 1) != 0:
            print(f"Info: Unconditionally adding last frame ({last_frame_num}) to the processing queue.")
            frame_filename = f"{last_frame_num:08d}.jpg"
            output_path = os.path.join(output_folder, frame_filename)
            tasks.append((last_frame, output_path, max_size_kb))

    if not tasks:
        print("No frames were selected for processing.")
        return

    # --- Step 2: Process the collected frames in parallel ---
    print(f"\nProcessing {len(tasks)} frames using {num_workers} worker(s)...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use tqdm to show a progress bar for the parallel processing
        list(tqdm(pool.imap_unordered(process_frame_task, tasks), total=len(tasks), desc="Saving Frames"))

    print("\nProcessing complete.")
    print(f"Total frames saved: {len(tasks)}")


if __name__ == '__main__':
    # Determine a good default for the number of workers (usually all but one CPU core)
    default_workers = max(1, os.cpu_count() - 1)

    parser = argparse.ArgumentParser(
        description="Extracts, compresses, and saves frames from a video in parallel."
    )
    parser.add_argument(
        "video_path", type=str, help="The full path to the video file to process."
    )
    parser.add_argument(
        "-s", "--skip", type=int, default=4,
        help="Optional: Number of frames to skip between extractions. Default: 4"
    )
    parser.add_argument(
        "-kb", "--max_size_kb", type=int, default=300,
        help="Optional: The maximum file size in KB for each saved frame. Default: 300"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=default_workers,
        help=f"Optional: Number of CPU cores to use for processing. Default: {default_workers}"
    )

    args = parser.parse_args()
    
    # Call the main function with all the parsed arguments
    extract_and_resize_frames(args.video_path, args.skip, args.max_size_kb, args.workers)

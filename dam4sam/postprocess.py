import os
import glob
import shutil
import argparse
from collections import defaultdict
import numpy as np

def create_merge_map(merge_groups):
    """Creates a mapping from old IDs to a new primary ID from argparser groups."""
    if not merge_groups:
        return {}
        
    merge_map = {}
    print("🤝 Applying the following ID merge rules:")
    for group in merge_groups:
        if len(group) < 2:
            print(f"⚠️ Warning: --merge group '{group}' has less than two IDs. Skipping.")
            continue
            
        ids = sorted(group)
        primary_id = ids[0]
        for other_id in ids[1:]:
            merge_map[other_id] = primary_id
        print(f"  - Merging IDs {ids[1:]} into primary ID -> {primary_id}")
        
    return merge_map

def load_and_process_data(folder_path, merge_map):
    """Loads, merges, and interpolates the tracking data."""
    
    # Step 1: Load all data from files
    print("\n🔄 Loading data from files...")
    all_data = defaultdict(dict)
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    if not txt_files:
        print(f"❌ No .txt files found in {folder_path}. Exiting.")
        return None

    for f_path in txt_files:
        try:
            frame_idx = int(os.path.basename(f_path).split('.')[0])
            with open(f_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        obj_id = int(parts[4])
                        # Apply merging rule if needed
                        final_id = merge_map.get(obj_id, obj_id)
                        bbox = np.array([float(p) for p in parts[:4]])
                        all_data[frame_idx][final_id] = bbox
        except (ValueError, IndexError) as e:
            print(f"⚠️ Warning: Could not parse file {os.path.basename(f_path)}. Skipping. Error: {e}")
    
    print(f"✅ Loaded data for {len(all_data)} frames.")

    # Step 2: Interpolate missing frames
    print("🔄 Interpolating missing frames...")
    sorted_frames = sorted(all_data.keys())
    
    for i in range(len(sorted_frames) - 1):
        start_frame_idx = sorted_frames[i]
        end_frame_idx = sorted_frames[i+1]
        
        if end_frame_idx - start_frame_idx > 1:
            start_objects = all_data[start_frame_idx]
            end_objects = all_data[end_frame_idx]
            
            common_object_ids = set(start_objects.keys()) & set(end_objects.keys())
            
            for obj_id in common_object_ids:
                bbox_start = start_objects[obj_id]
                bbox_end = end_objects[obj_id]
                total_steps = end_frame_idx - start_frame_idx
                
                for step in range(1, total_steps):
                    inter_frame_idx = start_frame_idx + step
                    ratio = step / total_steps
                    inter_bbox = bbox_start + (bbox_end - bbox_start) * ratio
                    all_data[inter_frame_idx][obj_id] = inter_bbox

    print("✅ Interpolation complete.")
    return all_data

def write_output_files(folder_path, all_data):
    """Writes the completed data to a new post-processed folder, ensuring all frames up to the max are created."""
    output_folder = f"{folder_path}_postprocessed"
    
    if os.path.exists(output_folder):
        print(f"⚠️ Output folder '{output_folder}' already exists. Removing it.")
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    print(f"\n💾 Saving results to '{output_folder}'...")

    # --- FIX STARTS HERE ---
    if not all_data:
        print("No data to write.")
        return

    # Determine the full range of frames to create
    max_frame_idx = 0
    # Find the highest frame number from the original text files
    original_files = glob.glob(os.path.join(folder_path, '*.txt'))
    if original_files:
        max_frame_idx = max(int(os.path.basename(f).split('.')[0]) for f in original_files)

    print(f"Highest frame number found is {max_frame_idx}. Will generate {max_frame_idx + 1} files.")

    # Loop from frame 0 to the maximum frame index found
    for frame_idx in range(max_frame_idx + 1):
        file_path = os.path.join(output_folder, f"{frame_idx:08d}.txt")
        with open(file_path, 'w') as f:
            # Check if there is data for the current frame before writing
            if frame_idx in all_data:
                for obj_id, bbox in all_data[frame_idx].items():
                    bbox_str = " ".join([f"{coord:.8f}" for coord in bbox])
                    f.write(f"{bbox_str} {obj_id}\n")
    # --- FIX ENDS HERE ---
    
    print("✨ All done!")


def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Tracker Data Post-Processing Tool: Merges object IDs and interpolates missing frames.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("folder_path", 
                        help="Path to the folder containing your .txt tracking files.")
    
    parser.add_argument("--merge", 
                        nargs='+', 
                        type=int, 
                        action='append', 
                        metavar='ID',
                        help="Specify a group of object IDs to merge into the lowest ID.\n"
                             "This flag can be used multiple times for different groups.\n"
                             "Example: --merge 2 5 8 --merge 10 12")

    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"❌ Error: The provided folder path does not exist: {args.folder_path}")
        return

    # 1. Create the merge map from command-line arguments
    merge_map = create_merge_map(args.merge)
    
    # 2. Load, merge, and interpolate
    final_data = load_and_process_data(args.folder_path, merge_map)
    
    # 3. Write results to a new folder
    if final_data:
        write_output_files(args.folder_path, final_data)

if __name__ == "__main__":
    main()
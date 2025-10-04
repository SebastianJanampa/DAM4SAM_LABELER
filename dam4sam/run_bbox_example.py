import os
import glob
import argparse
import shutil

import cv2
import numpy as np

from PIL import Image
from tqdm import tqdm

from dam4sam_tracker import DAM4SAMTracker
from utils.multi_box_selector import MultiFrameBoxSelector
from utils.visualization_utils import overlay_mask
import pprint # Added for pretty printing the new output format


# Attempt to import torch for VRAM monitoring
try:
    import torch
    if torch.cuda.is_available():
        VRAM_SUPPORT = True
    else:
        print("Warning: PyTorch CUDA is not available. VRAM monitoring disabled.")
        VRAM_SUPPORT = False
except ImportError:
    print("Warning: PyTorch not found. VRAM monitoring disabled.")
    VRAM_SUPPORT = False


# Custom color map for visualization
CUSTOM_COLOR_MAP_HEX = [
    "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#d2f53c", "#fabebe", "#008080", "#e6beff",
    "#aa6e28", "#fffac8", "#800000", "#aaffc3",
]

# --- MODIFICATION START ---
# Convert hex color strings to BGR tuples for OpenCV
def hex_to_bgr(hex_code):
    """Converts a hex color code to a (B, G, R) tuple."""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (4, 2, 0))

COLORS_BGR = [hex_to_bgr(h) for h in CUSTOM_COLOR_MAP_HEX]

def draw_text_with_background(scene, text, pos, bg_color, font_scale=0.6, text_color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, thickness=1):
    """Draws text with a filled background rectangle using OpenCV."""
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    
    # Calculate rectangle coordinates
    rect_x1, rect_y1 = pos[0], pos[1]
    rect_x2, rect_y2 = pos[0] + text_w + 4, pos[1] - text_h - 6
    
    # Draw the background rectangle and the text
    cv2.rectangle(scene, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, cv2.FILLED)
    cv2.putText(scene, text, (pos[0] + 2, pos[1] - 4), font, font_scale, text_color, thickness, cv2.LINE_AA)
# --- MODIFICATION END ---


def run_sequence(dir_path, file_extension, output_dir, model_name, use_fps16):
    """
    Runs the DAM4SAM tracker on a sequence of frames with enhanced visualization.
    """
    frames_dir = sorted(glob.glob(os.path.join(dir_path, f'*.{file_extension}')))
    
    if not frames_dir:
        print(f'Error: No frames with extension ".{file_extension}" found in the directory: {dir_path}')
        return

    print("-" * 50)
    print("Starting DAM4SAM Tracker")
    print(f"  -> Model:             {model_name}")
    print(f"  -> Input Directory:   {os.path.abspath(dir_path)}")
    print(f"  -> Output Directory:  {os.path.abspath(output_dir) if output_dir else 'None (visualization mode)'}")
    print(f"  -> Frames to process: {len(frames_dir)}")
    print("-" * 50)

    # Multi-box selection
    multi_box_selector = MultiFrameBoxSelector()
    subjects_info = multi_box_selector.select_boxes(frames_dir)
        
    if not subjects_info:
        print('Error: No initialization boxes were given')
        return
            
    print(f'Using boxes for tracking:')
    pprint.pprint(subjects_info)
        
    # Create tracker instance
    tracker = DAM4SAMTracker(model_name)

    if output_dir:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        annotated_frames_dir = os.path.join(output_dir, 'annotated_frames')
        labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(annotated_frames_dir)
        os.makedirs(labels_dir)
    else: # Setup for live visualization
        window_name = 'DAM4SAM Tracking'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        wait_ = 0

    pil_images = []
    for i, frame_path in enumerate(frames_dir): 
        img = Image.open(frame_path).convert("RGB")
        pil_images.append(img)

    print('Segmenting frames...')
    with tqdm(total=len(frames_dir), desc="Tracking", unit="frame") as pbar:
        for i, frame_path in enumerate(frames_dir):
            img = Image.open(frame_path).convert("RGB")
            img_vis = np.array(img)
            img_h, img_w, _ = img_vis.shape

            if i == 0:
                outputs = tracker.initialize(pil_images, None, subjects_info=subjects_info)
            else:
                outputs = tracker.track(img, use_fps16=use_fps16)
            # print(type(outputs))
            
            lines_for_label_file = []

            for obj_id, mask in outputs.items():
                # --- MODIFICATION: Get color from our BGR list ---
                color_bgr = COLORS_BGR[obj_id % len(COLORS_BGR)]

                mask_coords = np.column_stack(np.where(mask > 0)[::-1])
                if mask_coords.size == 0:
                    continue 

                x, y, w, h = cv2.boundingRect(mask_coords)
                
                overlay_mask(img_vis, mask, color_bgr, line_width=1, alpha=0.5)
                cv2.rectangle(img_vis, (x, y), (x + w, y + h), color_bgr, 2)
                
                # --- MODIFICATION: Use new custom drawing function ---
                draw_text_with_background(
                    scene=img_vis,
                    text=f"ID: {obj_id}",
                    pos=(x, y - 5),
                    bg_color=color_bgr
                )

                if output_dir:
                    norm_x1, norm_y1 = x / img_w, y / img_h
                    norm_x2, norm_y2 = (x + w) / img_w, (y + h) / img_h
                    
                    bbox_str = f"{norm_x1:.8f} {norm_y1:.8f} {norm_x2:.8f} {norm_y2:.8f}"
                    line = f"{bbox_str} {obj_id}"
                    lines_for_label_file.append(line)

            if output_dir:
                base_name = os.path.splitext(os.path.basename(frame_path))[0]
                annotated_frame_path = os.path.join(annotated_frames_dir, f'{base_name}.jpg')
                label_path = os.path.join(labels_dir, f'{base_name}.txt')

                cv2.imwrite(annotated_frame_path, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

                with open(label_path, 'w') as f:
                    f.write('\n'.join(lines_for_label_file))
            else: 
                cv2.imshow(window_name, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
                key_ = cv2.waitKey(wait_)
                
                if key_ == 27: break 
                elif key_ == 32: wait_ = 1 - wait_ 
            
            if VRAM_SUPPORT:
                vram_used_mb = torch.cuda.memory_allocated() / (1024**2)
                pbar.set_postfix_str(f"VRAM: {vram_used_mb:.2f} MB")
            
            pbar.update(1)

    if not output_dir:
        cv2.destroyAllWindows()
        
    print('Segmentation: Done.')


def main():
    parser = argparse.ArgumentParser(description='Run DAM4SAM tracker on a sequence of frames.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory with frames.')
    parser.add_argument('--ext', type=str, default='jpg', help='Image file extension (e.g., jpg, png).')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to the output directory to save masks.')
    parser.add_argument('--model_name', type=str, default='sam21pp-L', help='Model used for tracking')
    parser.add_argument('--use_fp16', type=bool, default=True, help='Model used for tracking')
    
    args = parser.parse_args()

    run_sequence(args.input_dir, args.ext, args.output_dir, args.model_name, args.use_fp16)

if __name__ == "__main__":
    main()

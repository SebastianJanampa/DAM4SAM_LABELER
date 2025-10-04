import os
import glob
import pprint
import logging
import argparse

import cv2
import numpy as np

from PIL import Image
from tqdm import tqdm

from dam4sam.global_vars import MODELS, COLORS_BGR
from dam4sam.video2frame import extract_and_resize_frames
from dam4sam.frame2video import create_video_from_frames
from dam4sam.dam4sam_tracker import DAM4SAMTracker

from dam4sam.utils.multi_box_selector import MultiFrameBoxSelector
from dam4sam.utils.visualization_utils import overlay_mask
from dam4sam.utils.download_checkpoints import download_checkpoint


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


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


class DAM4SAM(object):
    def __init__(self, model):
        super().__init__()
        self.load_model(model) 

    def load_model(self, model):
        self.model_name = model

        file_path = MODELS[model]['file_path']
        if os.path.isfile(file_path):
            logger.info(f"✅ The file '{file_path}' exists.")
        else:
            logger.info(f"❌ The file '{file_path}' does not exist.")
            logger.info(f"The file '{file_path}' does not exist.")
            logger.info(f"Downloading checkpoint for '{model}'.")
            os.makedirs('./checkpoints', exist_ok=True)
            download_checkpoint(model)
        self.tracker =  DAM4SAMTracker(model)

    def identify_input_type(self, input_dir):
        """
        Identifies if the given path is a folder, a video file, or something else.

        Args:
            input_dir (str): The path to check.

        Returns:
            str: A string indicating the type: "folder", "video", "other_file", or "not_found".
        """
        # A set of common video file extensions (lowercase)
        video_extensions = {
            ".mp4", ".avi", ".mov", ".mkv", ".flv", 
            ".wmv", ".webm", ".mpeg", ".mpg"
        }

        # First, check if the path points to a directory
        if os.path.isdir(input_dir):
            return "folder"
        
        # Next, check if the path points to a file
        elif os.path.isfile(input_dir):
            # If it's a file, get its extension
            # os.path.splitext splits 'video.mp4' into ('video', '.mp4')
            filename, file_extension = os.path.splitext(input_dir)
            
            # Check if the extension is in our set of video extensions
            if file_extension.lower() in video_extensions:
                return "video"
            else:
                return "other_file"
                
        # If the path is neither a file nor a directory, it doesn't exist
        else:
            return "not_found"

    def check_input(self, input_dir):
        input_type = self.identify_input_type(input_dir)
        if input_type == 'video':
            logger.info(f'{input_dir} is a video 📹')
        elif input_type == 'folder':
            pass
        else:
            raise f'{input_dir} is not video nor a folder full of images'

        if input_type == 'video':
            input_dir = self.convert_video2frames(input_dir)
        return input_dir


    @staticmethod
    def convert_video2frames(video_path, output_dir=None, exist_ok=False, frame_skip=0, max_size_kb=300, num_workers=None):
        if num_workers is None:
            num_workers = max(1, os.cpu_count() - 1)

        if output_dir is None:
            output_dir = os.path.join('./tmp', os.path.splitext(os.path.basename(video_path))[0])

        base_dir = output_dir
        counter = 1
        while os.path.exists(output_dir) and not exist_ok:
            output_dir = f"{base_dir}{counter}"
            counter += 1
        os.makedirs(output_dir, exist_ok=True)

        # Print information
        logger.info("-" * 50)
        logger.info("Converting video to frames")
        logger.info(f"  -> Video:                                 {video_path}")
        logger.info(f"  -> Output Directory:                      {output_dir}")
        logger.info(f"  -> Skipping frames:                       {frame_skip}")
        logger.info(f"  -> Maximum File Size in KB per Frame:     {max_size_kb}")
        logger.info(f"  -> Number of CPU cores used:              {num_workers}")
        logger.info("-" * 50)

        extract_and_resize_frames(video_path, output_dir, frame_skip, max_size_kb, num_workers)

        return output_dir


    @staticmethod
    def convert_frames2video(frames_dir, output_file=None, file_extension='jpg', exist_ok=False, fps=30):
        if output_file is None:
            output_file = './demo_output.mp4'

        base_file = output_file
        counter = 1
        while os.path.exists(output_file) and not exist_ok:
            output_file = f"{base_file}{counter}"
            counter += 1

        create_video_from_frames(frames_dir, file_extension, output_file, fps)

    def track(
        self, 
        input_dir, 
        output_dir=None, 
        file_extension='jpg',
        fp16=True,
        visualize=True,
        save_bboxes=True,
        save_frames=True,
        save_video=True,
        out_video_name=None,
        exist_ok=False,
        ):

        input_dir = self.check_input(input_dir)
        frames_dir = sorted(glob.glob(os.path.join(input_dir, f'*.{file_extension}')))

        save_bboxes = save_bboxes if output_dir is not None else False
        save_frames = save_frames if output_dir is not None else False
        visualize = visualize if output_dir is not None else True

        if save_video:
            if not save_frames:
                save_frames = True
                logger.info("We need frames to create the video. Changing saves_frames to True.")

        # Create output directories
        if output_dir:
            # If the specified directory exists, find a new unique name by appending a number
            base_dir = output_dir
            counter = 1
            while os.path.exists(output_dir) and not exist_ok:
                output_dir = f"{base_dir}{counter}"
                counter += 1
            
            # Create the new unique directory and its subdirectories
            if save_frames:
                annotated_frames_dir = os.path.join(output_dir, 'annotated_frames')
                os.makedirs(annotated_frames_dir)
            if save_bboxes:
                labels_dir = os.path.join(output_dir, 'labels')
                os.makedirs(labels_dir)

        # Print information
        logger.info("-" * 50)
        logger.info("Setting up variables")
        logger.info(f"  -> Model:                {self.model_name}")
        logger.info(f"  -> Using fp16:           {'Yes' if fp16 else 'No'}")
        logger.info(f"  -> Input Directory:      {os.path.abspath(input_dir)}")
        logger.info(f"  -> Output Directory:     {os.path.abspath(output_dir) if output_dir else 'None (only visualization mode)'}")
        logger.info(f"  -> Save bounding boxes:  {'Yes' if save_bboxes else 'No'}")
        logger.info(f"  -> Save image frames:    {'Yes' if save_frames else 'No'}")
        logger.info(f"  -> Save vide:            {'Yes' if save_video else 'No'}")
        logger.info(f"  -> Visualization mode:   {'Yes' if visualize else 'No'}")
        logger.info(f"  -> Frames to process:    {len(frames_dir)}")
        logger.info("-" * 50)

        # Multi-box selection
        if os.getenv("COLAB_RELEASE_TAG") or 'COLAB_GPU' in os.environ:
            logger.info("Running on Google Colab")
            from dam4sam.utils.colab_box_selector import ColabMultiFrameBoxSelector
            multi_box_selector = ColabMultiFrameBoxSelector(frames_dir)
            multi_box_selector.select()
            subjects_info = selector.results
            visualize = False # colab does not support video visualization
        else:
            # running on a server
            multi_box_selector = MultiFrameBoxSelector()
            subjects_info = multi_box_selector.select_boxes(frames_dir)
        if not subjects_info:
            print('Error: No initialization boxes were given')
            return
                    
        print(f'Using boxes for tracking:')
        pprint.pprint(subjects_info)


        # Set up live visualization
        if visualize:
            window_name = 'DAM4SAM Tracking'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            wait_ = 1


        pil_images = []
        for i, frame_path in enumerate(frames_dir): 
            img = Image.open(frame_path).convert("RGB")
            pil_images.append(img)

        user_stop = False
        logger.info("Segmenting Frames")
        with tqdm(total=len(frames_dir), desc="Tracking", unit="frame") as pbar:
            for i, frame_path in enumerate(frames_dir):
                img = Image.open(frame_path).convert("RGB")
                img_vis = np.array(img)
                img_h, img_w, _ = img_vis.shape

                if i == 0:
                    outputs = self.tracker.initialize(pil_images, None, subjects_info=subjects_info, use_fp16=fp16)
                else:
                    outputs = self.tracker.track(img, use_fp16=fp16)
                
                lines_for_label_file = []

                for obj_id, mask in outputs.items():
                    # Get bounding box from mask
                    mask_coords = np.column_stack(np.where(mask > 0)[::-1])
                    if mask_coords.size == 0:
                        continue 
                    x, y, w, h = cv2.boundingRect(mask_coords)

                    # Draw bounding boxes and masks of the current object
                    if visualize or save_frames:
                        color_bgr = COLORS_BGR[obj_id % len(COLORS_BGR)]
                        overlay_mask(img_vis, mask, color_bgr, line_width=1, alpha=0.5)
                        cv2.rectangle(img_vis, (x, y), (x + w, y + h), color_bgr, 2)
                        
                        draw_text_with_background(
                            scene=img_vis,
                            text=f"ID: {obj_id}",
                            pos=(x, y - 5),
                            bg_color=color_bgr
                        )

                    # Save bounding box (x1y1x2y2 format) of the current object
                    if save_bboxes:
                        norm_x1, norm_y1 = x / img_w, y / img_h
                        norm_x2, norm_y2 = (x + w) / img_w, (y + h) / img_h
                        
                        bbox_str = f"{norm_x1:.8f} {norm_y1:.8f} {norm_x2:.8f} {norm_y2:.8f}"
                        line = f"{bbox_str} {obj_id}"
                        lines_for_label_file.append(line)

                if save_frames:
                    base_name = os.path.splitext(os.path.basename(frame_path))[0]
                    annotated_frame_path = os.path.join(annotated_frames_dir, f'{base_name}.jpg')
                    cv2.imwrite(annotated_frame_path, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

                if save_bboxes:
                    label_path = os.path.join(labels_dir, f'{base_name}.txt')
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(lines_for_label_file))

                if visualize: 
                    cv2.imshow(window_name, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
                    key_ = cv2.waitKey(wait_)
                    
                    if key_ == 27:  # Esc key to exit
                        user_stop = True
                        break
                    elif key_ == ord('q'): # 'q' key to quit
                        user_stop = True
                        break
                    elif key_ == 32:  # Spacebar to pause/play
                        wait_ = 1 - wait_
                
                if VRAM_SUPPORT:
                    vram_used_mb = torch.cuda.memory_allocated() / (1024**2)
                    pbar.set_postfix_str(f"VRAM: {vram_used_mb:.2f} MB")
                
                pbar.update(1)

        if visualize:
            cv2.destroyAllWindows()
        if user_stop:
            logger.info('Code stop by the user.')
        else:
            logger.info("Segmenting Done.")

        if save_video:
            self.convert_frames2video(annotated_frames_dir, out_video_name)


            




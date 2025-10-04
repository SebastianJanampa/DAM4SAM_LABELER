import os
import cv2
import numpy as np
import pprint
import matplotlib.pyplot as plt


if os.getenv("COLAB_RELEASE_TAG") or 'COLAB_GPU' in os.environ:
    import ipywidgets as widgets
    from IPython.display import display, clear_output

class ColabMultiFrameBoxSelector:

    """
    An interactive tool for annotating objects with persistent IDs across multiple images
    of the same dimensions. This version clamps all bounding boxes upon finishing.
    """
    def __init__(self, image_paths: list, on_finish_callback=None):
        if not image_paths:
            raise ValueError("image_paths list cannot be empty.")
            
        self.image_paths = image_paths
        
        self.master_width = 0
        self.master_height = 0
        try:
            img = cv2.imread(image_paths[0])
            h, w, _ = img.shape
            self.master_height = h
            self.master_width = w
        except Exception as e:
            raise IOError(f"Could not load the first image to get dimensions: {image_paths[0]}\n{e}")

        self.annotations = {path: {} for path in image_paths}
        self.master_id_list = []
        self.next_id = 0
        
        self.active_image_idx = 0
        self.active_object_id = None
        self.image, self.img_height, self.img_width = None, 0, 0

        self.on_finish_callback = on_finish_callback # Store the callback function
        self.results = None

        self._setup_ui()
        self._load_image_and_update_state()

    def _setup_ui(self):
        # This method remains unchanged
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.canvas.header_visible = False
        plt.close(self.fig)

        self.image_slider = widgets.IntSlider(min=0, max=len(self.image_paths) - 1, value=0, description='Image:', style={'description_width': 'initial'}, continuous_update=False)
        self.object_selector = widgets.Dropdown(description='Active Box ID:', style={'description_width': 'initial'})
        
        self.x_slider = widgets.FloatSlider(step=1, description='X:')
        self.y_slider = widgets.FloatSlider(step=1, description='Y:')
        self.w_slider = widgets.FloatSlider(min=10, step=1, description='Width:')
        self.h_slider = widgets.FloatSlider(min=10, step=1, description='Height:')
        self.sliders = [self.x_slider, self.y_slider, self.w_slider, self.h_slider]
        
        self.add_new_id_button = widgets.Button(description="Create New ID", icon='plus', button_style='primary')
        self.existing_id_selector = widgets.Dropdown(description="Existing IDs:", style={'description_width': 'initial'})
        self.add_existing_button = widgets.Button(description="Add to Image", icon='link', button_style='info')
        self.delete_button = widgets.Button(description="Delete Box from Image", icon='trash', button_style='danger')
        self.finish_button = widgets.Button(description="Finish Session", button_style='success', icon='check')
        
        self.output_widget = widgets.Output()
        
        slider_box = widgets.VBox([widgets.HBox([self.x_slider, self.y_slider]), widgets.HBox([self.w_slider, self.h_slider])])
        add_existing_box = widgets.HBox([self.existing_id_selector, self.add_existing_button])
        action_buttons = widgets.HBox([self.add_new_id_button, self.delete_button, self.finish_button])
        
        self.ui_container = widgets.VBox([
            self.image_slider, self.object_selector, slider_box, 
            add_existing_box, action_buttons, self.output_widget
        ])
    
    # Event Handlers... (all remain the same)
    def _on_image_changed(self, change):
        self._on_slider_move(None) 
        self.active_image_idx = change['new']
        self._load_image_and_update_state()

    def _on_object_changed(self, change):
        if not change['new'] and change['new'] != 0: return
        self.active_object_id = change['new']
        self._update_sliders()
        self._redraw_canvas()

    def _on_slider_move(self, change):
        if self.active_object_id is None: return
        active_path = self.image_paths[self.active_image_idx]
        self.annotations[active_path][self.active_object_id] = {
            "x": self.x_slider.value, "y": self.y_slider.value,
            "w": self.w_slider.value, "h": self.h_slider.value
        }
        self._redraw_canvas()

    def _on_add_new_id_click(self, b):
        new_id = self.next_id
        self.master_id_list.append(new_id)
        self.next_id += 1
        active_path = self.image_paths[self.active_image_idx]
        self.annotations[active_path][new_id] = {"x": self.img_width*0.25, "y": self.img_height*0.25, "w": self.img_width*0.5, "h": self.img_height*0.5}
        self._update_all_id_selectors()
        self.object_selector.value = new_id

    def _on_add_existing_id_click(self, b):
        id_to_add = self.existing_id_selector.value
        if id_to_add is None: return
        active_path = self.image_paths[self.active_image_idx]
        self.annotations[active_path][id_to_add] = {"x": self.img_width*0.25, "y": self.img_height*0.25, "w": self.img_width*0.5, "h": self.img_height*0.5}
        self._update_all_id_selectors()
        self.object_selector.value = id_to_add

    def _on_delete_object_click(self, b):
        if self.active_object_id is None: return
        active_path = self.image_paths[self.active_image_idx]
        del self.annotations[active_path][self.active_object_id]
        self.active_object_id = None
        self._update_all_id_selectors()
        if self.object_selector.options: self.object_selector.value = self.object_selector.options[0][1]
        else: self._update_sliders(empty=True); self._redraw_canvas()

    def _on_finish_click(self, b):
        self._on_slider_move(None)
        self.results = self._reformat_and_clamp_annotations()
        
        for widget in self.ui_container.children: widget.disabled = True
        b.description = "✅ Done!"
        
        with self.output_widget:
            clear_output(wait=True)
            print("Annotation session finished! Triggering tracking process...")
        
        # This is the crucial part: call the callback with the results
        if self.on_finish_callback:
            self.on_finish_callback(self.results)
        
    def _load_image_and_update_state(self):
        active_path = self.image_paths[self.active_image_idx]
        self.image_slider.description = os.path.basename(active_path)
        try:
            self.image = cv2.cvtColor(cv2.imread(active_path), cv2.COLOR_BGR2RGB)
            self.img_height, self.img_width, _ = self.image.shape
        except Exception as e:
            with self.output_widget: clear_output(wait=True); print(f"Error loading image: {active_path}\n{e}")
            return
            
        for s in [self.x_slider, self.w_slider]: s.max = self.img_width
        for s in [self.y_slider, self.h_slider]: s.max = self.img_height
        self.active_object_id = None
        self._update_all_id_selectors()
        if self.object_selector.options: self.object_selector.value = self.object_selector.options[0][1]
        else: self._update_sliders(empty=True); self._redraw_canvas()

    # The rest of the helper methods are unchanged...
    def _update_all_id_selectors(self):
        active_path = self.image_paths[self.active_image_idx]
        ids_on_this_image = self.annotations[active_path].keys()
        self.object_selector.options = [(f'ID {obj_id}', obj_id) for obj_id in sorted(list(ids_on_this_image))]
        ids_to_add = sorted(list(set(self.master_id_list) - set(ids_on_this_image)))
        self.existing_id_selector.options = [(f'ID {obj_id}', obj_id) for obj_id in ids_to_add]
        self.add_existing_button.disabled = not ids_to_add
        self.delete_button.disabled = not ids_on_this_image

    def _update_sliders(self, empty=False):
        for s in self.sliders: s.unobserve(self._on_slider_move, 'value')
        if empty or self.active_object_id is None:
            for s in self.sliders: s.value = s.min
        else:
            active_path = self.image_paths[self.active_image_idx]
            active_bbox = self.annotations[active_path][self.active_object_id]
            self.x_slider.value, self.y_slider.value = active_bbox['x'], active_bbox['y']
            self.w_slider.value, self.h_slider.value = active_bbox['w'], active_bbox['h']
        for s in self.sliders: s.observe(self._on_slider_move, 'value')

    def _redraw_canvas(self):
        with self.output_widget:
            clear_output(wait=True)
            self.ax.clear()
            self.ax.imshow(self.image)
            active_path = self.image_paths[self.active_image_idx]
            bboxes_dict = self.annotations[active_path]
            self.ax.set_title(f"Image {self.active_image_idx+1}/{len(self.image_paths)}: {len(bboxes_dict)} boxes")
            self.ax.axis('off')
            for obj_id, bbox in bboxes_dict.items():
                if obj_id == self.active_object_id: edge_color, line_style = 'lime', '-'
                else: edge_color, line_style = 'yellow', '--'
                rect = plt.Rectangle((bbox['x'], bbox['y']), bbox['w'], bbox['h'], edgecolor=edge_color, facecolor='none', lw=2.5, linestyle=line_style)
                self.ax.add_patch(rect)
                self.ax.text(bbox['x'], bbox['y'] - 5, f'ID: {obj_id}', color=edge_color, fontsize=10, weight='bold')
            display(self.fig)

    def _reformat_and_clamp_annotations(self):
        final_output = {}
        img_w, img_h = self.master_width, self.master_height

        for frame_idx, image_path in enumerate(self.image_paths):
            annotations_on_frame = self.annotations[image_path]
            
            for object_id, bbox_dict in annotations_on_frame.items():
                if object_id not in final_output:
                    final_output[object_id] = {'frame_info': {}}
                
                x, y, w, h = bbox_dict['x'], bbox_dict['y'], bbox_dict['w'], bbox_dict['h']
                
                # --- MODIFICATION: More explicit and correct clamping logic ---
                # First, define the four corner points
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                
                # Clamp the corner points to the valid range.
                # The top-left (x1, y1) must be >= 0.
                # The bottom-right (x2, y2) must be <= dimension.
                clamped_x1 = min(max(0, x1), img_w-1)
                clamped_y1 = min(max(0, y1), img_h-1)
                clamped_x2 = min(img_w-1, x2)
                clamped_y2 = min(img_h-1, y2)
                
                # Recalculate width and height from the clamped corners.
                # This correctly handles boxes that are partially or fully outside.
                new_w = clamped_x2 - clamped_x1
                new_h = clamped_y2 - clamped_y1

                # Ensure width and height are not negative
                clamped_bbox = [int(clamped_x1), int(clamped_y1), int(max(0, new_w)), int(max(0, new_h))]
                
                final_output[object_id]['frame_info'][frame_idx] = {
                    'bbox': clamped_bbox,
                    'labels': None,
                    'points': None
                }
        return final_output

    def select(self):

        display(self.ui_container)
        self.image_slider.observe(self._on_image_changed, names='value')
        self.object_selector.observe(self._on_object_changed, names='value')
        self.add_new_id_button.on_click(self._on_add_new_id_click)
        self.add_existing_button.on_click(self._on_add_existing_id_click)
        self.delete_button.on_click(self._on_delete_object_click)
        self.finish_button.on_click(self._on_finish_click)
        for s in self.sliders: s.observe(self._on_slider_move, names='value')
    


import cv2
import numpy as np
import pprint
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class MultiFrameBoxSelector:
    """
    An advanced Tkinter-based GUI for annotating items (bboxes, points) in a sequence of frames.

    Features include creating, moving, resizing, and deleting items, toggling point labels,
    a real-time information panel with persistent view state, the ability to switch between
    object IDs for editing, and a comprehensive set of keyboard shortcuts.
    """
    def __init__(self):
        # --- Core Data and State ---
        self._internal_data = {}
        self.current_obj_id = 0
        self.next_obj_id = 1
        self.current_mode = 'SELECT'
        self.was_cancelled = True

        # --- Drawing/Editing State ---
        self.start_point = None
        self.drawing_item_id = None
        self.editing_item_info = None
        self.selected_item_info = None
        self.resizing_info = None

        # --- Frame Data ---
        self.frames = []
        self.pil_frames = []
        self.current_frame_index = 0
        
        # --- UI Elements ---
        self.root = None
        self.window = None
        self.canvas = None
        self.info_tree = None
        self.status_labels = {}
        
        # NEW: Add variables for image scaling
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.display_width = 0
        self.display_height = 0
        
        # --- Constants ---
        self.CLICK_RADIUS = 7
        self.HANDLE_SIZE = 4
        self.colors = {
            'bbox': 'green', 'point_pos': 'cyan', 'point_neg': 'red',
            'selected': 'magenta', 'drawing': 'yellow', 'handle': 'white'
        }

    #======================================================================
    # region: Core Data Logic
    #======================================================================
    def _check_bbox_exists_on_frame(self, obj_id):
        """Checks if a bbox already exists for the given obj_id on the CURRENT frame."""
        for item in self._internal_data.get(self.current_frame_index, []):
            if item['id'] == obj_id and item['type'] == 'bbox':
                return True
        return False

    def _add_bbox(self, bbox):
        if self._check_bbox_exists_on_frame(self.current_obj_id):
            print(f"Error: BBox for ID {self.current_obj_id} already exists on this frame.")
            return
        if bbox and bbox[2] > 0 and bbox[3] > 0:
            new_item = {'id': self.current_obj_id, 'type': 'bbox', 'data': bbox}
            self._internal_data.setdefault(self.current_frame_index, []).append(new_item)
            print(f"Frame {self.current_frame_index}: Added BBox for ID {self.current_obj_id}")

    def _add_point(self, x, y, label):
        new_item = {'id': self.current_obj_id, 'type': 'point', 'data': [x, y], 'label': label}
        self._internal_data.setdefault(self.current_frame_index, []).append(new_item)
        print(f"Frame {self.current_frame_index}: Added {'Pos' if label==1 else 'Neg'} Point for ID {self.current_obj_id}")

    def _get_item_at_coords(self, event_x, event_y):
        # MODIFIED: This function now compares coordinates in DISPLAY space for accuracy
        items_on_frame = self._internal_data.get(self.current_frame_index, [])
        
        # Check for handle clicks on the selected item first
        if self.selected_item_info:
            sel_idx = self.selected_item_info['item_idx']
            if 0 <= sel_idx < len(items_on_frame) and items_on_frame[sel_idx]['type'] == 'bbox':
                handle = self._get_handle_at_coords(event_x, event_y)
                if handle:
                    return self.selected_item_info['item_idx'], handle

        # Check for clicks on any item, iterating from top to bottom (last drawn to first)
        for i in range(len(items_on_frame) - 1, -1, -1):
            item = items_on_frame[i]
            if item['type'] == 'point':
                # Convert original point coords to display coords for comparison
                xp_orig, yp_orig = item['data']
                xp_disp, yp_disp = xp_orig / self.scale_x, yp_orig / self.scale_y
                if (event_x - xp_disp)**2 + (event_y - yp_disp)**2 <= self.CLICK_RADIUS**2:
                    return i, None
            elif item['type'] == 'bbox':
                # Convert original bbox coords to display coords for comparison
                xb_orig, yb_orig, wb_orig, hb_orig = item['data']
                xb_disp, yb_disp = xb_orig / self.scale_x, yb_orig / self.scale_y
                wb_disp, hb_disp = wb_orig / self.scale_x, hb_orig / self.scale_y
                if xb_disp <= event_x < (xb_disp + wb_disp) and yb_disp <= event_y < (yb_disp + hb_disp):
                    return i, None
        return None, None
    # endregion
    
    #======================================================================
    # region: UI Setup and Control
    #======================================================================
    def _setup_ui(self):
        self.window = tk.Toplevel(self.root)
        self.window.title("Multi-Frame Annotator")
        
        main_pane = ttk.PanedWindow(self.window, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding="10")
        main_pane.add(controls_frame, weight=0)
        canvas_frame = ttk.Frame(main_pane, padding="5")
        main_pane.add(canvas_frame, weight=1)
        info_frame = ttk.Frame(main_pane, padding="10")
        main_pane.add(info_frame, weight=0)
        
        mode_lf = ttk.Labelframe(controls_frame, text="Modes", padding=5)
        mode_lf.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(mode_lf, text="Select / Move (S)", command=lambda: self._set_mode('SELECT')).pack(fill=tk.X)
        ttk.Button(mode_lf, text="Add BBox (B)", command=lambda: self._set_mode('ADD_BBOX')).pack(fill=tk.X)
        ttk.Button(mode_lf, text="Add Positive Point (P)", command=lambda: self._set_mode('ADD_POS_POINT')).pack(fill=tk.X)
        ttk.Button(mode_lf, text="Add Negative Point (N)", command=lambda: self._set_mode('ADD_NEG_POINT')).pack(fill=tk.X)
        
        action_lf = ttk.Labelframe(controls_frame, text="Actions", padding=5)
        action_lf.pack(fill=tk.X, pady=10)
        ttk.Button(action_lf, text="Create New Object ID (I)", command=self._increment_obj_id).pack(fill=tk.X)
        ttk.Button(action_lf, text="Toggle Point Label (T)", command=self._toggle_selected_point_label).pack(fill=tk.X)
        ttk.Button(action_lf, text="Delete Selected (Del)", command=self._delete_selected_item).pack(fill=tk.X)

        status_lf = ttk.Labelframe(controls_frame, text="Status", padding=10)
        status_lf.pack(fill=tk.X, pady=10)
        for name in ["frame", "id", "mode"]:
            self.status_labels[name] = ttk.Label(status_lf, font=("Helvetica", 10))
            self.status_labels[name].pack(anchor=tk.W)

        finish_frame = ttk.Frame(controls_frame)
        finish_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10,0))
        ttk.Button(finish_frame, text="Finish", command=self._finish_selection).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(finish_frame, text="Cancel", command=self._cancel_selection).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # MODIFIED: Set canvas size to the calculated display size
        self.canvas = tk.Canvas(canvas_frame, width=self.display_width, height=self.display_height, cursor="cross", bg="black")
        self.canvas.pack()
        self.frame_slider = ttk.Scale(canvas_frame, from_=0, to=len(self.frames)-1, orient=tk.HORIZONTAL, command=self._on_frame_change)
        if len(self.frames) > 1: self.frame_slider.pack(fill=tk.X, pady=(5,0))

        ttk.Label(info_frame, text="Object Information").pack(anchor=tk.W)
        tree_frame = ttk.Frame(info_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.info_tree = ttk.Treeview(tree_frame, columns=('Value'), show='tree headings')
        self.info_tree.heading('#0', text='Item')
        self.info_tree.heading('Value', text='Details')
        self.info_tree.column('Value', width=200)
        self.info_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.info_tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_tree.configure(yscrollcommand=vsb.set)
        
        self.window.protocol("WM_DELETE_WINDOW", self._cancel_selection)

    def _finish_selection(self):
        self.was_cancelled = False
        self.window.destroy()

    def _cancel_selection(self):
        self._internal_data = {}
        self.was_cancelled = True
        self.window.destroy()
    # endregion
    
    #======================================================================
    # region: UI Callbacks and Event Handlers
    #======================================================================
    def _set_mode(self, mode):
        self.current_mode = mode
        if mode != 'RESIZE':
            self.resizing_info = None
        self._update_display()

    def _increment_obj_id(self):
        all_ids = list(set([item['id'] for items in self._internal_data.values() for item in items]))
        self.next_obj_id = max(all_ids) + 1 if all_ids else 0
        self.current_obj_id = self.next_obj_id
        self.selected_item_info = None
        self._set_mode('SELECT')
        print(f"Switched to new Object ID: {self.current_obj_id}")

    def _delete_selected_item(self):
        if self.selected_item_info:
            frame_idx, item_idx = self.selected_item_info['frame_idx'], self.selected_item_info['item_idx']
            items_on_frame = self._internal_data.get(frame_idx, [])
            if 0 <= item_idx < len(items_on_frame):
                items_on_frame.pop(item_idx)
                if not items_on_frame: del self._internal_data[frame_idx]
                self.selected_item_info = None
                self._update_display()
        
    def _toggle_selected_point_label(self):
        if self.selected_item_info:
            item = self._internal_data[self.selected_item_info['frame_idx']][self.selected_item_info['item_idx']]
            if item['type'] == 'point':
                item['label'] = 1 - item['label']
                self._update_display()

    def _on_frame_change(self, val, from_key=False):
        idx = int(float(val))
        if self.current_frame_index == idx: return
        self.current_frame_index = idx
        self.selected_item_info = None
        if not from_key: self.frame_slider.set(idx)
        self._update_display()

    def _update_display(self):
        """Refreshes all UI components."""
        self.status_labels["frame"].config(text=f"Frame: {self.current_frame_index + 1}/{len(self.frames)}")
        self.status_labels["id"].config(text=f"Current ID: {self.current_obj_id}")
        self.status_labels["mode"].config(text=f"Mode: {self.current_mode}")
        self._redraw_canvas()
        self._update_info_display()

    def _get_all_children(self, tree, item=""):
        """Recursively gets all item IDs in a TreeView."""
        children = list(tree.get_children(item))
        for child in tree.get_children(item):
            children.extend(self._get_all_children(tree, child))
        return children

    def _update_info_display(self):
        """Rebuilds the TreeView while preserving the expanded/collapsed state."""
        expanded_tags = set()
        collapsed_tags = set()
        if self.info_tree.get_children():
            all_iids = self._get_all_children(self.info_tree)
            for iid in all_iids:
                tags = self.info_tree.item(iid, 'tags')
                if tags:
                    if self.info_tree.item(iid, 'open'):
                        expanded_tags.add(tags[0])
                    else:
                        collapsed_tags.add(tags[0])

        self.info_tree.delete(*self.info_tree.get_children())
        output_dict = self._get_formatted_data()
        all_ids = sorted(list(set([item['id'] for items in self._internal_data.values() for item in items])))

        for obj_id in all_ids:
            obj_tag = f"obj_{obj_id}"
            is_obj_open = obj_tag not in collapsed_tags
            obj_node = self.info_tree.insert('', tk.END, text=f"Object {obj_id}", open=is_obj_open, tags=(obj_tag,), values=(f'ID: {obj_id}',))
            
            if obj_id in output_dict:
                for frame_idx, info in sorted(output_dict[obj_id]['frame_info'].items()):
                    frame_tag = f"obj_{obj_id}_frame_{frame_idx}"
                    is_frame_open = frame_tag not in collapsed_tags
                    frame_node = self.info_tree.insert(obj_node, tk.END, text=f"Frame {frame_idx}", open=is_frame_open, tags=(frame_tag,))
                    
                    if info['bbox']:
                        self.info_tree.insert(frame_node, tk.END, text="BBox", values=(f"{info['bbox']}",))
                    
                    if info['points']:
                        points_tag = f"{frame_tag}_points"
                        is_points_open = points_tag in expanded_tags
                        points_node = self.info_tree.insert(frame_node, tk.END, text=f"{len(info['points'])} Points", open=is_points_open, tags=(points_tag,))
                        for i, point in enumerate(info['points']):
                            label = info['labels'][i]
                            label_text = "Positive" if label == 1 else "Negative"
                            self.info_tree.insert(points_node, tk.END, text=f"Point {i+1}", values=(f"{point} -> {label_text}",))

    def _on_info_select(self, event):
        selected_iid = self.info_tree.selection()
        if not selected_iid: return
        item_text = self.info_tree.item(selected_iid[0], 'text')
        if item_text.startswith("Object"):
            try:
                obj_id = int(item_text.split(" ")[1])
                self.current_obj_id = obj_id
                self._update_display()
                print(f"Switched to edit Object ID: {obj_id}")
            except (ValueError, IndexError):
                pass
    
    def _redraw_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.pil_frames[self.current_frame_index])

        for i, item in enumerate(self._internal_data.get(self.current_frame_index, [])):
            is_selected = self.selected_item_info and self.selected_item_info['item_idx'] == i and self.selected_item_info['frame_idx'] == self.current_frame_index
            color = self.colors['selected'] if is_selected else None
            
            if item['type'] == 'bbox':
                if not color: color = self.colors['bbox']
                # MODIFIED: Scale original coords down to display coords for drawing
                x_orig, y_orig, w_orig, h_orig = item['data']
                x_disp, y_disp = x_orig / self.scale_x, y_orig / self.scale_y
                w_disp, h_disp = w_orig / self.scale_x, h_orig / self.scale_y
                
                self.canvas.create_rectangle(x_disp, y_disp, x_disp + w_disp, y_disp + h_disp, outline=color, width=2)
                self.canvas.create_text(x_disp, y_disp - 10, text=f"ID:{item['id']}", fill=color, anchor=tk.W)
                if is_selected: self._draw_handles(item['data'])

            elif item['type'] == 'point':
                label, (x_orig, y_orig) = item['label'], item['data']
                if not color: color = self.colors['point_pos'] if label == 1 else self.colors['point_neg']
                # MODIFIED: Scale original coords down to display coords for drawing
                x_disp, y_disp = x_orig / self.scale_x, y_orig / self.scale_y
                
                r = self.CLICK_RADIUS
                if label == 1:
                    self.canvas.create_line(x_disp - r, y_disp, x_disp + r, y_disp, fill=color, width=2)
                    self.canvas.create_line(x_disp, y_disp - r, x_disp, y_disp + r, fill=color, width=2)
                else:
                    self.canvas.create_oval(x_disp - r, y_disp - r, x_disp + r, y_disp + r, outline=color, width=2)
                self.canvas.create_text(x_disp + 8, y_disp + 8, text=f"ID:{item['id']}", fill=color, anchor=tk.W)

    def _draw_handles(self, bbox):
        # MODIFIED: Scale original bbox coords down to display coords for drawing handles
        x_orig, y_orig, w_orig, h_orig = bbox
        x_disp, y_disp = x_orig / self.scale_x, y_orig / self.scale_y
        w_disp, h_disp = w_orig / self.scale_x, h_orig / self.scale_y
        
        hs = self.HANDLE_SIZE
        coords = [
            (x_disp, y_disp), (x_disp + w_disp/2, y_disp), (x_disp + w_disp, y_disp),
            (x_disp, y_disp + h_disp/2), (x_disp + w_disp, y_disp + h_disp/2),
            (x_disp, y_disp + h_disp), (x_disp + w_disp/2, y_disp + h_disp), (x_disp + w_disp, y_disp + h_disp)
        ]
        for cx, cy in coords:
            self.canvas.create_rectangle(cx - hs, cy - hs, cx + hs, cy + hs, fill=self.colors['handle'], outline='black')

    def _get_handle_at_coords(self, event_x, event_y):
        item_idx = self.selected_item_info['item_idx']
        bbox_orig = self._internal_data.get(self.current_frame_index, [])[item_idx]['data']
        
        # MODIFIED: Convert original bbox coords to display coords for handle detection
        bx_orig, by_orig, bw_orig, bh_orig = bbox_orig
        bx_disp, by_disp = bx_orig / self.scale_x, by_orig / self.scale_y
        bw_disp, bh_disp = bw_orig / self.scale_x, bh_orig / self.scale_y
        
        hs = self.HANDLE_SIZE
        handles = {
            'nw': (bx_disp, by_disp), 'n': (bx_disp + bw_disp/2, by_disp), 'ne': (bx_disp + bw_disp, by_disp),
            'w': (bx_disp, by_disp + bh_disp/2), 'e': (bx_disp + bw_disp, by_disp + bh_disp/2),
            'sw': (bx_disp, by_disp + bh_disp), 's': (bx_disp + bw_disp/2, by_disp + bh_disp), 'se': (bx_disp + bw_disp, by_disp + bh_disp)
        }
        for name, (hx, hy) in handles.items():
            if hx - hs <= event_x <= hx + hs and hy - hs <= event_y <= hy + hs:
                return name
        return None
    
    def _on_mouse_press(self, event):
        item_idx, handle = self._get_item_at_coords(event.x, event.y)

        # Deselect if clicking on empty space
        if item_idx is None and handle is None:
            if self.selected_item_info:
                self.selected_item_info = None
                self._update_display() 

        if handle:
            self.resizing_info = {'item_idx': item_idx, 'handle': handle}
            self._set_mode('RESIZE')
            return

        # MODIFIED: Scale event coordinates up to original image space for calculations
        orig_x, orig_y = event.x * self.scale_x, event.y * self.scale_y
        
        if self.current_mode == 'SELECT' and item_idx is not None:
            item = self._internal_data.get(self.current_frame_index, [])[item_idx]
            # Calculate offset in original coordinate space
            off_x = orig_x - item['data'][0]
            off_y = orig_y - item['data'][1]
            self.editing_item_info = {'item_idx': item_idx, 'off_x': off_x, 'off_y': off_y}
            self._update_display()
        else: 
            self.editing_item_info = None
            if self.current_mode == 'ADD_BBOX':
                self.start_point = (event.x, event.y) # Use display coords for drawing rect
                self.drawing_item_id = self.canvas.create_rectangle(*self.start_point, *self.start_point, outline=self.colors['drawing'], width=2)
            elif self.current_mode == 'ADD_POS_POINT': 
                self._add_point(orig_x, orig_y, 1) # Add point in original coords
                self._set_mode('SELECT')
            elif self.current_mode == 'ADD_NEG_POINT': 
                self._add_point(orig_x, orig_y, 0) # Add point in original coords
                self._set_mode('SELECT')

    def _on_mouse_drag(self, event):
        if self.resizing_info:
            item = self._internal_data[self.current_frame_index][self.resizing_info['item_idx']]
            # MODIFIED: Scale event coords to original space for resizing logic
            orig_x, orig_y = event.x * self.scale_x, event.y * self.scale_y
            x, y, w, h = item['data']
            handle = self.resizing_info['handle']
            
            if 'e' in handle: w = max(1, orig_x - x)
            if 'w' in handle: w = max(1, x + w - orig_x); x = orig_x
            if 's' in handle: h = max(1, orig_y - y)
            if 'n' in handle: h = max(1, y + h - orig_y); y = orig_y
            
            item['data'] = [x, y, w, h]
            self._redraw_canvas()
        elif self.editing_item_info:
            item = self._internal_data[self.current_frame_index][self.editing_item_info['item_idx']]
            # MODIFIED: Scale event coords to original space for moving logic
            orig_x, orig_y = event.x * self.scale_x, event.y * self.scale_y
            
            item['data'][0] = orig_x - self.editing_item_info['off_x']
            item['data'][1] = orig_y - self.editing_item_info['off_y']
            self._redraw_canvas()
        elif self.drawing_item_id:
            # Drawing rectangle uses display coordinates directly
            self.canvas.coords(self.drawing_item_id, *self.start_point, event.x, event.y)

    def _on_mouse_release(self, event):
        if self.drawing_item_id:
            if self.canvas.find_withtag(self.drawing_item_id):
                # Get final display coordinates of the drawn rectangle
                x0_disp, y0_disp, x1_disp, y1_disp = self.canvas.coords(self.drawing_item_id)
                # MODIFIED: Scale them up to original coordinates before saving
                x0_orig, y0_orig = x0_disp * self.scale_x, y0_disp * self.scale_y
                x1_orig, y1_orig = x1_disp * self.scale_x, y1_disp * self.scale_y
                
                self._add_bbox([min(x0_orig, x1_orig), min(y0_orig, y1_orig), abs(x1_orig - x0_orig), abs(y1_orig - y0_orig)])
                self.canvas.delete(self.drawing_item_id)
            self.drawing_item_id = None
            self.start_point = None
            self._set_mode('SELECT')
            
        if self.resizing_info: self._set_mode('SELECT')
        self.editing_item_info = None
        self.resizing_info = None

    def _on_right_click(self, event):
        self._set_mode('SELECT')
        item_idx, _ = self._get_item_at_coords(event.x, event.y)
        if item_idx is not None:
            self.selected_item_info = {'frame_idx': self.current_frame_index, 'item_idx': item_idx}
        else:
            self.selected_item_info = None
        self._update_display()

    def _on_key_press(self, event):
        key = event.keysym.lower()
        if key in ['b', 'p', 'n', 's', 'i', 't', 'delete', 'escape']:
            if key == 'b': self._set_mode('ADD_BBOX')
            elif key == 'p': self._set_mode('ADD_POS_POINT')
            elif key == 'n': self._set_mode('ADD_NEG_POINT')
            elif key == 's' or key == 'escape': self._set_mode('SELECT')
            elif key == 'i': self._increment_obj_id()
            elif key == 'delete': self._delete_selected_item()
            elif key == 't': self._toggle_selected_point_label()
        elif key == 'right' and self.current_frame_index < len(self.frames) - 1:
            self._on_frame_change(self.current_frame_index + 1, from_key=True)
        elif key == 'left' and self.current_frame_index > 0:
            self._on_frame_change(self.current_frame_index - 1, from_key=True)
    # endregion
    
    #======================================================================
    # region: Main Callable Method
    #======================================================================
    def _get_formatted_data(self):
        """Builds the final data dictionary from the internal representation."""
        output_dict = {}
        for frame_idx, items in sorted(self._internal_data.items()):
            for item in items:
                obj_id = item['id']
                output_dict.setdefault(obj_id, {'frame_info': {}})
                output_dict[obj_id]['frame_info'].setdefault(frame_idx, {'bbox': None, 'points': [], 'labels': []})
                
                if item['type'] == 'bbox':
                    output_dict[obj_id]['frame_info'][frame_idx]['bbox'] = [int(v) for v in item['data']]
                elif item['type'] == 'point':
                    output_dict[obj_id]['frame_info'][frame_idx]['points'].append([int(v) for v in item['data']])
                    output_dict[obj_id]['frame_info'][frame_idx]['labels'].append(item['label'])
        
        for obj_id in output_dict:
            for frame_idx in output_dict[obj_id]['frame_info']:
                frame_data = output_dict[obj_id]['frame_info'][frame_idx]
                if not frame_data['points']:
                    frame_data['points'] = None
                    frame_data['labels'] = None
        
        return output_dict
    
    def select_boxes(self, frame_paths):
        if not frame_paths: return {}
        self.frames = [cv2.imread(p) for p in frame_paths if p is not None]
        if not self.frames: return {}

        self.root = tk.Tk()
        self.root.withdraw()

        # --- MODIFIED: Scaling Logic ---
        # Get original image dimensions from the first frame
        original_height, original_width, _ = self.frames[0].shape

        # Define max display size (e.g., 90% of screen size)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        max_width = int(screen_width * 0.9)
        max_height = int(screen_height * 0.9)

        # Calculate the aspect ratio to fit the image within the max dimensions.
        # By adding min(1.0, ...), we ensure the ratio is never > 1, preventing upscaling.
        ratio = min(1.0, max_width / original_width, max_height / original_height)
        
        self.display_width = int(original_width * ratio)
        self.display_height = int(original_height * ratio)

        # Calculate scaling factors
        # If ratio is 1.0, these will also be 1.0 (no scaling)
        self.scale_x = original_width / self.display_width
        self.scale_y = original_height / self.display_height
        
        # Resize images before creating PhotoImage objects
        self.pil_frames = []
        for frame in self.frames:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_resized = img_pil.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
            self.pil_frames.append(ImageTk.PhotoImage(image=img_resized))
        # --- End of Scaling Logic ---

        self._setup_ui()
        self.canvas.bind("<Button-1>", self._on_mouse_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.info_tree.bind("<<TreeviewSelect>>", self._on_info_select)
        self.window.bind("<KeyPress>", self._on_key_press)
        
        self._update_display()
        self.root.wait_window(self.window)
        
        return {} if self.was_cancelled else self._get_formatted_data()

# Example usage (you can uncomment this to test the class)
if __name__ == '__main__':
    # Create a dummy large image file for testing
    dummy_image_path = 'large_test_image.png'
    # Use a real image path on your system
    # real_image_path = 'path/to/your/large/image.jpg' 
    
    # Check if the user-provided image exists, otherwise create a dummy one
    import os
    if os.path.exists(dummy_image_path):
         image_to_use = dummy_image_path
    else:
        print(f"Creating a dummy image at: {dummy_image_path}")
        dummy_img = np.zeros((2000, 3000, 3), dtype=np.uint8)
        cv2.putText(dummy_img, 'This is a large test image (3000x2000)', (50, 1000), 
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        cv2.imwrite(dummy_image_path, dummy_img)
        image_to_use = dummy_image_path

    # The selector can take a list of frame paths
    frame_paths = [image_to_use]
    
    selector = MultiFrameBoxSelector()
    annotations = selector.select_boxes(frame_paths)

    if annotations:
        print("\n--- Final Annotations (using original image coordinates) ---")
        pprint.pprint(annotations)
    else:
        print("\n--- Annotation process was cancelled. ---")

MODELS = {
    "sam21pp-L": {"file_path": "./checkpoints/sam2.1_hiera_large.pt"},
    "sam21pp-B": {"file_path": "./checkpoints/sam2.1_hiera_base_plus.pt"},
    "sam21pp-S": {"file_path": "./checkpoints/sam2.1_hiera_small.pt"},
    "sam21pp-T": {"file_path": "./checkpoints/sam2.1_hiera_tiny.pt"},
    "sam2pp-L":  {"file_path": "./checkpoints/sam2_hiera_large.pt"},
    "sam2pp-B":  {"file_path": "./checkpoints/sam2_hiera_base_plus.pt"},
    "sam2pp-S":  {"file_path": "./checkpoints/sam2_hiera_small.pt"},
    "sam2pp-T":  {"file_path": "./checkpoints/sam2_hiera_tiny.pt"},
}

# Custom color map for visualization
CUSTOM_COLOR_MAP_HEX = [
    "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#d2f53c", "#fabebe", "#008080", "#e6beff",
    "#aa6e28", "#fffac8", "#800000", "#aaffc3",
]

# Convert hex color strings to BGR tuples for OpenCV
def hex_to_bgr(hex_code):
    """Converts a hex color code to a (B, G, R) tuple."""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (4, 2, 0))

COLORS_BGR = [hex_to_bgr(h) for h in CUSTOM_COLOR_MAP_HEX]
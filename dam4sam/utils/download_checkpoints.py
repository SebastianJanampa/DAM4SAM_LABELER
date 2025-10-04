import requests
from tqdm import tqdm

# Define the base URLs
SAM2p1_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM2_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824"

# Create a nested dictionary to hold URLs for both model versions
model_urls = {
    # SAM 2.1 Models
        "sam21pp-T": f"{SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt",
        "sam21pp-S": f"{SAM2p1_BASE_URL}/sam2.1_hiera_small.pt",
        "sam21pp-B": f"{SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt",
        "sam21pp-L": f"{SAM2p1_BASE_URL}/sam2.1_hiera_large.pt",
    # SAM 2 Models
        "sam2pp-T": f"{SAM2_BASE_URL}/sam2_hiera_tiny.pt",
        "sam2pp-S": f"{SAM2_BASE_URL}/sam2_hiera_small.pt",
        "sam2pp-B": f"{SAM2_BASE_URL}/sam2_hiera_base_plus.pt",
        "sam2pp-L": f"{SAM2_BASE_URL}/sam2_hiera_large.pt",
}

# You can now access any URL by specifying the version first:
# print(model_urls["sam2.1"]["hiera_t"])
# print(model_urls["sam2"]["hiera_l"])

def download_checkpoint(model_name):
    try:
        url = model_urls[model_name]
        local_filename = f"./checkpoints/{url.split('/')[-1]}"
        # Make the request with streaming enabled
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get the total file size from the headers
        total_size_in_bytes = int(response.headers.get('content-length', 0))

        # Define the chunk size
        chunk_size = 10 * 2**30  # 8 KB

        # Set up the progress bar
        progress_bar = tqdm(
            total=total_size_in_bytes, 
            unit='iB', 
            unit_scale=True,
            desc=f"Dowloading {local_filename}"
        )

        # Open the local file in binary write mode
        with open(local_filename, 'wb') as file:
            # Iterate over the response content in chunks
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                progress_bar.update(len(chunk)) # Update the progress bar

        progress_bar.close() # Close the progress bar
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("❌ ERROR, something went wrong during download")
        else:
            print(f"✅ File downloaded successfully to {local_filename}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading file: {e}")
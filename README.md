
<h2 align="center">
  DAM4SAMv2: An improved engineered version of DAM4SAM
</h2>

<p align="center">
  <a href="https://github.com/SebastianJanampa/DAM4SAMv2/blob/main/LICENSE">
        <img alt="colab" src="https://img.shields.io/badge/license-apache%202.0-blue?style=for-the-badge">
  </a>   
</p>

<p align="center">
Sebastian Janampa (Developer)
</p>

<p align="center">
The University of New Mexico
  <br>
Department of Electrical and Computer Engineering
</p>

---

DAM4SAMv2 is an improved version of DAM4SAM. We keep the great distractor-aware memory (DAM) consisting 
of two key parts: the recent appearances memory (RAM) and distractor-resolving memory (DRM).

DAM4SAM has not only surpassed SAM2 metrics, but it also addresses problems such as out-of-memory (OOM) errors caused by SAM2. 
Due to these factors, DAM4SAM has become a popular choice for creating segmentation/tracking ground-truth datasets.
However, the official DAM4SAM implementation has two main drawbacks: 1) it only accepts bounding boxes as input, 
2) single-object tracking, and 3) only uses the first frame of the video. 
Although the dam4sam-multi repo expands DAM4SAM capabilities for multi-object tracking, it does not address 1) and 2).

In this repo, we improve those drawbacks by developing DAM4SAMv2 (an improved engineered version of DAM4SAM). 
We allow users to select/edit/delete bounding boxes and points (which can also change the class of the points), use different frames from the video, and track multiple objects.
For the objects, it is not required that they share the same frames, since not all objects appear in the first frame. 
This feature provides the user with more flexibility and facilitates the segmentation/tracking process.


https://github.com/user-attachments/assets/2bcac2a4-06cc-42f1-882c-0173e4c32f42


## Video demos
We show some videos on how to use the repo in a local server and in Colab.
<details close>
<summary> Local Server Demo </summary>
  
https://github.com/user-attachments/assets/bdc2c051-a058-4675-8f32-13dd244dba3c

</details>

<details close>
<summary> Colab Notebook Demo </summary>

https://github.com/user-attachments/assets/4f157338-aca4-4b36-a466-79d70577a013

</details>

## 📝 TODO
- [ ] Create Python package.
- [ ] Report with evaluations (fps, memory usage, accuracy) on different datasets.

## 🚀 Updates
- [x] **\[2025.10.04\]** DAM4SAMv2 code release.

## Model Zoo
This repo supports the following models
<items>
   - sam21pp-L
   - sam21pp-B
   - sam21pp-S
   - sam21pp-T
   - sam2pp-L
   - sam2pp-B
   - sam2pp-S
   - sam2pp-T
</items>

## Quick start

### Set up

#### Method 1: git clone
```shell
git clone DAM4SAMv2
cd DAM4SAMv2
pip install -e .
```

#### Method 2: pip + git
```shell
pip install git+https://github.com/SebastianJanampa/DAM4SAMv2
```

### Usage
Easy to run. For available models, see the [`Model Zoo`](https://github.com/SebastianJanampa/DAM4SAMv2/new/main?filename=README.md#model-zoo) section. 
```python
from dam4sam import DAM4SAM

tracker = DAM4SAM(model='sam21pp-L')
tracker.track(<video_file>)
```

#### `track` Function Parameters
This function runs the full tracking pipeline, handling everything from input validation to saving the final results.

* **`input_dir`** (str): **Required.** The path to the input video file or the directory containing your sequence of image frames.
* **`output_dir`** (str, optional): Specifies the directory where all output files (annotated frames, bounding boxes) will be saved. If `None`, the function runs in a visualization-only mode without saving files. Defaults to `None`.
* **`file_extension`** (str, optional): The file extension for the image frames (e.g., 'png', 'jpg') when `input_dir` is a directory. Defaults to `'jpg'`.
* **`fp16`** (bool, optional): If `True`, the model will use half-precision (FP16) for inference, which can improve performance on supported GPUs. Defaults to `True`.
* **`visualize`** (bool, optional): If `True`, a window will be displayed showing the live tracking process. Defaults to `True`.
* **`save_bboxes`** (bool, optional): If `True`, the bounding box coordinates for each tracked object are saved in `.txt` files, one for each frame. Defaults to `True`.
* **`save_frames`** (bool, optional): If `True`, the annotated frames with masks and bounding boxes will be saved as individual image files. Defaults to `True`.
* **`save_video`** (bool, optional): If `True`, a video will be generated from the annotated frames. This automatically enables `save_frames` if it's disabled. Defaults to `True`.
* **`out_video_name`** (str, optional): The desired filename for the output video. Defaults to `None`.
* **`exist_ok`** (bool, optional): If `False`, the function will create a new output directory with a numerical suffix (e.g., `output2`, `output3`) if the specified `output_dir` already exists. Defaults to `False`.

## Platforms
This code was tested on a Linux machine with access to a GPU. 
We know that not everyone may have access to a GPU. That's why we also built this repo to support the Colab Environment.
Nevertheless, if you use Colab, you will not be able to add points, but you will you will still retain all the other features. 
Our built-in COLAB GUI is also not very user-friendly, but we were unable to find an alternative solution since Colab does not support dynamic windows.

## Citation
If you use this repo, please cite the DAM4SAM article. And if you like this repo, we would appreciate a 🌟

```bibtex
@InProceedings{dam4sam,
  author = {Videnovic, Jovana and Lukezic, Alan and Kristan, Matej},
  title = {A Distractor-Aware Memory for Visual Object Tracking with {SAM2}},
  booktitle = {Comp. Vis. Patt. Recognition},
  year = {2025}
}
```

## Acknowledgement
This work is built upon [SAM2](https://github.com/facebookresearch/sam2/tree/main), [DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM), and [dam4sam-multi](https://github.com/danielkorth/dam4sam-multi).
The GUIs were developed using GEMINI from GOOGLE. Big thanks, Google, for creating such a great multi-modal model. 

✨ Feel free to contribute and reach out if you have any questions! ✨


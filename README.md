# Deep Illuminator 

Deep Illuminator is a data augmentation tool designed for image relighting. It can be used to easily and efficiently generate a wide range of illumination variants of a single image. It has been tested with several datasets and models and has been shown to succesfully improve performance. 
It has a built in visualizer created with [Steamlit](https://github.com/streamlit/streamlit) to preview how the target image can be relit. 

## Example Augmentations
<p align="center">
  <img src=assets/combined.gif>
</p>

## Usage
The simplest method to use this tool is through Docker Hub:

```
docker pull kartvel/deep-illuminator
```
#### Visualizer
Once you have the Deep Illuminator image run the following command to launch the visualizer: 

```
docker run -it --rm  --gpus all \
-p 8501:8501 --entrypoint streamlit \ 
kartvel/deep-illuminator run streamlit/streamlit_app.py
```
You will be able to interact with it on `localhost:8501`. 
Note: If you do not have NVIDIA gpu support enabled for docker simply remove the `--gpus all` option.

#### Generating Variants
It is possible to quickly generate multiple variants for images contained in a directory by using the following command:
```
docker run -it --rm --gpus all \                                                                                               ─╯
-v /path/to/input/images:/app/probe_relighting/originals \
-v /path/to/save/directory:/app/probe_relighting/output \
kartvel/deep-illuminator --[options]
```

#### Options

| Option  | Values |  Description  |  
| ------------- | ------------- | ------------- |
| mode   | ['synthetic', 'mid'] | Selecting the style of probes used as a relighting guide.| 
| step   | int | Increment for the granularity of relighted images. max mid: 24, max synthetic: 360| 

#### Buidling Docker image or running without a container
Please read the following for other options: [instructions](app/)
## Benchmarks
**Improved performance of [R2D2](https://github.com/naver/r2d2) for MMA@3 on [HPatches](https://hpatches.github.io)**

| Training Dataset  | Overall | Viewpoint  | Illumination | 
| ------------- | ------------- | ------------- | ------------- |
| COCO - Original   | 71.0 | 65.4  | 77.1  |
| COCO - Augmented  | 72.2 (+1.7%)  |  65.7 (+0.4%)  | 79.2 (+2.7%) |
| |  | | |
| VIDIT - Original   | 66.7 | 60.5  | 73.4   |
| VIDIT - Augmented  |  69.2 (+3.8%) |  60.9 (+0.6%)  | 78.1 (+6.4%)  |
| |  | | |
| Aachen - Original   | 69.4 | 64.1  | 75.0   |
| Aachen - Augmented  |  72.6 (+4.6%) |  66.1 (+3.1%) | 79.6 (+6.1%) |


**Improved performance of [R2D2](https://github.com/naver/r2d2) for the [Long-Term Visual Localization](https://www.visuallocalization.net) challenge on Aachen v1.1**

| Training Dataset  | 0.25m, 2° |  0.5m, 5°  | 5m, 10° | 
| ------------- | ------------- | ------------- | ------------- |
| COCO - Original   | 62.3 | 77.0  | 79.5  |
| COCO - Augmented  | 65.4 (+5.0%)  |  83.8 (+8.8%)  | 92.7 (+16%)  |
| |  | | |
| VIDIT - Original   | 40.8 | 53.4  | 61.3  |
| VIDIT - Augmented  |   53.9 (+32%) |  71.2 (+33%) | 83.2(+36%)  |
| |  | | |
| Aachen - Original   | 60.7 | 72.8  | 83.8   |
| Aachen - Augmented  |  63.4 (+4.4%) |  81.7 (+12%) | 92.1 (+9.9%) |


### Acknowledgment

The developpement of the VAE for the visualzier was made possible by the [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE) repository. 

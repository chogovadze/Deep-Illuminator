Main README: [here](https://github.com/chogovadze/Deep-Illuminator/)
## Building Image

First, to build the docker image, it is necessary to download the [model](https://drive.google.com/file/d/1lTuHvxWtvPuaOYGInYKGE58Zpe1uOcup/view?usp=sharing) and the [probes](https://drive.google.com/file/d/117_sd2l0ZhRxuXPQCH6bMZmt_bG-RxkC/view?usp=sharing). The files should be transfered in the following manner:
```
                    
├── probe_relighting     
│   ├── checkpoint.ckpt
│   ├── data
│   │   ├── mid_probes
│   │   ├── point_10kW
│   │   ├── ...
│   │   └── 
│   ├──...
│   └── 
├── ...       
└── ..           

```
Once the folder structure is complete the image can be built.

## Without container

To use the repository without containerizing, the same structure should be set up. 
Navigate to `./app/` and run:

`python3 -m pip install -e .` 

Followed with:

`python3 -m pip install -r probe_relighting/requirements.txt`

#### Streamlit

Streamlit can be simply launched by navigating to `./app/probe_relighting/` and running the command:

`streamlit run streamlit/streamlit_app.py`

#### Generating Variants

Similarly variants can be generated from `./app/probe_relighting/` by using:

`python3 generate_images.py --[options]`

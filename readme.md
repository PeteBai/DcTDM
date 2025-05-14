# DcTDM: Dual-conditioned Temporal Diffusion Modeling

Implementation of our Dual-conditioned Temporal Diffusion Model. The paper is accepted at ICRA 2025. This repository is based on the existing work MakeLongVideo[⇱](https://github.com/xuduo35/MakeLongVideo) and ControlVideo[⇱](https://github.com/thu-ml/controlvideo) on GitHub.

## Dataset
We have two partitions: DrivingSceneDDM (~104GB), and DrivingSceneDDM-XL (~510GB). We trained our model on both.

### Download
- DrivingSceneDDM: [RGB+Depth](https://zzzura-secure.duckdns.org/downloads/dataset.zip), [Canny Edge](https://zzzura-secure.duckdns.org/downloads/canny.zip)
- DrivingSceneDDM-XL: [RGB](https://zzzura-secure.duckdns.org/downloads/DSDDM_XL_rgb.zip), [Depth Maps](https://zzzura-secure.duckdns.org/downloads/DSDDM_XL_depth.zip), [Canny Edge](https://zzzura-secure.duckdns.org/downloads/DSDDM_XL_canny.zip)

You will need to manually place them into the correct folder:
```shell
- dataset
|-images
 |-carla_01
 |-...
 |-B0 (from XL)
 |-...
|-depth
 |-carla_01
 |-...
|-canny


```

## Setup

This repository has been tested on Ubuntu 20.04 with CUDA 12. First, make sure CUDA is installed. Install ```torch==2.0.1``` based on your machine and GPU configurations. Then, execute the following commands.

```shell
pip install -r requirements.txt
```

## Dataset Preperation

Due to the size of the training dataset, you will need to compile the video dataset yourself. Under ```dataset_scripts``` you will find the scripts to turn image datasets to videos for training purpose. **Note:** you don't need to prepared the dataset if you are using the model for inference only.

1. Modify the ```dataset_loc``` varible in ```dataset_scripts/gen_video_multithread.py``` to the location where you downloaded the dataset.

1. Create the videos with ```python ./dataset_scripts/gen_video_multithread.py```.

1. If needed, you can change the storage location of dataset index file ```train.txt``` by modifying ```txt_file_loc``` varible in ```dataset_scripts/gen_train_list.py```.

1. Create the dataset index file ```train.txt``` by  executing ```python ./dataset_scripts/gen_train_list.py```.

## Backbone Preperation

Download ```control_sd15_depth``` and ```control_sd15_canny``` from Hugging Face[⇱](https://huggingface.co/lllyasviel/ControlNet/tree/main) and store the them to ```checkpoints/ControlNet/models``` folder.
The checkpoint folder should read ```checkpoints/ControlNet/models/control_sd15_depth``` and ```checkpoints/ControlNet/models/control_sd15_canny```. Download ```stable-diffusion-v1-5```[⇱](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) and put it in ```checkpoint``` folder as well.

## Model Training and Fine-tuning

- Modify the amount of GPU (as noted by ```num_processes```) you have in ```./configs/multigpu.yaml```.

- Modify the location of train data, output path, validation data in ```./configs/makelongvideo.yaml```.

- To start training, run:

```shell
accelerate launch --config_file ./configs/multigpu.yaml train.py --config configs/makelongvideo.yaml
```


- Additionally, if you wish to start with an existing (downloaded) checkpoint, you need to unwrap that checkpoint first by executing

```shell
TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch train.py --config configs/makelongvideo.yaml --unwrap <your-checkpoint-path>
```

## Model Inference

You can either use your own checkpoints or a downloaded one. Don't forget to **unwrap** the checkpoint before any inference! To download our pre-trained checkpoints, visit this [website](https://www.bing.com/search?q=what+should+i+do+if+i+can%27t+find+a+pretrained+checkpoint+reddit&FORM=AWRE). **Note 1:** You will need a series of depth file for the inference. To learn how to estimate depth from a video, continue reading. **Note 2:** You ight need to edit ```pretrained_model_path``` in ```infer.py``` if you unwrapped the checkpoint to a different location than ```./outputs/makelongvideo/```.

In ```infer.py``` you will find all sorts of options:

- ```--prompt```: Your input prompt

- ```--n_frames```: The number of frames to generate. Longer videos take more time and GPU memory. The default length is 24.

- ```--depth_start_file```: The location to the first depth frame (*.npy) of the video. All frames should be named as a 6 digit number with leading zeros.

- ```--canny_start_file```: The location to the first canny frame (*.png) of the video. All frames should be named as a 6 digit number with leading zeros.

- ```--batch```: An integer that enables generating videos in batch mode, the number is the # of videos generated

- ```--width``` and ```--height```: Default to 256.

- ```--out_folder```: Name of the sub folder in ```./outputs/result/```

- ```--full_edit```: Generate all the supported styles for the provided depth files.

- ```--config_file```: The path to the generation config text file that lists a series of depth start files and prompts.

- ```--do_gif```: While generating results also include a gif file for quick demostration.

- ```--do_depth``` and ```--do_canny``` is for when you want to only use depth or canny edge data to guide the generation.

- ```--my_model_path```: Load a specific pre-trained checkpoint, required for inference.

For example:
```shell
python infer.py --width 256 --height 256 --config_file ./configs/makeicraexp.txt --prompt "A vehicle driving in-road in a small town with a river and several bridges; rainly, cloudy town surroundings." --depth_start_file /work/zura-storage/Data/DrivingSceneDDM/depth/carla_01/000000.npy --n_frames 96 --out_folder icraexp --do_gif --do_depth --do_canny --canny_start_file /work/zura-storage/Data/DrivingSceneDDM/canny/carla_01/000000.png --my_model_path ./outputs/slam_synthetic2/checkpoint-2326
```

## Depth estimation

The ```midas_pipeline``` folder offers tools to generate corresponding dense depth map. You will need to first download the MiDaS pre-trained checkpoint[⇱](https://huggingface.co/ckpt/ControlNet/blob/main/dpt_hybrid-midas-501f0c75.pt) to ```midas_pipeline/midas_models```. Then, modify the ```parent_loc``` in ```midas_pipeline/test_midas.py``` to the root folder that **only** contains the datasets you wish to estimate in seperate folders. Execute ```python midas_pipeline/test_midas.py``` and you will find a newly created ```depth``` folder that contains all the depth npy files you wish.

## Canny edge estimation
We use the built in ```cv2.Canny``` function to estimate the canny edge from RGB images. The script is provided in ```dataset_scripts\gen_canny_edge.py```. Modify the ```parent_loc``` variable in the beginning of the file and point it to your folder that contains images you wish to work on.

## Citation

```bibtex
@inproceedings{bai2024dctdm,
    author = {Bai, Xiangyu and Jiang, Le and Luo, Yedi and Singh, Hanumant and Ostadabbas, Sarah},
    title = {Dual-conditioned Temporal Diffusion Modeling for Driving Scene Generation},
    booktitle = {IEEE International Conference of Robotics and Automation (ICRA) Under Review},
    year = {2025}
}
```
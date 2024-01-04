# face_animation
# FACE IMAGE ANIMATION BASED ON FIRST ORDER MOTION MODEL

Given a source image and  driving video, animates the source image in accordance with head motion and facial expressions from driving video 

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Option: Linux

### Installation
 <br>

1. Clone repo

    ```bash
    git clone https://github.com/vedant-916/face_animation.git
    cd face_animation
    ```

2. Install dependent packages

    ```bash
    pip install -r requirements.txt

    ## Install the Face Alignment lib
    pip install face-alignment
    ```


You can use demo.py to run the model on your own data after training your own model


The result will be stored in ```result.mp4```. The driving videos and source images should be cropped before it can be used in our method. To obtain some semi-automatic crop suggestions you can use ```python crop-video.py --inp some_youtube_video.mp4```. It will generate commands for crops using ffmpeg. 




## :computer: Training


### Datasets
 
1) **VoxCeleb**. Please follow the instruction from https://github.com/AliaksandrSiarohin/video-preprocessing.

### Train on VoxCeleb
To train a model on specific dataset run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_addr="0.0.0.0" --master_port=12348 run.py --config config/vox-adv-256.yaml --name DaGAN --rgbd --batchsize 12 --kp_num 15 --generator DepthAwareGenerator
```
<div id="dataparallel" >Or</div>

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_dataparallel.py --config config/vox-adv-256.yaml --device_ids 0,1,2,3 --name DaGAN_voxceleb2_depth --rgbd --batchsize 48 --kp_num 15 --generator DepthAwareGenerator
```


<!-- CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --master_addr="0.0.0.0" --master_port=12348 run.py --config config/vox-adv-256.yaml --name SpadeDaGAN --rgbd --batchsize 6 --kp_num 15 --generator SPADEDepthAwareGenerator -->

The code will create a folder in the log directory (each run will create a new name-specific directory).
Checkpoints will be saved to this folder.
To check the loss values during training see ```log.txt```.
By default the batch size is tunned to run on 8 GeForce RTX 3090 gpu (You can obtain the best performance after about 150 epochs). You can change the batch size in the train_params in ```.yaml``` file.


Also, you can watch the training loss by running the following command:
```bash
tensorboard --logdir log/DaGAN/log
```
When you kill your process for some reasons in the middle of training, a zombie process may occur, you can kill it using our provided tool:
 ```bash
python kill_port.py PORT
```

### Training on your own dataset
1) Resize all the videos to the same size e.g 256x256, the videos can be in '.gif', '.mp4' or folder with images.
We recommend the later, for each video make a separate folder with all the frames in '.png' format. This format is loss-less, and it has better i/o performance.

2) Create a folder ```data/dataset_name``` with 2 subfolders ```train``` and ```test```, put training videos in the ```train``` and testing in the ```test```.

3) Create a config ```config/dataset_name.yaml```, in dataset_params specify the root dir the ```root_dir:  data/dataset_name```. Also adjust the number of epoch in train_params.



## :scroll: Acknowledgement

 Our DaGAN implementation is inspired by [FOMM](https://github.com/AliaksandrSiarohin/first-order-model). We appreciate the authors of [FOMM](https://github.com/AliaksandrSiarohin/first-order-model) for making their codes available to public.

## :scroll: BibTeX

```
@inproceedings{hong2022depth,
            title={Depth-Aware Generative Adversarial Network for Talking Head Video Generation},
            author={Hong, Fa-Ting and Zhang, Longhao and Shen, Li and Xu, Dan},
            journal={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
            year={2022}
          }

@article{hong2023dagan,
            title={DaGAN++: Depth-Aware Generative Adversarial Network for Talking Head Video Generation},
            author={Hong, Fa-Ting and and Shen, Li and Xu, Dan},
            journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
            year={2023}
          }
```

### :e-mail: Contact

If you have any question or collaboration need (research purpose or commercial purpose), please email `fhongac@cse.ust.hk`.

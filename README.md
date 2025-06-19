# AVH-Align

[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2412.00175)

**Official PyTorch Implementation of the Paper:**

> **Ștefan Smeu, Dragoș-Alexandru Boldisor, Dan Oneață and Elisabeta Oneață**  
> [Circumventing shortcuts in audio-visual deepfake detection datasets with unsupervised learning](https://arxiv.org/abs/2412.00175)  
> *CVPR, 2025*

## Data

To set up your data, follow these steps:

**Download the datasets:**
   - **AV-Deepfake1M(AV1M) Dataset:** Follow instructions from [AV-Deepfake1M](https://github.com/ControlNet/AV-Deepfake1M)
   - **FakeAVCeleb Dataset:** Follow instructions from [FakeAVCeleb GitHub repo](https://github.com/DASH-Lab/FakeAVCeleb)
   - **AVLips Dataset:** Follow instructions from [LipFD GitHub repo](https://github.com/AaronComo/LipFD)

## Set-up AV-Hubert
```bash 
# clone/install AV-Hubert
git clone https://github.com/facebookresearch/av_hubert.git
cd av_hubert/avhubert
git submodule init
git submodule update
cd ../fairseq
pip install --editable ./
cd ../avhubert
# install additional files for AV-Hubert
mkdir -p content/data/misc/
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O content/data/misc/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d content/data/misc/shape_predictor_68_face_landmarks.dat.bz2
wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy -O content/data/misc/20words_mean_face.npy
cd ../../
# moving our feature extraction files into avhubert space
cp deepfake_feature_extraction.py av_hubert/avhubert/deepfake_feature_extraction.py 
cp deepfake_preprocess.py av_hubert/avhubert/deepfake_preprocess.py

# download avhubert checkpoint
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt
mv self_large_vox_433h.pt av_hubert/avhubert/self_large_vox_433h.pt
```

This repository also integrates code from the following repositories:
- [FACTOR](https://github.com/talreiss/FACTOR)
- [AV-Hubert](https://github.com/facebookresearch/av_hubert)

## Installation

Main prerequisites:

* `Python 3.10.14`
* `pytorch=2.2.0` (older version for compability with AVHubert)
* `pytorch-cuda=12.4`
* `lightning=2.4.0`
* `torchvision>=0.17`
* `scikit-learn>=1.3.2`
* `pandas>=2.1.1`
* `numpy>=1.26.4`
* `pillow>=10.0.1`
* `librosa>=0.9.1`
* `dlib>=19.24.9`
* `skvideo>=1.1.10`
* `ffmpeg>=4.3`

## Feature extraction

1. **Preprocess video files**
Run deepfake_preprocess.py from av_hubert/avhubert. Example for AV-Deepfake1M
```bash
python deepfake_preprocess.py \
    --dataset AV1M \
    --split train \
    --metadata /av1m_metadata/train_metadata.csv \
    --data_path /path/to/AV1M_root \
    --save_path /path/to/save/output_videos_and_audio
```
and FakeAVCeleb
```bash
python deepfake_preprocess.py \
    --dataset FakeAVCeleb \
    --metadata /path/to/FakeAVCeleb_metadata.csv \
    --data_path /path/to/FakeAVCeleb_root \
    --save_path /path/to/save/output_videos_and_audio
    --category all \
```

2. **Extract features**
Run deepfake_feature_extraction.py from av_hubert/avhubert. Example for AV-Deepfake1M

```bash
python deepfake_feature_extraction.py \
    --dataset AV1M \
    --split train \
    --metadata /av1m_metadata/train_metadata.csv \
    --ckpt_path self_large_vox_433h.pt \
    --data_path /path/to/preprocessed/data \
    --save_path /path/to/save/features
```
and FakeAVCeleb
```bash
python deepfake_feature_extraction.py \
    --dataset FakeAVCeleb \
    --metadata /path/to/FakeAVCeleb_metadata.csv \
    --ckpt_path self_large_vox_433h.pt \
    --data_path /path/to/preprocessed/data \
    --save_path /path/to/save/features \
    --category all
```

add ```--trimmed``` for the trimmed version of features

## Train

 To train the models mentioned in the article, follow:

 **Set up training and validation data paths** in `config.py` or specify them as arguments when running the training routine as the example below:

 ```bash
 python train.py --name=<experiment_name> --data_root_path=<path_to_the_features_data> --metadata_root_path=<path_to_the_folder_containing_the_dataset_metadata_files>
 ```
 The model weights will be available at `<save_path>/<name>.pt`

## Pretrained Models
We provide weights for our AVH-Align model trained on 45000 real videos from AV-Deepfake1M in `checkpoints/AVH-Align_AV1M.pt`.

## Evaluation

To evaluate a model, use/modify the following example:

```bash 
python eval.py \ 
    --checkpoint_path checkpoints/AVH-Align_AV1M.pt \ 
    --features_path /path/to/saved/features \ 
    --metadata /av1m_metadata/test_metadata.csv \ 
    --dataset AV1M 
```

## License

<p xmlns:cc="http://creativecommons.org/ns#">The code is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0 <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>

## Citation

If you find this work useful in your research, please cite it.

```
@InProceedings{AVH-Align,
    author    = {Smeu, Stefan and Boldisor, Dragos-Alexandru and Oneata, Dan and Oneata, Elisabeta},
    title     = {Circumventing shortcuts in audio-visual deepfake detection datasets with unsupervised learning localization},
    booktitle = {Proceedings of The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025}
}
```

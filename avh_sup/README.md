# AVH-Align/sup

For data preprocessing, AV-HuBERT setup, installation and how to extract features, check the main README.md.

## Train

To train the models mentioned in the article, follow:
 **Complete the config file** in `configs/train_config.yaml` there are several parameters that need to be specified:
 * `data_info.name` - choose between AV1M (AV-Deepfake1M) and FAVC (FakeAVCeleb)
 * `data_info.root_path` - root path where features are saved
 * `csv_root_path` - path to the folder containing split csv's (you can find it in `AVH-Align/avh_sup/csv_metadata/av1m` or `AVH-Align/avh_sup/csv_metadata/favc`)
 
 You can also specify where the logs and checkpoints should be saved (check `callbacks.logger.log_path` and `callbacks.ckpt_args.ckpt_dir`).
 There are also some other parameters that can be set (the default values found in config were used to train our models).

After that, run the following command:
```bash
python train_test.py --config=<path_to_the_train_config_file>
```
As explained previously, you can use the already defined YAML file (`configs/train_config.yaml`).

## Pretrained Models
We provide weights for our AVH-Align/sup models, one trained on AV1M (`checkpoints/avh_sup/AVH_Sup_AV1M.ckpt`) and another on FAVC (`checkpoints/avh_sup/AVH_Sup_FAVC.ckpt`).

## Evaluation

To evaluate a model, use the config file `configs/test_config.yaml`. The parameters of `data_info` are similar to the ones found in `configs/train_config.yaml`. Complete also the `ckpt_path` (path to the checkpoint model) and `output_path` (path to the output folder).

Then, run the following command:

```bash 
python train_test.py --test --config=<path_to_the_train_config_file>
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
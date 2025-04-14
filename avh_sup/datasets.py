import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


###### AV1M ######

class AV1M_trainval_dataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        self.avhubert_model = self.config["avhubert_model"]

        self.root_path = self.config["root_path"]
        self.csv_root_path = self.config["csv_root_path"]

        self.df = pd.read_csv(os.path.join(self.csv_root_path, f"{self.split}_labels.csv"))
        self.feats_dir = os.path.join(self.root_path, "av_hubert_features", self.avhubert_model, self.split)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feats = np.load(os.path.join(self.feats_dir, row["path"][:-4] + ".npz"), allow_pickle=True)
        label = int(row["label"])

        video = feats['visual']
        audio = feats['audio']

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        return torch.tensor(video), torch.tensor(audio), label, row["path"][:-4] + ".npz"  # video, audio, label, path


class AV1M_test_dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.avhubert_model = config["avhubert_model"]
        self.csv_root_path = self.config["csv_root_path"]
        self.root_path = os.path.join(config["root_path"], self.avhubert_model, "test_features")

        self.paths = np.load(os.path.join(self.root_path, "paths.npy"), allow_pickle=True)
        self.audio_feats = np.load(os.path.join(self.root_path, "audio.npy"), allow_pickle=True)
        self.video_feats = np.load(os.path.join(self.root_path, "video.npy"), allow_pickle=True)

        self.labels = self._get_labels()
    
    def _get_labels(self):
        df = pd.read_csv(os.path.join(self.csv_root_path, "test_labels.csv"))
        labels = {}
        for path in self.paths:
            row = df.loc[df['path'] == path]
            if len(row.index) != 1:
                raise ValueError("Multiple or no entries in test_labels.csv for a single path!")
            row = row.iloc[0]
            labels[path] = int(row['label'])

        return labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        video = self.video_feats[idx]
        audio = self.audio_feats[idx]

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        label = self.labels[path]

        return torch.tensor(video), torch.tensor(audio), label, path  # video, audio, label, path


###### ######

###### AVLips ######

class AVLips_Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root_path = config["root_path"]
        self.avhubert_model = self.config["avhubert_model"]

        dir_path_real = os.path.join(self.root_path, self.avhubert_model, "0_real")
        dir_path_fake = os.path.join(self.root_path, self.avhubert_model, "1_fake")

        paths_real = np.load(os.path.join(dir_path_real, "paths.npy"), allow_pickle=True)
        paths_fake = np.load(os.path.join(dir_path_fake, "paths.npy"), allow_pickle=True)

        self.labels = np.concatenate((np.zeros(paths_real.shape[0]), np.ones(paths_fake.shape[0]))).astype(np.int32)
        self.paths = np.concatenate((paths_real, paths_fake))
        self.audio_feats = np.concatenate((
            np.load(os.path.join(dir_path_real, "audio.npy"), allow_pickle=True),
            np.load(os.path.join(dir_path_fake, "audio.npy"), allow_pickle=True),
        ))
        self.video_feats = np.concatenate((
            np.load(os.path.join(dir_path_real, "video.npy"), allow_pickle=True),
            np.load(os.path.join(dir_path_fake, "video.npy"), allow_pickle=True),
        ))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video = self.video_feats[idx]
        audio = self.audio_feats[idx]
        path = self.paths[idx]
        label = self.labels[idx]

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        return torch.tensor(video), torch.tensor(audio), label, path  # video, audio, label, path


###### ######

###### FakeAVCeleb ######

class FakeAVCeleb_Dataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.avhubert_model = self.config["avhubert_model"]
        self.root_path = self.config["root_path"]
        self.csv_root_path = self.config["csv_root_path"]
        dir_path = os.path.join(self.root_path, "avhubert_checkpoints", self.avhubert_model)

        labels = pd.read_csv(os.path.join(self.csv_root_path, f"{split}_split.csv"))

        self.videos, self.audios, self.paths = np.array([]), np.array([]), np.array([])

        for folder_name in os.listdir(dir_path):
            vids = np.load(os.path.join(dir_path, folder_name, "video.npy"), allow_pickle=True)
            self.videos = np.concatenate((self.videos, vids))

            auds = np.load(os.path.join(dir_path, folder_name, "audio.npy"), allow_pickle=True)
            self.audios = np.concatenate((self.audios, auds))

            ps = np.load(os.path.join(dir_path, folder_name, "paths.npy"), allow_pickle=True)
            self.paths = np.concatenate((self.paths, ps))

        self.useful_data = []
        for idx in labels.index:
            row = labels.iloc[idx]
            path = row['full_path'].replace("FakeAVCeleb/", "")
            label = int(row['category'] != 'A')

            for id_path in range(len(self.paths)):
                if self.paths[id_path] == path:
                    self.useful_data.append((id_path, label))
                    break

    def __len__(self):
        return len(self.useful_data)

    def __getitem__(self, idx):
        id_path, label = self.useful_data[idx]
        video = self.videos[id_path]
        audio = self.audios[id_path]
        path = self.paths[id_path]

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        return torch.tensor(video), torch.tensor(audio), label, path  # video, audio, label, path


###### ######

def load_data(config, test=False):
    if test:
        if config["name"] == "AV1M":
            test_ds = AV1M_test_dataset(config)
        elif config["name"] == "AVLips":
            test_ds = AVLips_Dataset(config)
        elif config["name"] == "FAVC":
            test_ds = FakeAVCeleb_Dataset(config, split="test")
        else:
            raise ValueError("Dataset name error. Expected: AV1M, AVLips, FAVC; Got: " + config["name"])

        test_dl = DataLoader(test_ds, shuffle=False, batch_size=1)

        return test_dl

    else:
        if config["name"] == "AV1M":
            train_ds = AV1M_trainval_dataset(config, split="train")
            val_ds = AV1M_trainval_dataset(config, split="val")
        elif config["name"] == "FAVC":
            train_ds = FakeAVCeleb_Dataset(config, split="train")
            val_ds = FakeAVCeleb_Dataset(config, split="val")
        else:
            raise ValueError("Dataset name error. Expected: AV1M, FAVC; Got: " + config["name"])

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=1)
        val_dl = DataLoader(val_ds, shuffle=False, batch_size=1)

        return train_dl, val_dl

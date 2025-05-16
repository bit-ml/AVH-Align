import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import pandas as pd
import os

class FeatureDataset(IterableDataset):
    def __init__(self, metadata_path, features_root_path, tau=15, audio_dim=1024):
        super().__init__()
        self.tau = tau
        self.features_root_path = features_root_path
        self.audio_dim = audio_dim
        self.metadata = pd.read_csv(metadata_path)
        self.num_videos = len(self.metadata)
        self.num_frames = [self.metadata.iloc[i]['num_frames'] for i in range(len(self.metadata))]
        
        self.paths = self.metadata['path'].apply(lambda x: os.path.join(self.features_root_path, x.replace(".mp4", ".npz"))).tolist()

    def __len__(self):
        return sum(self.num_frames)  # Total number of frames across all videos

    def _load_temporal_window(self, audio_feat, video_idx, frame_idx):
        num_frames = self.num_frames[video_idx]
        audio_window = []
        
        for t in range(frame_idx - self.tau, frame_idx + self.tau + 1):
            if 0 <= t < num_frames:
                audio_feature = torch.from_numpy(audio_feat[t]).float()
                audio_feature = audio_feature / (torch.linalg.norm(audio_feature, ord=2, dim=-1, keepdim=True))
            else:
                audio_feature = torch.zeros(self.audio_dim)
            audio_window.append(audio_feature)
            
        return torch.stack(audio_window).float()

    def _load_features(self, video_idx):
        feature = np.load(self.paths[video_idx], allow_pickle=True, mmap_mode='r')
        return feature['visual'], feature['audio']

    def _get_worker_videos(self):
        """
        Multi-worker data partition
        """
        worker_info = get_worker_info()
        if worker_info is None:
            # Single worker case (no multiprocessing)
            return range(self.num_videos)
        
        # Multi-worker case: Split videos across workers
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
        return range(worker_id, self.num_videos, num_workers)

    def __iter__(self):
        worker_videos = self._get_worker_videos()
        
        for video_idx in worker_videos:
            try:
                feature_visual, feature_audio = self._load_features(video_idx)
                num_frames = len(feature_visual)

                for local_frame_idx in range(num_frames):
                    visual_tensor = torch.from_numpy(feature_visual[local_frame_idx]).float()
                    visual_tensor = visual_tensor / (torch.linalg.norm(visual_tensor, ord=2, dim=-1, keepdim=True))
                    
                    # extract audio neighborhood centered at local_frame_idx
                    audio_tensor = self._load_temporal_window(feature_audio, video_idx, local_frame_idx)
                    
                    yield visual_tensor, audio_tensor, video_idx, local_frame_idx
            except Exception as e:
                print(f"Error loading video {video_idx}: {e}")
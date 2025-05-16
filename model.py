import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, visual_dim=1024, audio_dim=1024, hidden_dim=1024):
        super(FusionModel, self).__init__()
        
        self.visual_proj = nn.Linear(visual_dim, hidden_dim // 2)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim // 2)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, visual_features, audio_features):
        # Project visual and audio features separately and concatenate
        visual_proj = self.visual_proj(visual_features)
        audio_proj = self.audio_proj(audio_features)
        fused_features = torch.cat((visual_proj, audio_proj), dim=-1)
        
        output = self.mlp(fused_features)
            
        return output

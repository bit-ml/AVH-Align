import lightning as L
import torch
import torch.nn as nn


class AVH_Sup(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config["model_hparams"]
        self.model_type = self.config["model_type"]
        self.input_type = self.config["input_type"]

        self.save_hyperparameters()

        if self.model_type == "linear":
            if self.input_type == "both":
                self.head = nn.Linear(2 * 1024, 1)
            else:
                self.head = nn.Linear(1024, 1)
        elif self.model_type == "mlp":
            hidden_dim = 1024
            if self.input_type == "both":
                self.visual_proj = nn.Linear(1024, hidden_dim // 2)
                self.audio_proj = nn.Linear(1024, hidden_dim // 2)
            else:
                self.proj = nn.Linear(1024, hidden_dim)
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

    def forward(self, input_feats):
        video_feats, audio_feats = input_feats[0], input_feats[1]

        if self.model_type == "linear":

            if self.input_type == "both":
                fused_features = torch.cat((video_feats, audio_feats), dim=-1)
            elif self.input_type == "audio":
                fused_features = audio_feats
            elif self.input_type == "video":
                fused_features = video_feats
            else:
                raise ValueError(f"Error! Input type: {self.input_type}")

            output = self.head(fused_features)[:, :, 0]

            return torch.logsumexp(output, dim=-1)

        elif self.model_type == "mlp":
            if self.input_type == "both":
                visual_proj = self.visual_proj(video_feats)
                audio_proj = self.audio_proj(audio_feats)

                fused_features = torch.cat((visual_proj, audio_proj), dim=-1)
            elif self.input_type == "audio":
                fused_features = self.proj(audio_feats)
            elif self.input_type == "video":
                fused_features = self.proj(video_feats)

            output = self.mlp(fused_features)[:, :, 0]

            return torch.logsumexp(output, dim=-1)

    def predict_scores(self, video_feats, audio_feats):
        scores = self.forward((video_feats, audio_feats))
        return scores

    def training_step(self, batch, batch_idx):
        video_feats, audio_feats, labels, _ = batch

        output = self.forward((video_feats, audio_feats))
        score = output.unsqueeze(1)
        score = torch.cat((-score, score), 1)

        loss = torch.nn.functional.cross_entropy(score, labels)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        video_feats, audio_feats, labels, _ = batch

        output = self.forward((video_feats, audio_feats))
        score = output.unsqueeze(1)
        score = torch.cat((-score, score), 1)

        loss = torch.nn.functional.cross_entropy(score, labels)
        self.log("val_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return [optimizer], []

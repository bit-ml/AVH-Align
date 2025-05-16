import argparse
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
import os

from model import FusionModel
from utils import seed_run


def process_video(data, fusion_model, device):
    visual_tensor = torch.from_numpy(data["visual"]).to(device)
    audio_tensor = torch.from_numpy(data["audio"]).to(device)

    # L2 norm
    visual_tensor = visual_tensor / (torch.linalg.norm(visual_tensor, ord=2, dim=-1, keepdim=True))
    audio_tensor = audio_tensor / (torch.linalg.norm(audio_tensor, ord=2, dim=-1, keepdim=True))

    output = fusion_model(visual_tensor, audio_tensor)
    score = torch.logsumexp(-output, dim=0).detach().cpu().squeeze()

    return score


def main(args):
    seed_run()

    print(f"Evaluating AVH-Align on {args.dataset} with pretrained weights saved at {args.checkpoint_path} ...")

    # Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_model_weights = torch.load(args.checkpoint_path, weights_only=False)

    fusion_model = FusionModel().to(device)
    fusion_model.load_state_dict(fusion_model_weights["state_dict"])
    fusion_model.eval()
    
    # Load metadata for access to labels
    metadata = pd.read_csv(args.metadata)

    outputs = []
    ground_truths = []
    for _, row in tqdm(metadata.iterrows()):
        data = np.load(os.path.join(args.features_path, row["path"].replace(".mp4", ".npz")), allow_pickle=True)
        label = row["label"]
        score = process_video(data, fusion_model, device)
        outputs.append(score)
        ground_truths.append(label)

    outputs = np.array(outputs)
    ground_truths = np.array(ground_truths)

    auc = roc_auc_score(ground_truths, outputs)
    ap = average_precision_score(ground_truths, outputs)

    print(f"AP: {ap}")
    print(f"AUC: {auc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fusion Model on Deepfake Dataset")

    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/AVH-Align_AV1M.pt",
                        help="Path to the pretrained fusion model checkpoint.")
    parser.add_argument("--features_path", type=str,
                        default=f"av1m_features/val/",
                        help="Path to the root folder of test data.")
    parser.add_argument("--metadata", type=str,
                        default="av1m_metadata/test_metadata.csv",
                        help="CSV file containing ground truth labels.")
    parser.add_argument("--dataset", type=str, default="AV1M",
                        help="Dataset name")

    args = parser.parse_args()
    main(args)

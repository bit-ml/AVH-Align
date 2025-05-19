# this file incorporates code from Reiss et al. FACTOR(https://github.com/talreiss/FACTOR)

import argparse
import os
import subprocess
import numpy as np
import cv2
import csv
import dlib
import skvideo.io
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg

# Backward compatibility of np.float and np.int
np.float = np.float64
np.int = np.int_

# Constants for both datasets
FACE_PREDICTOR_PATH = "content/data/misc/shape_predictor_68_face_landmarks.dat"
MEAN_FACE_PATH = "content/data/misc/20words_mean_face.npy"
STD_SIZE = (256, 256)
STABLE_PNTS_IDS = [33, 36, 39, 42, 45]

def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def preprocess_video(input_video_dir, video_filename, output_video_dir, face_predictor_path, mean_face_path):
    # skip if file already exists
    if not os.path.exists(os.path.join(output_video_dir, video_filename[:-4] + '_roi.mp4')):
        os.makedirs(output_video_dir, exist_ok=True)
    else:
        return True

    input_path = os.path.join(input_video_dir, video_filename)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    mean_face_landmarks = np.load(mean_face_path)
    try:
        videogen = skvideo.io.vread(input_path)
    except:
        print(f"Failed to read video: {input_path}")
        return False
    
    frames = np.array([frame for frame in videogen])
    landmarks = [detect_landmark(frame, detector, predictor) for frame in frames]
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    
    try:
        rois = crop_patch(input_path, preprocessed_landmarks, mean_face_landmarks, STABLE_PNTS_IDS, STD_SIZE, 
                        window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
    except:
        print(f"Failed to preprocess video: {input_path}; passing whole video")
        rois = frames[..., ::-1]

    roi_path = os.path.join(output_video_dir, video_filename[:-4] + '_roi.mp4')
    audio_fn = os.path.join(output_video_dir, video_filename[:-4] + '.wav')
    write_video_ffmpeg(rois, roi_path, "/usr/bin/ffmpeg")
    
    subprocess.run([
        "/usr/bin/ffmpeg",
        "-i", input_path,
        "-f", "wav",
        "-vn",
        "-y", audio_fn,
        "-loglevel", "quiet"
    ])
    return True

def process_av1m(metadata_file_path, path_to_images_root, save_path, max_workers):
    with open(metadata_file_path, "r") as f, ProcessPoolExecutor(max_workers=max_workers) as executor:
        reader = csv.DictReader(f)

        futures = {
            executor.submit(
                preprocess_video,
                path_to_images_root,
                row['path'],
                save_path,
                FACE_PREDICTOR_PATH,
                MEAN_FACE_PATH
            ): (path_to_images_root, row['path'])
            for row in reader
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing... "):
            input_dir, filename = futures[future]
            try:
                result = future.result()
                if not result:
                    print(f"[WARN] Failed to process video: {os.path.join(input_dir, filename)}")
            except Exception as e:
                print(f"[ERROR] Error in video {os.path.join(input_dir, filename)}: {e}")


def process_fakeavceleb(category, metadata_file_path, input_root, save_path, max_workers):
    # Load metadata CSV and filter by category
    selected_videos = []
    with open(metadata_file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] == category:
                original_file_path = row["path"].replace("FakeAVCeleb/", "")
                filename = row["filename"]
                input_dir = os.path.join(input_root, original_file_path)
                output_dir = os.path.join(save_path, original_file_path)
                selected_videos.append((input_dir, filename, output_dir))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                preprocess_video,
                input_dir,
                filename,
                output_dir,
                FACE_PREDICTOR_PATH,
                MEAN_FACE_PATH
            ): (input_dir, filename)
            for input_dir, filename, output_dir in selected_videos
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {category}..."):
            input_dir, filename = futures[future]
            try:
                result = future.result()
                if not result:
                    print(f"[WARN] Failed to process video: {os.path.join(input_dir, filename)}")
            except Exception as e:
                print(f"[ERROR] Error in video {os.path.join(input_dir, filename)}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess videos for FakeAVCeleb or AV1M dataset")
    parser.add_argument('--dataset', default='AV1M', help='Select dataset: FakeAVCeleb (favc) or AV1M (av1m)')
    parser.add_argument('--split', default='train', help='For AV1M: data split to process (e.g., val, train)')
    parser.add_argument("--metadata", type=str, default="av1m_metadata/train_metadata.csv", help="Path to the dataset metadata")
    parser.add_argument('--category', choices=['RealVideo-RealAudio', 'RealVideo-FakeAudio', 'FakeVideo-RealAudio', 'FakeVideo-FakeAudio'], default='all', help='For FakeAVCeleb: select category (RealVideo-RealAudio, etc.)')
    parser.add_argument('--data_path', default="av1m/", help='Path to the dataset root folder')
    parser.add_argument('--max_workers', type=int, default=32, help='Number of parallel workers (default: number of CPU cores)')
    parser.add_argument('--save_path', default="av1m_preprocessed/", help='Path to save avhubert prerpocess outputs (lips crop)')
    args = parser.parse_args()

    if args.dataset == 'FakeAVCeleb':
        if args.category == 'all':
            categories = ['RealVideo-RealAudio', 'RealVideo-FakeAudio', 'FakeVideo-RealAudio', 'FakeVideo-FakeAudio']
        elif args.category:
            categories = [args.category]

        for category in categories:
            process_fakeavceleb(category, args.metadata, args.data_path, args.save_path, args.max_workers)

    elif args.dataset == 'AV1M':
        if args.split == "test":
            path_to_images_root = os.path.join(args.data_path, "val")
            save_path =  os.path.join(args.save_path, "val")
        else:
            path_to_images_root = os.path.join(args.data_path, "train")
            save_path =  os.path.join(args.save_path, "train")
        process_av1m(args.metadata, path_to_images_root, save_path, args.max_workers)

if __name__ == "__main__":
    main()

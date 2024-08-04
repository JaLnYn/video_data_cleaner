import os
import random
import cv2
import torch
import torchvision
import models
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer

def pil_loader(image_array):
    img = Image.fromarray(image_array)
    return img.convert('RGB')

def extract_random_frames(video_path, num_frames=3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sorted(random.sample(range(total_frames), num_frames))
    
    frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if success:
            frames.append(pil_loader(frame))
        else:
            frames.append(None)  # Handle the case where the frame couldn't be read
    cap.release()
    return frames

class LightningWrapper(LightningModule):
    def __init__(self, model_hyper):
        super().__init__()
        self.model_hyper = model_hyper 
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    def batch_process_frames(self, frames):
        tensor_frames = [self.transforms(frame) for frame in frames if frame is not None]
        tensor_frames = torch.stack(tensor_frames).cuda()
        paras = self.model_hyper(tensor_frames)
        model_target = models.TargetNet(paras).cuda()
        for param in model_target.parameters():
            param.requires_grad = False

        # Quality prediction
        pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
        return pred

def sorted_walk(top, topdown=True, onerror=None, followlinks=False):
    for root, dirs, files in os.walk(top, topdown=topdown, onerror=onerror, followlinks=followlinks):
        dirs.sort()  # Sort directories in-place for consistent traversal
        files.sort()  # Sort files in-place for consistent processing order
        yield root, dirs, files


class VideoDataset(Dataset):
    def __init__(self, directory_path, log_file='logs.txt', transform=None, num_frames=3):
        """
        Args:
            directory_path (str): Path to the directory containing video files.
            log_file (str): Path to the log file containing processed video paths.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.video_paths = []
        self.num_frames=num_frames

        # Read processed videos from log file
        processed_videos = set()
        if os.path.exists(log_file):
            with open(log_file, 'r') as file:
                processed_videos = set(file.read().splitlines())

        # Collect unprocessed video paths
        for root, _, files in sorted_walk(directory_path):
            for file in files:
                if file.endswith('.mp4'):
                    path = os.path.join(root, file)
                    if path not in processed_videos:
                        self.video_paths.append(path)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = extract_random_frames(video_path, self.num_frames)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        return frames


def walk_directory_and_process(directory_path, threshold, output_file, batch_size=10, NUM_FRAMES=1):
    count = 0
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
    model_hyper.train(False)
    model_hyper.load_state_dict(torch.load('./pretrained/koniq_pretrained.pkl'))
    model_hyper = LightningWrapper(model_hyper)

    dataset = VideoDataset(directory_path=directory_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    trainer = Trainer(gpus=-1, strategy="ddp")

    score = trainer.predict(model_hyper, dataloaders=dataloader)

    # Process in batches
    buckets = [[] for i in range(11)]
    with open(output_file, 'w') as f:
        for i in tqdm(range(0, len(video_paths), batch_size)):
            batch_paths = video_paths[i:i+batch_size]
            frames = []
            for video_path in batch_paths:
                frames += extract_random_frames(video_path, NUM_FRAMES)
            # scores = model_hyper.batch_process_frames(frames, transforms)
            scores = trainer.predict(model_hyper, dataloaders=)
            assert  NUM_FRAMES * len(batch_paths) == len(scores)
            for idx in range(len(batch_paths)):
                score = scores[idx*NUM_FRAMES:(idx+1) * NUM_FRAMES].sum()/NUM_FRAMES
                buckets[int(score.item()//10)].append(batch_paths[idx])
                if score > threshold:
                    count += 1
                    f.write(f"{batch_paths[idx]}\n")

    with open('buckets.txt', 'w') as bucket_file:
        for idx in range(len(buckets)):
            for video_path in buckets[idx]:
                bucket_file.write(f"{video_path}\n")
            bucket_file.write("\n")
        for idx in range(len(buckets)):
            bucket_file.write(f"bucket {idx}: {len(buckets[idx])} videos\n")

    with open("logs.txt", 'a') as file:
        for path in batch_paths:
            file.write(path + '\n')

    print(f"Processing complete {count}/{len(video_paths)} remains")

# Usage
directory_path = '../talkinghead/dataset/mp4/'
# directory_path = '/datasets/voxceleb2/voxceleb2_AV/dev/mp4/'

output_file = 'quality_scores.txt'
threshold = 40  # Set your quality threshold
walk_directory_and_process(directory_path, threshold, output_file, batch_size=4, NUM_FRAMES=4)

# ./talkinghead/




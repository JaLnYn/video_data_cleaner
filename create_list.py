import os
import random
import cv2
import torch
import torchvision
import models
from PIL import Image
from tqdm import tqdm
import numpy as np

def pil_loader(image_array):
    img = Image.fromarray(image_array)
    return img.convert('RGB')

def extract_random_frames(video_path, num_frames=3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        frame_indices = sorted(random.sample(range(total_frames), num_frames))
    except Exception as e:
        print(e, range(total_frames))
        return None
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

def batch_process_frames(frames, model_hyper, transforms):
    tensor_frames = [transforms(frame) for frame in frames if frame is not None]
    tensor_frames = torch.stack(tensor_frames).cuda()
    paras = model_hyper(tensor_frames)
    model_target = models.TargetNet(paras).cuda()
    for param in model_target.parameters():
        param.requires_grad = False

    # Quality prediction
    pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
    return pred

def evaluate_quality(paras):
    model_target = models.TargetNet(paras).cuda()
    for param in model_target.parameters():
        param.requires_grad = False
    
    scores = []
    for param in paras:
        pred = model_target(param['target_in_vec'])
        scores.append(float(pred.item()))
    return scores

def sorted_walk(top, topdown=True, onerror=None, followlinks=False):
    for root, dirs, files in os.walk(top, topdown=topdown, onerror=onerror, followlinks=followlinks):
        dirs.sort()  # Sort directories in-place for consistent traversal
        files.sort()  # Sort files in-place for consistent processing order
        yield root, dirs, files

def walk_directory_and_process(directory_path, threshold, output_file, batch_size=10, NUM_FRAMES=1):
    count = 0
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
    model_hyper.train(False)
    model_hyper.load_state_dict(torch.load('./pretrained/koniq_pretrained.pkl'))

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    processed_videos = set()
    if os.path.exists("logs.txt"):
        with open("logs.txt", 'r') as file:
            processed_videos = set(file.read().splitlines())

    video_paths = []
    for root, dirs, files in sorted_walk(directory_path):
        for file in files:
            if file.endswith('.mp4'):
                path = os.path.join(root, file)
                if path not in processed_videos:
                    video_paths.append(path)

    # Process in batches
    buckets = [[] for i in range(11)]
    with open(output_file, 'a') as f:
        with open("logs.txt", 'a') as file:
            for i in tqdm(range(0, len(video_paths), batch_size)):
                batch_paths = video_paths[i:i+batch_size]
                frames = []
                scores = torch.zeros(NUM_FRAMES * len(batch_paths))
                valid_scores = []
                for idx, video_path in enumerate(batch_paths):
                    fra = extract_random_frames(video_path, NUM_FRAMES)
                    if fra is not None:
                        frames += fra
                        for score_idx in range(NUM_FRAMES):
                            valid_scores.append(idx * NUM_FRAMES + score_idx)
                    # if not frames:
                    #     continue
                
                processed_scores = batch_process_frames(frames, model_hyper, transforms)
                for score_idx, frame_idx in enumerate(valid_scores):
                    scores[frame_idx] = processed_scores[score_idx]
                assert  NUM_FRAMES * len(batch_paths) == len(scores)
                print(scores)
                for idx in range(len(batch_paths)):

                    score = scores[idx*NUM_FRAMES:(idx+1) * NUM_FRAMES].sum()/NUM_FRAMES
                    buckets[int(score.item()//10)].append(batch_paths[idx])
                    if score > threshold:
                        count += 1
                        f.write(f"{batch_paths[idx]}\n")
                    file.write(batch_paths[idx] + '\n')
                if i % 10 == 0:
                    f.flush()
                    file.flush()

    with open('buckets.txt', 'w') as bucket_file:
        for idx in range(len(buckets)):
            for video_path in buckets[idx]:
                bucket_file.write(f"{video_path}\n")
            bucket_file.write("\n")
        for idx in range(len(buckets)):
            bucket_file.write(f"bucket {idx}: {len(buckets[idx])} videos\n")

        
    print(f"Processing complete {count}/{len(video_paths)} remains")

# Usage
# directory_path = '../talkinghead/dataset/mp4/'
directory_path = '/datasets/voxceleb2/voxceleb2_AV/dev/mp4/'
# directory_path = './videos'
output_file = 'quality_scores.txt'
threshold = 40  # Set your quality threshold
walk_directory_and_process(directory_path, threshold, output_file, batch_size=64, NUM_FRAMES=4)

# ./talkinghead/




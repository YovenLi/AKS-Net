import os
import argparse
import time
import random
import numpy as np
import pandas as pd
import shutil
from PIL import Image
import logging
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils import data

from models import AKMnet


def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Keyframe extraction for point cloud videos")
    parser.add_argument('--gpu', help="GPU device id to use [0]", default=1, type=int)
    parser.add_argument('--num_epochs', help='Maximum number of training epochs', default=1, type=int)
    parser.add_argument('--batch_size', help='Batch size', default=1, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=0.0005, help='Weight decay rate')
    parser.add_argument('--model', default='AKMnet', type=str, help='Model architecture')
    parser.add_argument('--database', default='sjtu', type=str, help='Dataset name (sjtu, WPC, etc.)')
    parser.add_argument('--label', default='/home/wanghm/wanghaomiao/MM/csvfiles/wpc_data_info/total.csv', 
                        type=str, help='Path to CSV file with video information')
    parser.add_argument('--data_dir_frame', default='/home/wanghm/wanghaomiao/frames', 
                        type=str, help='Path to directory containing 120 frames per video')
    parser.add_argument('--output_base_dir', type=str, default='/home/wanghm/wanghaomiao/AKS_Net/first/keyframes',
                        help='Base output directory for keyframes')
    parser.add_argument('--num_keyframes', type=int, default=9, 
                        help='Number of keyframes to extract per video (1-12 recommended)')
    parser.add_argument('--bw_loss_weight', type=float, default=11.0,
                        help='Weight factor for frame weight loss (BwLoss)')
    parser.add_argument('--b_loss_weight', type=float, default=10.0,
                        help='Weight factor for binary matrix loss (BLoss)')
    return parser.parse_args()


def set_rand_seed(seed: int = 1998) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_keyframes(weights_dict: Dict, num_keyframes: int, args) -> None:
    """Save keyframes based on weights matrix."""
    logger = logging.getLogger(__name__)
    
    # Create output directories
    dataset_dir = os.path.join(args.output_base_dir, args.database)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create output directory for frames
    frames_dir = os.path.join(dataset_dir, f"{num_keyframes}_frame")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Process each video
    for video_key, weights in weights_dict.items():
        # Handle different key formats
        if isinstance(video_key, tuple):
            video_name = video_key[0]
        else:
            video_name = video_key
            
        # Create video-specific output directory
        video_dir = os.path.join(frames_dir, video_name)
        os.makedirs(video_dir, exist_ok=True)
        
        # Get source frames directory
        source_dir = os.path.join(args.data_dir_frame, video_name)
        if not os.path.exists(source_dir):
            continue
        
        # Handle weights tensor
        try:
            # Flatten and convert to (index, weight) pairs
            weights_flat = weights.view(-1)
            frame_weights = [(idx, weight.item()) for idx, weight in enumerate(weights_flat)]
            sorted_frames = sorted(frame_weights, key=lambda x: x[1], reverse=True)
        except Exception:
            continue
        
        # Save the top frames
        for rank, (frame_idx, weight) in enumerate(sorted_frames[:num_keyframes]):
            frame_num = frame_idx + 1
            source_path = os.path.join(source_dir, f"{frame_num:03d}.png")
            
            # Check if the source file exists
            if not os.path.exists(source_path):
                continue
            
            # Save the frame
            target_path = os.path.join(video_dir, f"{frame_num:03d}.png")
            try:
                shutil.copy(source_path, target_path)
            except Exception:
                pass


class KeyDataset(data.Dataset):
    """Dataset for keyframe extraction from videos with 120 frames each."""
    
    def __init__(self, data_dir_frame: str, datainfo_path: str, 
                 transform: transforms.Compose, is_train: bool, crop_size: int = 224):
        super(KeyDataset, self).__init__()
        
        # Load video information
        self.data_info = pd.read_csv(datainfo_path, header=0, sep=',', 
                                    index_col=False, encoding="utf-8-sig")
        self.video_names = self.data_info[['name']]
        self.mos_scores = self.data_info['mos']
        self.num_videos = len(self.video_names)
        
        # Store parameters
        self.data_dir_frame = data_dir_frame
        self.is_train = is_train
        self.transform = transform
        self.crop_size = crop_size
        
    def __len__(self) -> int:
        return self.num_videos

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        # Get video name and create path to frames
        video_name = self.video_names.iloc[idx, 0]
        frame_dir = os.path.join(self.data_dir_frame, video_name)
        
        # Initialize frame tensor
        num_frames = 120
        frame_tensor = torch.zeros([num_frames, 3, self.crop_size, self.crop_size])
        frame_count = 0
        
        # Load and transform each frame
        for i in range(1, num_frames + 1):
            frame_path = os.path.join(frame_dir, f"{i:03d}.png")
            if os.path.exists(frame_path):
                frame = Image.open(frame_path).convert('RGB')
                frame = self.transform(frame)
                frame_tensor[frame_count] = frame
                frame_count += 1
        
        # Get MOS score
        mos_score = self.mos_scores.iloc[idx]
        mos_tensor = torch.FloatTensor(np.array(mos_score))
        
        return frame_tensor, mos_tensor, video_name


class BwLoss(nn.Module):
    """Loss function for frame weights."""
    
    def __init__(self):
        super(BwLoss, self).__init__()
    
    def forward(self, Bw: torch.Tensor) -> torch.Tensor:
        loss_Bw = 0.0
        for i in range(Bw.shape[0]):
            temp = Bw[i, :]
            loss_Bw += 2.0 - (torch.mean(temp[temp > torch.mean(temp)]) - 
                             torch.mean(temp[temp < torch.mean(temp)]))
        return loss_Bw / Bw.shape[0]


class BLoss(nn.Module):
    """Loss function for binary matrix."""
    
    def __init__(self, device):
        super(BLoss, self).__init__()
        self.device = device
    
    def forward(self, B: torch.Tensor) -> torch.Tensor:
        loss_B = 0.0
        for i in range(B.shape[0]):
            loss_B += torch.max(torch.Tensor([0.0]).to(self.device),
                              torch.sum(B[i, :]) - torch.Tensor([1.0]).to(self.device))
        return 0.1 * loss_B / B.shape[0]


def main():
    """Main training and keyframe extraction function."""
    # Setup logging
    logger = setup_logging()
    
    # Parse arguments and set random seed
    args = parse_args()
    set_rand_seed()
    
    # Create necessary directories
    dataset_dir = os.path.join(args.output_base_dir, args.database)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create checkpoint directory
    model_dir = os.path.join(dataset_dir, 'ckpt')
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Initialize model
    model = AKMnet.resnet18(pretrained=False)
    model = model.to(device)
    
    # Define data transformations
    transformations_train = transforms.Compose([
        transforms.RandomCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transformations_test = transforms.Compose([
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = KeyDataset(
        data_dir_frame=args.data_dir_frame, 
        datainfo_path=args.label,
        transform=transformations_train, 
        is_train=True
    )
    
    test_dataset = KeyDataset(
        data_dir_frame=args.data_dir_frame, 
        datainfo_path=args.label,
        transform=transformations_test, 
        is_train=False
    )
    
    # Initialize training components
    optimizer = optim.SGD(model.parameters(), args.learning_rate, 
                         momentum=0.9, weight_decay=args.decay_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs/5, eta_min=1e-8, last_epoch=-1
    )
    
    criterionBw = BwLoss().to(device)
    criterionB = BLoss(device).to(device)
    
    # Training variables
    best_loss = float('inf')
    best_weights = {}
    best_binary = {}
    
    # Training loop
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        
        # Set model to training mode
        model.train()
        
        # Initialize epoch variables
        epoch_weights = {}
        epoch_binary = {}
        total_loss_bw = 0.0
        total_loss_b = 0.0
        
        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=8
        )
        
        # Training batch loop
        num_batches = 0
        for batch_idx, (frames, mos, names) in enumerate(train_loader):
            frames = frames.to(device)
            mos = mos.to(device)
            
            # Forward pass
            try:
                outputs, B, Bw = model(frames)
                
                # Store batch results for each video in the batch
                for i, name in enumerate(names):
                    epoch_weights[name] = Bw[i:i+1]
                    epoch_binary[name] = B[i:i+1]
                
                # Calculate losses
                loss_bw = criterionBw(Bw) * args.bw_loss_weight
                loss_b = criterionB(B) * args.b_loss_weight
                total_loss = loss_bw + loss_b
                
                # Backward pass
                total_loss.backward()
                
                # Update weights
                if (batch_idx + 1) % 8 == 0 or (batch_idx + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Accumulate losses
                total_loss_bw += loss_bw.item()
                total_loss_b += loss_b.item()
                num_batches += 1
            
            except Exception:
                continue
        
        # Calculate average losses
        if num_batches > 0:
            avg_loss_bw = total_loss_bw / num_batches
            avg_loss_b = total_loss_b / num_batches
            total_avg_loss = avg_loss_bw + avg_loss_b
            
            # Save best model
            if total_avg_loss < best_loss:
                best_loss = total_avg_loss
                best_weights = epoch_weights
                best_binary = epoch_binary
                
                # Save model
                model_path = os.path.join(model_dir, 'model_best.pth')
                torch.save(model.state_dict(), model_path)
        
        scheduler.step()
    
    # Extract keyframes using best model
    if best_weights:
        save_keyframes(best_weights, args.num_keyframes, args)


if __name__ == '__main__':
    main()



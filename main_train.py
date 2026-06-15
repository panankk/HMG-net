import torch
import os
import math
import argparse
import yaml
import numpy as np
import random

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


from data import build_dataset
from models import build_model
from engine import SequentialFusionLoss, Trainer

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class ConfigNode:
   
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigNode(v))
            else:
                setattr(self, k, v)
                
    def get(self, key, default=None):
       
        return getattr(self, key, default)

def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    return ConfigNode(cfg_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the yaml config file (e.g., configs/v39_full_model.yaml)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    # 1. 读取真实的 YAML 配置
    cfg = load_config(args.config)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f" Launching Training Engine")
    print(f" Config File: {args.config}")
    
    # 2. 构建数据引擎
    dataset = build_dataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # 3. 拼装
    model = build_model(cfg).to(device)
    print(" Model Built Successfully!")
    
    # 4. 构建 Loss 和 优化器
    criterion = SequentialFusionLoss().to(device)
    optimizer = AdamW(list(model.parameters()) + list(criterion.parameters()), lr=cfg.TRAIN.LEARNING_RATE, weight_decay=1e-4)
    
    # 5. 学习率调度器
    total_steps = len(loader) * cfg.TRAIN.NUM_EPOCHS
    warmup_steps = len(loader) * cfg.TRAIN.WARMUP_EPOCHS
    def lr_lambda(current_step):
        if current_step < warmup_steps: 
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # 6. 断点续训逻辑
    start_epoch = 0
    if args.resume:
        print(f" Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        criterion.load_state_dict(ckpt['criterion_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f" Resumed at Epoch {start_epoch}")

    # 7. 启动引擎
    trainer = Trainer(
        model=model, 
        dataloader=loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        device=device, 
        config=cfg.TRAIN  
    )
    
    trainer.train(start_epoch)

if __name__ == "__main__": 
    seed_everything(seed=42)
    main()

"""CBIS‑DDSM trainer with **freeze‑then‑unfreeze** fine‑tuning,
ResNet / ResNet‑V2 backbones, dropout flag, label smoothing,
weight‑decay, and optional W&B logging.

Now supports **4‑class classification** out of the box (e.g. benign‑mass, benign‑calc, malignant‑mass, malignant‑calc).

Example – 5 frozen + 25 full fine‑tune epochs (total 30):
    python train_cbis_ddsm_4cat.py \
        --model resnet50v2 \
        --freeze-backbone --freeze-epochs 5 --head-lr 3e-3 \
        --epochs 25 --lr 3e-4 \
        --batch-size 32 --image-size 224 \
        --label-smoothing 0.25 --lr-scheduler plateau \
        --dropout 0.3 --weight-decay 1e-4 --wandb \
        --num-classes 4
"""

import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import (
    resnet50, resnet101, resnet152,
    ResNet50_Weights, ResNet101_Weights, ResNet152_Weights,
)
from torchvision.transforms import InterpolationMode

try:
    import timm  # for *v2 models
except ImportError:
    timm = None

import wandb

# --------------------------- CONSTANTS ---------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
# TRAIN_DIR = "data/dataset/pathology/train"
# VAL_DIR   = "data/dataset/pathology/val"
#TEST_DIR  = "data/dataset/pathology/test"

# TRAIN_DIR = "data/dataset_16bit_fixed420/pathology/train"
# VAL_DIR   = "data/dataset_16bit_fixed420/pathology/val"
# TEST_DIR  = "data/dataset_16bit_fixed420/pathology/test"

#TRAIN_DIR = "data/dataset_16bit_fixed900/pathology/train"
#VAL_DIR   = "data/dataset_16bit_fixed900/pathology/val"
#TEST_DIR  = "data/dataset_16bit_fixed900/pathology/test"
# ----------------------------------------------------------------
TRAIN_DIR = "DATASET_MASS_SHAPE/train"
VAL_DIR   = "DATASET_MASS_SHAPE/val"
TEST_DIR  = "DATASET_MASS_SHAPE/test"
# =============================== UTILS ===========================

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_dataloaders(sz: int, bs: int, nw: int, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    seed_everything(seed)
    dup3 = T.Lambda(lambda t: t if t.shape[0] == 3 else t.repeat(3, 1, 1))
    train_tf = T.Compose([
        T.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(10, interpolation=InterpolationMode.NEAREST, fill=0),
        T.RandomVerticalFlip(p=0.5),
        T.ToTensor(),
        dup3,
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        # T.CenterCrop(sz),
        T.ToTensor(),
        dup3,
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = ImageFolder(TRAIN_DIR, train_tf)
    val_ds   = ImageFolder(VAL_DIR,  eval_tf)
    test_ds  = ImageFolder(TEST_DIR, eval_tf)
    common = dict(batch_size=bs, num_workers=min(nw, os.cpu_count()//2), pin_memory=True)
    return (DataLoader(train_ds, shuffle=True , **common),
            DataLoader(val_ds,   shuffle=False, **common),
            DataLoader(test_ds,  shuffle=False, **common),
            train_ds.classes)

# ============================= MODEL =============================

class CustomResNet(nn.Module):
    _TV = {
        "resnet50":  (resnet50,  ResNet50_Weights.DEFAULT),
        "resnet101": (resnet101, ResNet101_Weights.DEFAULT),
        "resnet152": (resnet152, ResNet152_Weights.DEFAULT),
    }
    _TIMM = {
        "resnet50v2":  "resnetv2_50",
        "resnet101v2": "resnetv2_101",
        "resnet152v2": "resnetv2_152",
    }
    def __init__(self, name: str, n_classes: int, dropout: float):
        super().__init__()
        if name in self._TV:
            ctor, wts = self._TV[name]
            self.backbone = ctor(weights=wts)
            in_f = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_f, 512), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(512, n_classes))
        elif name in self._TIMM:
            if timm is None:
                raise ImportError("Install timm for *v2 models")
            self.backbone = timm.create_model(self._TIMM[name], pretrained=True,
                                              num_classes=n_classes, drop_rate=dropout)
        else:
            raise ValueError(name)
    def forward(self, x):
        return self.backbone(x)

# --------------------------- FREEZE UTILS ------------------------

def set_backbone_trainable(model: "CustomResNet", trainable_backbone: bool):
    """
    If ``trainable_backbone`` is ``False`` – freezes *all* layers except the classifier head.
    If ``True`` – everything is trainable.
    Works for both torchvision and timm models.
    """
    # 1) First, give every parameter the desired flag
    for p in model.backbone.parameters():
        p.requires_grad = trainable_backbone

    # 2) Make absolutely sure the *head* stays learnable
    heads = []
    if hasattr(model.backbone, "fc"):          # torchvision ResNet
        heads.append(model.backbone.fc)
    if hasattr(model.backbone, "classifier"):  # many timm models
        heads.append(model.backbone.classifier)
    if hasattr(model.backbone, "head"):        # resnetv2_* in timm
        heads.append(model.backbone.head)

    for head in heads:
        for p in head.parameters():
            p.requires_grad = True
# ========================= TRAIN / EVAL ==========================

def train_epoch(model, loader, crit, opt, dev):
    model.train(); t_loss=t_corr=t_tot=0
    for x,y in loader:
        x,y=x.to(dev),y.to(dev); opt.zero_grad(); out=model(x); loss=crit(out,y)
        loss.backward(); opt.step()
        t_loss+=loss.item()*y.size(0); t_corr+=(out.argmax(1)==y).sum().item(); t_tot+=y.size(0)
    return t_loss/t_tot, 100*t_corr/t_tot

@torch.no_grad()
def eval_epoch(model, loader, crit, dev):
    model.eval(); v_loss=v_corr=v_tot=0
    for x,y in loader:
        x,y=x.to(dev),y.to(dev); out=model(x); loss=crit(out,y)
        v_loss+=loss.item()*y.size(0); v_corr+=(out.argmax(1)==y).sum().item(); v_tot+=y.size(0)
    return v_loss/v_tot, 100*v_corr/v_tot

# ====================== SCHEDULER FACTORY ========================

def make_scheduler(kind, opt, epochs, steps, base_lr, monitor_max):
    if kind=="plateau":
        mode="max" if monitor_max else "min"
        return optim.lr_scheduler.ReduceLROnPlateau(opt, mode=mode, factor=0.5, patience=2)
    if kind=="step":
        return optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    if kind=="onecycle":
        return optim.lr_scheduler.OneCycleLR(opt, max_lr=base_lr*10, total_steps=epochs*steps)
    if kind=="exp":
        return optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
    raise ValueError(kind)

# ================================ MAIN ===========================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["resnet50","resnet101","resnet152","resnet50v2","resnet101v2","resnet152v2"], default="resnet50")
    p.add_argument("--epochs", type=int, default=20, help="epochs AFTER unfreeze")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--freeze-epochs", type=int, default=3)
    p.add_argument("--head-lr", type=float, default=3e-3)
    p.add_argument("--lr", type=float, default=3e-4, help="LR for full fine‑tune")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-smoothing", type=float, default=0.25)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr-scheduler", choices=["plateau","step","onecycle","exp"], default="plateau") #understand what they are meant for , add the val to train '
    p.add_argument("--num-classes", type=int, default=4, help="Number of target categories (default: 4)")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    if args.wandb:
        wandb.init(project="cbis-ddsm", config=vars(args))

    train_loader, val_loader, test_loader, classes = get_dataloaders(args.image_size, args.batch_size, args.num_workers, args.seed)

    # ------------------------------------------------------------------
    # Sanity‑check: dataset vs requested number of classes
    # ------------------------------------------------------------------
    if len(classes) != args.num_classes:
        print(f"[INFO] Detected {len(classes)} class folders on disk: {classes} – overriding to {args.num_classes} as requested.")
    n_classes = args.num_classes

    model = CustomResNet(args.model, n_classes=n_classes, dropout=args.dropout).to(device)

    # ---------- phase 0: maybe frozen ----------
    phase = "frozen" if args.freeze_backbone else "full"
    total_epochs = args.freeze_epochs + args.epochs if args.freeze_backbone else args.epochs

    set_backbone_trainable(model, not args.freeze_backbone)
    base_lr = args.head_lr if args.freeze_backbone else args.lr
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(args.lr_scheduler, optimizer,
                               args.freeze_epochs if phase == "frozen" else total_epochs,
                               len(train_loader), base_lr,
                               monitor_max=(phase == "frozen"))

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_metric = 0.0 if phase == "frozen" else float("inf")

    for epoch in range(1, total_epochs + 1):
        # --- train & validate ---
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = eval_epoch(model, val_loader, criterion, device)

        # --- scheduler step ---
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            monitor = val_acc if phase == "frozen" else val_loss
            scheduler.step(monitor)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"E{epoch:02d}/{total_epochs} | {phase:6} | "
              f"Train {train_loss:.4f}/{train_acc:.2f}%  | "
              f"Val {val_loss:.4f}/{val_acc:.2f}% | LR {current_lr:.2e}")

        # --- W&B log ---
        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "phase": phase,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": current_lr,
            })

        # --- save best ---
        is_best = (val_acc > best_metric) if phase == "frozen" else (val_loss < best_metric)
        if is_best:
            best_metric = val_acc if phase == "frozen" else val_loss
            os.makedirs("models", exist_ok=True)
            ckpt_path = f"models/{args.model}_{phase}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print("  ✔  Saved new best →", ckpt_path)

        # --- unfreeze backbone ---
        if args.freeze_backbone and epoch == args.freeze_epochs:
            print(">>> Unfreezing backbone – switching to full fine‑tune …")
            phase = "full"
            set_backbone_trainable(model, True)
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=args.weight_decay)
            scheduler = make_scheduler(args.lr_scheduler, optimizer, args.epochs, len(train_loader), args.lr, monitor_max=False)
            best_metric = float("inf")  # reset for loss tracking

    # ---------- final test ----------
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"*** TEST ***  loss = {test_loss:.4f}   acc = {test_acc:.2f}%")
    if args.wandb:
        wandb.summary["test/loss"] = test_loss
        wandb.summary["test/acc"] = test_acc


if __name__ == "__main__":
    main()

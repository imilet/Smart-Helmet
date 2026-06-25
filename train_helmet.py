import argparse
from pathlib import Path

import torch
import yaml
from torch import optim
from tqdm import tqdm

from yolo.models.yolo import Model
from yolo.utils.datasets import create_dataloader
from yolo.utils.torch_utils import select_device
from yolo.utils.utils import compute_loss


DEFAULT_HYP = {
    "giou": 0.05,
    "cls": 0.5,
    "cls_pw": 1.0,
    "obj": 1.0,
    "obj_pw": 1.0,
    "iou_t": 0.20,
    "anchor_t": 4.0,
    "fl_gamma": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="训练安全帽 YOLOv5 检测模型")
    parser.add_argument("--data", default="configs/helmet_data.yaml")
    parser.add_argument("--cfg", default="configs/helmet_yolov5s.yaml")
    parser.add_argument("--weights", default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="runs/train/helmet")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def load_data_config(path):
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    for key in ["train", "val", "nc", "names"]:
        if key not in data:
            raise KeyError(f"数据配置缺少字段: {key}")
    return data


def build_model(cfg, nc):
    model = Model(cfg, ch=3, nc=nc)
    model.nc = nc
    model.hyp = DEFAULT_HYP
    model.gr = 1.0
    return model


def train(args):
    data = load_data_config(args.data)
    device = select_device(args.device)
    output_dir = Path(args.output)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.cfg, data["nc"]).to(device)
    if args.weights:
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint["model"].float().state_dict()
        model.load_state_dict(state_dict, strict=False)
    model.names = data["names"]

    dataloader, _ = create_dataloader(
        data["train"],
        args.img_size,
        args.batch_size,
        int(model.stride.max()),
        args,
        hyp=DEFAULT_HYP,
        augment=True,
        workers=args.workers,
    )
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.937, weight_decay=5e-4)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"helmet epoch {epoch + 1}/{args.epochs}")
        for images, targets, _, _ in pbar:
            images = images.to(device).float() / 255.0
            targets = targets.to(device)
            pred = model(images)
            loss, loss_items = compute_loss(pred, targets, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()), box=float(loss_items[0]))

        checkpoint = {"epoch": epoch, "model": model, "names": model.names}
        torch.save(checkpoint, weights_dir / "last.pt")
        torch.save(checkpoint, weights_dir / "best.pt")


def main():
    train(parse_args())


if __name__ == "__main__":
    main()

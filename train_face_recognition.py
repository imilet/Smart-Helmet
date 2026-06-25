import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from face.models.arcmargin import ArcNet
from face.models.mobilefacenet import MobileFaceNet
from face.recognition_dataset import BinaryFaceDataset


def parse_args():
    parser = argparse.ArgumentParser(description="训练 MobileFaceNet 人脸识别模型")
    parser.add_argument("--train-root", default="datasets/faces/train_data", help="不带 .data 后缀的数据根路径")
    parser.add_argument("--val-root", default="datasets/faces/val_data", help="不带 .data 后缀的数据根路径")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="runs/train/face_recognition")
    return parser.parse_args()


def build_models(num_classes):
    model = MobileFaceNet()
    metric = ArcNet(feature_dim=512, class_dim=num_classes)
    return model, metric


def train(args):
    device = torch.device(args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    train_dataset = BinaryFaceDataset(args.train_root, is_train=True)
    if len(train_dataset) == 0:
        raise FileNotFoundError(
            f"未找到训练数据，请准备 {args.train_root}.data/.label/.header"
        )

    model, metric = build_models(train_dataset.num_classes)
    model = model.to(device)
    metric = metric.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        list(model.parameters()) + list(metric.parameters()),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        metric.train()
        pbar = tqdm(dataloader, desc=f"face epoch {epoch + 1}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).long().view(-1, 1)
            features = model(images)
            logits = metric(features, labels)
            loss = criterion(logits, labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        torch.jit.save(torch.jit.script(model.cpu()), output_dir / "mobilefacenet.pth")
        model.to(device)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()

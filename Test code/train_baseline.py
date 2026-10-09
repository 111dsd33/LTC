import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = (
    PROJECT_DIR
    / "07_experiments"
    / "segformer_b2_baseline"
)
CONFIG_PATH = EXPERIMENT_DIR / "config.json"

CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
LOG_DIR = EXPERIMENT_DIR / "logs"

IMAGE_DIR = PROJECT_DIR / "01_原始果园图像"
MASK_DIR = PROJECT_DIR / "02_分割标注" / "分割掩膜"

MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class WinterJujubeDataset(Dataset):
    def __init__(
        self,
        split_file: Path,
        processor: SegformerImageProcessor,
    ) -> None:
        self.processor = processor

        self.filenames = [
            line.strip()
            for line in split_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if not self.filenames:
            raise ValueError(
                f"划分文件为空：{split_file}"
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        filename = self.filenames[index]

        image_path = IMAGE_DIR / filename
        mask_path = MASK_DIR / f"{image_path.stem}.png"

        if not image_path.exists():
            raise FileNotFoundError(
                f"找不到图片：{image_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"找不到掩膜：{mask_path}"
            )

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        mask_array = np.asarray(mask, dtype=np.uint8)
        unique_values = set(
            np.unique(mask_array).tolist()
        )

        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"{mask_path.name} 包含异常像素值："
                f"{sorted(unique_values)}"
            )

        encoded = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt",
        )

        return {
            "pixel_values": (
                encoded["pixel_values"]
                .squeeze(0)
            ),
            "labels": (
                encoded["labels"]
                .squeeze(0)
                .long()
            ),
            "filename": filename,
        }


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
) -> None:
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)

    valid = (
        (target >= 0)
        & (target < num_classes)
    )

    indices = (
        target[valid] * num_classes
        + prediction[valid]
    )

    counts = torch.bincount(
        indices,
        minlength=num_classes ** 2,
    )

    confusion_matrix += counts.reshape(
        num_classes,
        num_classes,
    ).cpu()


def calculate_metrics(
    confusion_matrix: torch.Tensor,
) -> dict:
    matrix = confusion_matrix.double()

    true_positive = matrix.diag()
    ground_truth = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)

    union = (
        ground_truth
        + predicted
        - true_positive
    )

    iou = true_positive / union.clamp(min=1)
    class_accuracy = (
        true_positive
        / ground_truth.clamp(min=1)
    )

    branch_tp = true_positive[1]
    branch_fp = predicted[1] - branch_tp
    branch_fn = ground_truth[1] - branch_tp

    branch_precision = (
        branch_tp
        / (branch_tp + branch_fp).clamp(min=1)
    )

    branch_recall = (
        branch_tp
        / (branch_tp + branch_fn).clamp(min=1)
    )

    branch_f1 = (
        2
        * branch_precision
        * branch_recall
        / (
            branch_precision
            + branch_recall
        ).clamp(min=1e-12)
    )

    dice = (
        2 * branch_tp
        / (
            2 * branch_tp
            + branch_fp
            + branch_fn
        ).clamp(min=1)
    )

    return {
        "background_iou": iou[0].item(),
        "branch_iou": iou[1].item(),
        "miou": iou.mean().item(),
        "background_pa": class_accuracy[0].item(),
        "branch_pa": class_accuracy[1].item(),
        "mpa": class_accuracy.mean().item(),
        "branch_precision": branch_precision.item(),
        "branch_recall": branch_recall.item(),
        "branch_f1": branch_f1.item(),
        "branch_dice": dice.item(),
    }


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    use_amp,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    progress = tqdm(
        dataloader,
        desc="训练",
        leave=False,
    )

    for batch in progress:
        pixel_values = batch["pixel_values"].to(
            device,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

            loss = outputs.loss

        if not math.isfinite(loss.item()):
            raise ValueError(
                f"训练损失异常：{loss.item()}"
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = pixel_values.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / total_samples


@torch.inference_mode()
def validate_one_epoch(
    model,
    dataloader,
    device,
    use_amp,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    confusion_matrix = torch.zeros(
        (2, 2),
        dtype=torch.int64,
    )

    progress = tqdm(
        dataloader,
        desc="验证",
        leave=False,
    )

    for batch in progress:
        pixel_values = batch["pixel_values"].to(
            device,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

        resized_logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        prediction = resized_logits.argmax(dim=1)

        update_confusion_matrix(
            confusion_matrix,
            prediction,
            labels,
        )

        batch_size = pixel_values.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    validation_loss = total_loss / total_samples
    metrics = calculate_metrics(confusion_matrix)

    return validation_loss, metrics, confusion_matrix


def save_log_row(
    log_path: Path,
    row: dict,
) -> None:
    file_exists = log_path.exists()

    with log_path.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"找不到配置文件：{CONFIG_PATH}"
        )

    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    seed = int(config["random_seed"])
    input_size = int(config["input_size"])
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])
    epochs = int(
        config["epochs_for_first_test"]
    )

    learning_rate = float(
        config["learning_rate"]
    )
    momentum = float(config["momentum"])
    weight_decay = float(
        config["weight_decay"]
    )
    use_amp = bool(
        config["mixed_precision"]
    )

    set_random_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "没有检测到CUDA显卡"
        )

    device = torch.device("cuda")

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_split = (
        PROJECT_DIR
        / config["train_split"]
    )

    val_split = (
        PROJECT_DIR
        / config["val_split"]
    )

    print("正在从本地缓存加载处理器……")

    processor = (
        SegformerImageProcessor
        .from_pretrained(
            MODEL_NAME,
            do_reduce_labels=False,
            size={
                "height": input_size,
                "width": input_size,
            },
            local_files_only=True,
        )
    )

    train_dataset = WinterJujubeDataset(
        train_split,
        processor,
    )

    val_dataset = WinterJujubeDataset(
        val_split,
        processor,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print("正在从本地缓存加载模型……")

    model = (
        SegformerForSemanticSegmentation
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label={
                0: "background",
                1: "branch",
            },
            label2id={
                "background": 0,
                "branch": 1,
            },
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
    )

    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    log_path = (
        LOG_DIR
        / "short_training_log.csv"
    )

    if log_path.exists():
        log_path.unlink()

    best_branch_iou = -1.0

    print("=" * 60)
    print("开始两轮短训练")
    print(f"训练集：{len(train_dataset)}张")
    print(f"验证集：{len(val_dataset)}张")
    print(f"批量大小：{batch_size}")
    print(f"输入尺寸：{input_size}×{input_size}")
    print(f"训练轮数：{epochs}")
    print(f"运行设备：{device}")
    print("=" * 60)

    training_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        validation_loss, metrics, matrix = (
            validate_one_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                use_amp=use_amp,
            )
        )

        epoch_minutes = (
            time.time() - epoch_start
        ) / 60

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation_loss,
            "background_iou": (
                metrics["background_iou"]
            ),
            "branch_iou": (
                metrics["branch_iou"]
            ),
            "miou": metrics["miou"],
            "mpa": metrics["mpa"],
            "branch_precision": (
                metrics["branch_precision"]
            ),
            "branch_recall": (
                metrics["branch_recall"]
            ),
            "branch_f1": (
                metrics["branch_f1"]
            ),
            "branch_dice": (
                metrics["branch_dice"]
            ),
            "epoch_minutes": epoch_minutes,
        }

        save_log_row(log_path, row)

        print("\n" + "=" * 60)
        print(f"Epoch {epoch}/{epochs}")
        print(f"训练损失：{train_loss:.6f}")
        print(
            f"验证损失："
            f"{validation_loss:.6f}"
        )
        print(
            f"背景IoU："
            f"{metrics['background_iou']:.4f}"
        )
        print(
            f"枝条IoU："
            f"{metrics['branch_iou']:.4f}"
        )
        print(
            f"mIoU："
            f"{metrics['miou']:.4f}"
        )
        print(
            f"mPA："
            f"{metrics['mpa']:.4f}"
        )
        print(
            f"枝条Precision："
            f"{metrics['branch_precision']:.4f}"
        )
        print(
            f"枝条Recall："
            f"{metrics['branch_recall']:.4f}"
        )
        print(
            f"枝条F1："
            f"{metrics['branch_f1']:.4f}"
        )
        print(
            f"枝条Dice："
            f"{metrics['branch_dice']:.4f}"
        )
        print(
            f"本轮用时："
            f"{epoch_minutes:.2f}分钟"
        )
        print("混淆矩阵：")
        print(matrix.numpy())
        print("=" * 60)

        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scaler_state_dict": (
                scaler.state_dict()
            ),
            "branch_iou": (
                metrics["branch_iou"]
            ),
            "config": config,
        }

        torch.save(
            last_checkpoint,
            CHECKPOINT_DIR
            / "short_training_last.pt",
        )

        if (
            metrics["branch_iou"]
            > best_branch_iou
        ):
            best_branch_iou = (
                metrics["branch_iou"]
            )

            best_dir = (
                CHECKPOINT_DIR
                / "short_training_best"
            )

            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)

            print(
                "已保存新的最佳模型，"
                f"枝条IoU="
                f"{best_branch_iou:.4f}"
            )

    total_minutes = (
        time.time() - training_start
    ) / 60

    print("\n两轮短训练完成")
    print(
        f"最佳枝条IoU："
        f"{best_branch_iou:.4f}"
    )
    print(
        f"总用时：{total_minutes:.2f}分钟"
    )
    print(f"训练日志：{log_path}")
    print(f"模型目录：{CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from train_baseline import (
    PROJECT_DIR,
    WinterJujubeDataset,
    validate_one_epoch,
)


EXPERIMENT_DIR = (
    PROJECT_DIR
    / "07_experiments"
    / "segformer_b2_baseline_effbs10"
)

CONFIG_PATH = EXPERIMENT_DIR / "config.json"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
LOG_DIR = EXPERIMENT_DIR / "logs"

MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def train_one_epoch_with_accumulation(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    use_amp: bool,
    accumulation_steps: int,
):
    """
    使用梯度累积训练。

    每个微批次的平均损失先乘以该批次的样本数量。
    累积完成后，再用累计样本总数除梯度。

    这样最后不足5张的微批次也能得到正确权重：
    例如最后两批为5张和1张时，等价于一个6张批次，
    而不是把两个微批次各占50%的权重。
    """
    model.train()

    total_loss = 0.0
    total_samples = 0
    accumulated_samples = 0
    optimizer_step_count = 0

    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(
        enumerate(dataloader, start=1),
        total=len(dataloader),
        desc="训练",
        leave=False,
    )

    for micro_step, batch in progress:
        pixel_values = batch["pixel_values"].to(
            device,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )

        current_batch_size = pixel_values.size(0)

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

        # 模型返回的是平均损失。
        # 乘以当前样本数，先转换为该微批次的损失总和。
        weighted_loss = loss * current_batch_size

        scaler.scale(weighted_loss).backward()

        accumulated_samples += current_batch_size
        total_loss += loss.item() * current_batch_size
        total_samples += current_batch_size

        is_accumulation_boundary = (
            micro_step % accumulation_steps == 0
        )

        is_last_micro_batch = (
            micro_step == len(dataloader)
        )

        if (
            is_accumulation_boundary
            or is_last_micro_batch
        ):
            # 先解除混合精度缩放，再将累计梯度变成样本平均梯度。
            scaler.unscale_(optimizer)

            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(
                        accumulated_samples
                    )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

            accumulated_samples = 0
            optimizer_step_count += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            updates=optimizer_step_count,
        )

    average_loss = total_loss / total_samples

    return average_loss, optimizer_step_count


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

    micro_batch_size = int(
        config["micro_batch_size"]
    )

    accumulation_steps = int(
        config["gradient_accumulation_steps"]
    )

    effective_batch_size = int(
        config["effective_batch_size"]
    )

    num_workers = int(
        config["num_workers"]
    )

    epochs = int(
        config["formal_epochs"]
    )

    learning_rate = float(
        config["learning_rate"]
    )

    momentum = float(
        config["momentum"]
    )

    weight_decay = float(
        config["weight_decay"]
    )

    use_amp = bool(
        config["mixed_precision"]
    )

    calculated_effective_batch = (
        micro_batch_size
        * accumulation_steps
    )

    if (
        calculated_effective_batch
        != effective_batch_size
    ):
        raise ValueError(
            "有效批量配置不一致："
            f"{micro_batch_size} × "
            f"{accumulation_steps} = "
            f"{calculated_effective_batch}，"
            f"但配置文件写的是"
            f"{effective_batch_size}"
        )

    set_random_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "没有检测到可用的CUDA显卡"
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
        split_file=train_split,
        processor=processor,
    )

    val_dataset = WinterJujubeDataset(
        split_file=val_split,
        processor=processor,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print("正在从本地缓存加载SegFormer-B2……")

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
        / "formal_training_log.csv"
    )

    if log_path.exists():
        log_path.unlink()

    best_branch_iou = -1.0

    expected_optimizer_steps = math.ceil(
        len(train_loader)
        / accumulation_steps
    )

    print("=" * 60)
    print("开始有效Batch Size=10的200轮正式训练")
    print(f"训练集：{len(train_dataset)}张")
    print(f"验证集：{len(val_dataset)}张")
    print(
        f"实际微批量：{micro_batch_size}"
    )
    print(
        f"梯度累积次数：{accumulation_steps}"
    )
    print(
        f"有效Batch Size：{effective_batch_size}"
    )
    print(
        f"每轮微批次数：{len(train_loader)}"
    )
    print(
        "每轮优化器更新次数："
        f"{expected_optimizer_steps}"
    )
    print(f"输入尺寸：{input_size}×{input_size}")
    print(f"正式训练轮数：{epochs}")
    print(f"学习率：{learning_rate}")
    print(f"运行设备：{device}")
    print("=" * 60)

    training_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, optimizer_steps = (
            train_one_epoch_with_accumulation(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                accumulation_steps=(
                    accumulation_steps
                ),
            )
        )

        if (
            optimizer_steps
            != expected_optimizer_steps
        ):
            raise RuntimeError(
                "优化器更新次数异常："
                f"预期{expected_optimizer_steps}，"
                f"实际{optimizer_steps}"
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
            "micro_batch_size": (
                micro_batch_size
            ),
            "accumulation_steps": (
                accumulation_steps
            ),
            "effective_batch_size": (
                effective_batch_size
            ),
            "optimizer_steps": (
                optimizer_steps
            ),
            "epoch_minutes": epoch_minutes,
        }

        save_log_row(
            log_path,
            row,
        )

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
            "优化器更新次数："
            f"{optimizer_steps}"
        )
        print(
            f"本轮用时："
            f"{epoch_minutes:.2f}分钟"
        )
        print("混淆矩阵：")
        print(matrix.numpy())
        print("=" * 60)

        checkpoint = {
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
            "micro_batch_size": (
                micro_batch_size
            ),
            "gradient_accumulation_steps": (
                accumulation_steps
            ),
            "effective_batch_size": (
                effective_batch_size
            ),
            "config": config,
        }

        torch.save(
            checkpoint,
            CHECKPOINT_DIR
            / "formal_training_last.pt",
        )

        if (
            metrics["branch_iou"]
            > best_branch_iou
        ):
            best_branch_iou = (
                metrics["branch_iou"]
            )

            best_model_dir = (
                CHECKPOINT_DIR
                / "formal_training_best"
            )

            model.save_pretrained(
                best_model_dir
            )

            processor.save_pretrained(
                best_model_dir
            )

            print(
                "已保存新的最佳模型，"
                f"枝条IoU="
                f"{best_branch_iou:.4f}"
            )

    total_minutes = (
        time.time() - training_start
    ) / 60

    print("\n200轮正式训练完成")
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

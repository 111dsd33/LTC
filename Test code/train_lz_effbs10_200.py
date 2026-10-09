import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from train_baseline import (
    PROJECT_DIR,
    WinterJujubeDataset,
    calculate_metrics,
    update_confusion_matrix,
)


EXPERIMENT_DIR = (
    PROJECT_DIR
    / "07_experiments"
    / "segformer_b2_lz_effbs10"
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


def calculate_lz_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ce_weight: float,
    focal_weight: float,
    dice_weight: float,
    focal_alpha: float,
    focal_gamma: float,
    dice_epsilon: float,
):
    """
    Lz = 1*CE + 2*Focal + 2*Dice。

    CE和Focal对全部有效像素计算。
    Dice只针对branch前景类别计算。
    """

    resized_logits = F.interpolate(
        logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    # 损失计算统一转为float32，提高数值稳定性。
    float_logits = resized_logits.float()

    # 逐像素交叉熵。
    ce_map = F.cross_entropy(
        float_logits,
        labels,
        reduction="none",
    )

    ce_loss = ce_map.mean()

    # 多分类Focal Loss。
    # pt表示真实类别对应的预测概率。
    pt = torch.exp(-ce_map)

    focal_map = (
        focal_alpha
        * torch.pow(1.0 - pt, focal_gamma)
        * ce_map
    )

    focal_loss = focal_map.mean()

    # Dice Loss只计算branch类别，即类别1。
    probabilities = torch.softmax(
        float_logits,
        dim=1,
    )

    branch_probability = probabilities[:, 1]

    branch_target = (
        labels == 1
    ).float()

    intersection = (
        branch_probability
        * branch_target
    ).sum(dim=(1, 2))

    denominator = (
        branch_probability.sum(dim=(1, 2))
        + branch_target.sum(dim=(1, 2))
    )

    dice_score = (
        2.0 * intersection
        + dice_epsilon
    ) / (
        denominator
        + dice_epsilon
    )

    dice_loss = (
        1.0 - dice_score
    ).mean()

    total_loss = (
        ce_weight * ce_loss
        + focal_weight * focal_loss
        + dice_weight * dice_loss
    )

    return {
        "total_loss": total_loss,
        "ce_loss": ce_loss,
        "focal_loss": focal_loss,
        "dice_loss": dice_loss,
        "resized_logits": resized_logits,
    }


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    use_amp: bool,
    accumulation_steps: int,
    loss_config: dict,
):
    model.train()

    totals = {
        "total_loss": 0.0,
        "ce_loss": 0.0,
        "focal_loss": 0.0,
        "dice_loss": 0.0,
    }

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
        pixel_values = batch[
            "pixel_values"
        ].to(
            device,
            non_blocking=True,
        )

        labels = batch[
            "labels"
        ].to(
            device,
            non_blocking=True,
        )

        current_batch_size = (
            pixel_values.size(0)
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(
                pixel_values=pixel_values
            )

            losses = calculate_lz_loss(
                logits=outputs.logits,
                labels=labels,
                ce_weight=float(
                    loss_config[
                        "cross_entropy_weight"
                    ]
                ),
                focal_weight=float(
                    loss_config[
                        "focal_weight"
                    ]
                ),
                dice_weight=float(
                    loss_config[
                        "dice_weight"
                    ]
                ),
                focal_alpha=float(
                    loss_config[
                        "focal_alpha"
                    ]
                ),
                focal_gamma=float(
                    loss_config[
                        "focal_gamma"
                    ]
                ),
                dice_epsilon=float(
                    loss_config[
                        "dice_epsilon"
                    ]
                ),
            )

            loss = losses["total_loss"]

        if not math.isfinite(loss.item()):
            raise ValueError(
                f"Lz训练损失异常：{loss.item()}"
            )

        # 模型损失是微批次平均值。
        # 乘以样本数后反向传播，累积结束再除以总样本数。
        weighted_loss = (
            loss * current_batch_size
        )

        scaler.scale(
            weighted_loss
        ).backward()

        accumulated_samples += (
            current_batch_size
        )

        total_samples += (
            current_batch_size
        )

        for key in totals:
            totals[key] += (
                losses[key].item()
                * current_batch_size
            )

        accumulation_boundary = (
            micro_step
            % accumulation_steps
            == 0
        )

        last_micro_batch = (
            micro_step
            == len(dataloader)
        )

        if (
            accumulation_boundary
            or last_micro_batch
        ):
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
            lz=f"{loss.item():.4f}",
            updates=optimizer_step_count,
        )

    average_losses = {
        key: value / total_samples
        for key, value in totals.items()
    }

    return (
        average_losses,
        optimizer_step_count,
    )


@torch.inference_mode()
def validate_one_epoch(
    model,
    dataloader,
    device,
    use_amp: bool,
    loss_config: dict,
):
    model.eval()

    totals = {
        "total_loss": 0.0,
        "ce_loss": 0.0,
        "focal_loss": 0.0,
        "dice_loss": 0.0,
    }

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
        pixel_values = batch[
            "pixel_values"
        ].to(
            device,
            non_blocking=True,
        )

        labels = batch[
            "labels"
        ].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(
                pixel_values=pixel_values
            )

            losses = calculate_lz_loss(
                logits=outputs.logits,
                labels=labels,
                ce_weight=float(
                    loss_config[
                        "cross_entropy_weight"
                    ]
                ),
                focal_weight=float(
                    loss_config[
                        "focal_weight"
                    ]
                ),
                dice_weight=float(
                    loss_config[
                        "dice_weight"
                    ]
                ),
                focal_alpha=float(
                    loss_config[
                        "focal_alpha"
                    ]
                ),
                focal_gamma=float(
                    loss_config[
                        "focal_gamma"
                    ]
                ),
                dice_epsilon=float(
                    loss_config[
                        "dice_epsilon"
                    ]
                ),
            )

        prediction = (
            losses["resized_logits"]
            .argmax(dim=1)
        )

        update_confusion_matrix(
            confusion_matrix,
            prediction,
            labels,
        )

        current_batch_size = (
            pixel_values.size(0)
        )

        total_samples += (
            current_batch_size
        )

        for key in totals:
            totals[key] += (
                losses[key].item()
                * current_batch_size
            )

    average_losses = {
        key: value / total_samples
        for key, value in totals.items()
    }

    metrics = calculate_metrics(
        confusion_matrix
    )

    return (
        average_losses,
        metrics,
        confusion_matrix,
    )


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
        config[
            "gradient_accumulation_steps"
        ]
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

    loss_config = config["loss"]

    calculated_effective_batch = (
        micro_batch_size
        * accumulation_steps
    )

    if (
        calculated_effective_batch
        != effective_batch_size
    ):
        raise ValueError(
            "有效Batch Size配置不一致："
            f"{micro_batch_size} × "
            f"{accumulation_steps} = "
            f"{calculated_effective_batch}，"
            f"配置文件写的是"
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
        split_file=(
            PROJECT_DIR
            / config["train_split"]
        ),
        processor=processor,
    )

    val_dataset = WinterJujubeDataset(
        split_file=(
            PROJECT_DIR
            / config["val_split"]
        ),
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
        raise FileExistsError(
            "formal_training_log.csv已经存在。"
            "为防止覆盖旧实验，本次训练已停止。"
        )

    implementation_record = {
        "loss_formula": (
            "Lz = 1*CE + 2*Focal + 2*Dice"
        ),
        "focal_scope": "all_valid_pixels",
        "focal_alpha": float(
            loss_config["focal_alpha"]
        ),
        "focal_gamma": float(
            loss_config["focal_gamma"]
        ),
        "dice_scope": (
            "branch_foreground_only"
        ),
        "loss_weight_normalization": False,
        "effective_batch_size": (
            effective_batch_size
        ),
        "random_seed": seed,
    }

    (
        EXPERIMENT_DIR
        / "loss_implementation.json"
    ).write_text(
        json.dumps(
            implementation_record,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    expected_optimizer_steps = math.ceil(
        len(train_loader)
        / accumulation_steps
    )

    best_branch_iou = -1.0
    training_start = time.time()

    print("=" * 60)
    print("开始Lz损失200轮正式训练")
    print(f"训练集：{len(train_dataset)}张")
    print(f"验证集：{len(val_dataset)}张")
    print(f"实际微批量：{micro_batch_size}")
    print(f"梯度累积次数：{accumulation_steps}")
    print(f"有效Batch Size：{effective_batch_size}")
    print(f"每轮优化器更新：{expected_optimizer_steps}")
    print(f"训练轮数：{epochs}")
    print("损失权重：CE:Focal:Dice = 1:2:2")
    print(
        "Focal参数："
        f"alpha={loss_config['focal_alpha']}，"
        f"gamma={loss_config['focal_gamma']}"
    )
    print("Dice范围：branch前景类别")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_losses, optimizer_steps = (
            train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                accumulation_steps=(
                    accumulation_steps
                ),
                loss_config=loss_config,
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

        val_losses, metrics, matrix = (
            validate_one_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                use_amp=use_amp,
                loss_config=loss_config,
            )
        )

        epoch_minutes = (
            time.time() - epoch_start
        ) / 60

        row = {
            "epoch": epoch,
            "train_lz": (
                train_losses["total_loss"]
            ),
            "train_ce": (
                train_losses["ce_loss"]
            ),
            "train_focal": (
                train_losses["focal_loss"]
            ),
            "train_dice": (
                train_losses["dice_loss"]
            ),
            "val_lz": (
                val_losses["total_loss"]
            ),
            "val_ce": (
                val_losses["ce_loss"]
            ),
            "val_focal": (
                val_losses["focal_loss"]
            ),
            "val_dice": (
                val_losses["dice_loss"]
            ),
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
            "optimizer_steps": (
                optimizer_steps
            ),
            "epoch_minutes": (
                epoch_minutes
            ),
        }

        save_log_row(log_path, row)

        print("\n" + "=" * 60)
        print(f"Epoch {epoch}/{epochs}")
        print(
            "训练Lz："
            f"{train_losses['total_loss']:.6f}"
        )
        print(
            "训练CE/Focal/Dice："
            f"{train_losses['ce_loss']:.6f} / "
            f"{train_losses['focal_loss']:.6f} / "
            f"{train_losses['dice_loss']:.6f}"
        )
        print(
            "验证Lz："
            f"{val_losses['total_loss']:.6f}"
        )
        print(
            "验证CE/Focal/Dice："
            f"{val_losses['ce_loss']:.6f} / "
            f"{val_losses['focal_loss']:.6f} / "
            f"{val_losses['dice_loss']:.6f}"
        )
        print(
            f"枝条IoU："
            f"{metrics['branch_iou']:.4f}"
        )
        print(
            f"mIoU：{metrics['miou']:.4f}"
        )
        print(
            f"mPA：{metrics['mpa']:.4f}"
        )
        print(
            "枝条Precision："
            f"{metrics['branch_precision']:.4f}"
        )
        print(
            "枝条Recall："
            f"{metrics['branch_recall']:.4f}"
        )
        print(
            f"枝条F1："
            f"{metrics['branch_f1']:.4f}"
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
            "best_branch_iou": (
                best_branch_iou
            ),
            "config": config,
            "implementation_record": (
                implementation_record
            ),
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

            best_metrics = {
                "epoch": epoch,
                "branch_iou": (
                    metrics["branch_iou"]
                ),
                "miou": metrics["miou"],
                "mpa": metrics["mpa"],
                "branch_precision": (
                    metrics[
                        "branch_precision"
                    ]
                ),
                "branch_recall": (
                    metrics["branch_recall"]
                ),
                "branch_f1": (
                    metrics["branch_f1"]
                ),
                "val_lz": (
                    val_losses["total_loss"]
                ),
            }

            (
                CHECKPOINT_DIR
                / "best_validation_metrics.json"
            ).write_text(
                json.dumps(
                    best_metrics,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                "已保存新的最佳Lz模型，"
                f"枝条IoU="
                f"{best_branch_iou:.4f}"
            )

    total_minutes = (
        time.time() - training_start
    ) / 60

    print("\nLz损失200轮训练完成")
    print(
        f"最佳验证集枝条IoU："
        f"{best_branch_iou:.4f}"
    )
    print(
        f"总用时："
        f"{total_minutes:.2f}分钟"
    )
    print(f"日志文件：{log_path}")
    print(f"模型目录：{CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()

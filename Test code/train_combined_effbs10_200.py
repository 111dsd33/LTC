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
from transformers import SegformerImageProcessor

from combined_model import (
    build_combined_model,
    save_combined_implementation_record,
    validate_combined_structure,
)
from tpsa_model import TPSAStageWrapper
from train_baseline import (
    PROJECT_DIR,
    WinterJujubeDataset,
    calculate_metrics,
    update_confusion_matrix,
)
from train_lz_effbs10_200 import calculate_lz_loss


MODEL_NAME = (
    "nvidia/"
    "segformer-b2-finetuned-ade-512-512"
)

EXPERIMENT_DIR = (
    PROJECT_DIR
    / "07_experiments"
    / "segformer_b2_tpsa_ceca_lz_effbs10"
)

CONFIG_PATH = (
    EXPERIMENT_DIR
    / "config.json"
)

CHECKPOINT_DIR = (
    EXPERIMENT_DIR
    / "checkpoints"
)

LOG_DIR = (
    EXPERIMENT_DIR
    / "logs"
)

LOG_PATH = (
    LOG_DIR
    / "formal_training_log.csv"
)

LAST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "formal_training_last.pt"
)

BEST_STATE_PATH = (
    CHECKPOINT_DIR
    / "formal_training_best_state.pt"
)

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "formal_training_best_checkpoint.pt"
)

BEST_METRICS_PATH = (
    CHECKPOINT_DIR
    / "best_validation_metrics.json"
)


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


def get_tpsa_stage_scales(model) -> dict:
    stage_scales = {}

    for stage_index, stage in enumerate(
        model.segformer.stages,
        start=1,
    ):
        if not isinstance(
            stage,
            TPSAStageWrapper,
        ):
            raise TypeError(
                f"Stage {stage_index}"
                "不是TPSA包装器"
            )

        stage_scales[
            f"stage{stage_index}_scale"
        ] = float(
            stage.tpsa
            .residual_scale
            .detach()
            .cpu()
            .item()
        )

    return stage_scales


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

    total_values = {
        "total_loss": 0.0,
        "ce_loss": 0.0,
        "focal_loss": 0.0,
        "dice_loss": 0.0,
    }

    total_samples = 0
    accumulated_samples = 0
    optimizer_step_count = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    progress = tqdm(
        enumerate(
            dataloader,
            start=1,
        ),
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

        current_batch_size = int(
            pixel_values.size(0)
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(
                pixel_values=pixel_values,
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

            total_loss = losses[
                "total_loss"
            ]

        if not math.isfinite(
            total_loss.item()
        ):
            raise ValueError(
                "联合模型训练损失异常："
                f"{total_loss.item()}"
            )

        # 微批次损失是样本平均值。
        # 乘以实际样本数后累计，更新前再除以
        # 本次累积的真实样本数。
        weighted_loss = (
            total_loss
            * current_batch_size
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

        for key in total_values:
            total_values[key] += (
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
            lz=(
                f"{total_loss.item():.4f}"
            ),
            updates=optimizer_step_count,
        )

    average_values = {
        key: value / total_samples
        for key, value
        in total_values.items()
    }

    return (
        average_values,
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

    total_values = {
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
                pixel_values=pixel_values,
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

        predictions = (
            losses["resized_logits"]
            .argmax(dim=1)
        )

        update_confusion_matrix(
            confusion_matrix,
            predictions,
            labels,
        )

        current_batch_size = int(
            pixel_values.size(0)
        )

        total_samples += (
            current_batch_size
        )

        for key in total_values:
            total_values[key] += (
                losses[key].item()
                * current_batch_size
            )

    average_values = {
        key: value / total_samples
        for key, value
        in total_values.items()
    }

    metrics = calculate_metrics(
        confusion_matrix
    )

    return (
        average_values,
        metrics,
        confusion_matrix,
    )


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"找不到配置文件："
            f"{CONFIG_PATH}"
        )

    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    seed = int(
        config["random_seed"]
    )

    input_size = int(
        config["input_size"]
    )

    micro_batch_size = int(
        config["micro_batch_size"]
    )

    accumulation_steps = int(
        config[
            "gradient_accumulation_steps"
        ]
    )

    effective_batch_size = int(
        config[
            "effective_batch_size"
        ]
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

    modules = config["modules"]

    if not all(
        [
            bool(
                modules["tpsa_enabled"]
            ),
            bool(
                modules["ceca_enabled"]
            ),
            bool(
                modules["lz_enabled"]
            ),
        ]
    ):
        raise ValueError(
            "三模块联合配置未全部启用"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "没有检测到可用的CUDA显卡"
        )

    if LOG_PATH.exists():
        raise FileExistsError(
            "formal_training_log.csv已经存在。"
            "为避免覆盖旧实验，训练已停止。"
        )

    set_random_seed(seed)

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

    print(
        "正在从ADE20K预训练权重创建"
        "TPSA+CECA+Lz联合模型……"
    )

    model = build_combined_model(
        model_name_or_path=MODEL_NAME,
        num_labels=2,
        local_files_only=True,
        state_dict_path=None,
    )

    validate_combined_structure(model)

    model.to(device)

    save_combined_implementation_record(
        model=model,
        output_path=(
            EXPERIMENT_DIR
            / "combined_implementation.json"
        ),
    )

    processor.save_pretrained(
        CHECKPOINT_DIR
        / "processor"
    )

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

    expected_optimizer_steps = math.ceil(
        len(train_loader)
        / accumulation_steps
    )

    tpsa_parameters = sum(
        parameter.numel()
        for name, parameter
        in model.named_parameters()
        if ".tpsa." in name
    )

    ceca_parameters = sum(
        parameter.numel()
        for name, parameter
        in model.named_parameters()
        if ".ceca." in name
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    best_branch_iou = -1.0
    best_epoch = 0

    training_start = time.time()

    print("=" * 70)
    print(
        "开始TPSA + CECA + Lz"
        "三模块联合200轮正式训练"
    )
    print(
        f"训练集数量："
        f"{len(train_dataset)}"
    )
    print(
        f"验证集数量："
        f"{len(val_dataset)}"
    )
    print(
        f"微批量："
        f"{micro_batch_size}"
    )
    print(
        f"梯度累积："
        f"{accumulation_steps}"
    )
    print(
        f"有效Batch Size："
        f"{effective_batch_size}"
    )
    print(
        f"每轮微批次数："
        f"{len(train_loader)}"
    )
    print(
        f"每轮优化器更新："
        f"{expected_optimizer_steps}"
    )
    print(
        f"训练轮数：{epochs}"
    )
    print("TPSA：启用")
    print("CECA：启用")
    print("Lz：启用")
    print(
        "损失："
        "1*CE + 2*Focal + 2*Dice"
    )
    print(
        "单模块训练权重加载：否"
    )
    print(
        f"TPSA参数量："
        f"{tpsa_parameters}"
    )
    print(
        f"CECA参数量："
        f"{ceca_parameters}"
    )
    print(
        f"模型总参数量："
        f"{total_parameters}"
    )

    print("TPSA阶段配置：")

    for stage_index, stage in enumerate(
        model.segformer.stages,
        start=1,
    ):
        print(
            f"Stage {stage_index}: "
            f"C={stage.tpsa.channels}, "
            f"S={stage.tpsa.num_branches}, "
            f"Top-k={stage.tpsa.top_k}, "
            f"Scale="
            f"{stage.tpsa.residual_scale.item():.4f}"
        )

    print(
        "CECA卷积核大小："
        f"{model.decode_head.ceca.kernel_size}"
    )
    print("=" * 70)

    for epoch in range(
        1,
        epochs + 1,
    ):
        epoch_start = time.time()

        (
            train_losses,
            optimizer_steps,
        ) = train_one_epoch(
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

        if (
            optimizer_steps
            != expected_optimizer_steps
        ):
            raise RuntimeError(
                "优化器更新次数异常："
                f"预期{expected_optimizer_steps}，"
                f"实际{optimizer_steps}"
            )

        (
            val_losses,
            metrics,
            confusion_matrix,
        ) = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            use_amp=use_amp,
            loss_config=loss_config,
        )

        stage_scales = (
            get_tpsa_stage_scales(
                model
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
            **stage_scales,
        }

        save_log_row(
            LOG_PATH,
            row,
        )

        print(
            "\n"
            + "=" * 70
        )
        print(
            f"Epoch {epoch}/{epochs}"
        )
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
            "背景IoU："
            f"{metrics['background_iou']:.4f}"
        )
        print(
            "枝条IoU："
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
            "枝条Precision："
            f"{metrics['branch_precision']:.4f}"
        )
        print(
            "枝条Recall："
            f"{metrics['branch_recall']:.4f}"
        )
        print(
            "枝条F1："
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

        print("TPSA残差缩放：")

        for key, value in (
            stage_scales.items()
        ):
            print(
                f"  {key}: "
                f"{value:.6f}"
            )

        print("混淆矩阵：")
        print(
            confusion_matrix.numpy()
        )
        print("=" * 70)

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
            "train_losses": (
                train_losses
            ),
            "val_losses": (
                val_losses
            ),
            "metrics": metrics,
            "stage_scales": (
                stage_scales
            ),
            "best_branch_iou": (
                best_branch_iou
            ),
            "best_epoch": (
                best_epoch
            ),
            "config": config,
        }

        torch.save(
            checkpoint,
            LAST_CHECKPOINT_PATH,
        )

        if (
            metrics["branch_iou"]
            > best_branch_iou
        ):
            best_branch_iou = float(
                metrics["branch_iou"]
            )

            best_epoch = epoch

            torch.save(
                model.state_dict(),
                BEST_STATE_PATH,
            )

            best_checkpoint = {
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
                "train_losses": (
                    train_losses
                ),
                "val_losses": (
                    val_losses
                ),
                "metrics": metrics,
                "stage_scales": (
                    stage_scales
                ),
                "config": config,
            }

            torch.save(
                best_checkpoint,
                BEST_CHECKPOINT_PATH,
            )

            best_metrics = {
                "epoch": epoch,
                "branch_iou": (
                    metrics["branch_iou"]
                ),
                "background_iou": (
                    metrics["background_iou"]
                ),
                "miou": (
                    metrics["miou"]
                ),
                "mpa": (
                    metrics["mpa"]
                ),
                "branch_precision": (
                    metrics[
                        "branch_precision"
                    ]
                ),
                "branch_recall": (
                    metrics[
                        "branch_recall"
                    ]
                ),
                "branch_f1": (
                    metrics["branch_f1"]
                ),
                "branch_dice": (
                    metrics["branch_dice"]
                ),
                "train_lz": (
                    train_losses[
                        "total_loss"
                    ]
                ),
                "train_ce": (
                    train_losses[
                        "ce_loss"
                    ]
                ),
                "train_focal": (
                    train_losses[
                        "focal_loss"
                    ]
                ),
                "train_dice": (
                    train_losses[
                        "dice_loss"
                    ]
                ),
                "val_lz": (
                    val_losses[
                        "total_loss"
                    ]
                ),
                "val_ce": (
                    val_losses[
                        "ce_loss"
                    ]
                ),
                "val_focal": (
                    val_losses[
                        "focal_loss"
                    ]
                ),
                "val_dice": (
                    val_losses[
                        "dice_loss"
                    ]
                ),
                "stage_scales": (
                    stage_scales
                ),
            }

            BEST_METRICS_PATH.write_text(
                json.dumps(
                    best_metrics,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                "已保存新的最佳三模块模型，"
                f"Epoch={epoch}，"
                f"枝条IoU="
                f"{best_branch_iou:.4f}"
            )

    total_minutes = (
        time.time() - training_start
    ) / 60

    print(
        "\n三模块联合200轮训练完成"
    )
    print(
        f"最佳轮次：{best_epoch}"
    )
    print(
        "最佳验证集枝条IoU："
        f"{best_branch_iou:.4f}"
    )
    print(
        f"总用时："
        f"{total_minutes:.2f}分钟"
    )
    print(
        f"日志文件：{LOG_PATH}"
    )
    print(
        "最佳模型状态字典："
        f"{BEST_STATE_PATH}"
    )


if __name__ == "__main__":
    main()

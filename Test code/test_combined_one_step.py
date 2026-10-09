import json
import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor

from ceca_model import (
    CECASegformerDecodeHead,
)
from combined_model import (
    build_combined_model,
    save_combined_implementation_record,
    validate_combined_structure,
)
from tpsa_model import TPSAStageWrapper
from train_baseline import (
    PROJECT_DIR,
    WinterJujubeDataset,
)
from train_lz_effbs10_200 import (
    calculate_lz_loss,
)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gradient_absolute_sum(
    parameters,
) -> float:
    total = 0.0

    for parameter in parameters:
        if parameter.grad is not None:
            total += (
                parameter.grad
                .detach()
                .abs()
                .sum()
                .item()
            )

    return total


def main() -> None:
    set_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "没有检测到CUDA显卡"
        )

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"找不到配置文件：{CONFIG_PATH}"
        )

    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    loss_config = config["loss"]

    device = torch.device("cuda")

    processor = (
        SegformerImageProcessor
        .from_pretrained(
            MODEL_NAME,
            do_reduce_labels=False,
            size={
                "height": 512,
                "width": 512,
            },
            local_files_only=True,
        )
    )

    dataset = WinterJujubeDataset(
        split_file=(
            PROJECT_DIR
            / "03_数据集划分"
            / "train.txt"
        ),
        processor=processor,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    print(
        "正在从ADE20K预训练权重"
        "创建TPSA+CECA联合模型……"
    )

    model = build_combined_model(
        model_name_or_path=MODEL_NAME,
        num_labels=2,
        local_files_only=True,
        state_dict_path=None,
    )

    validate_combined_structure(model)

    model.to(device)
    model.train()

    if not isinstance(
        model.decode_head,
        CECASegformerDecodeHead,
    ):
        raise RuntimeError(
            "CECA解码头检查失败"
        )

    tpsa_parameter_count = sum(
        parameter.numel()
        for name, parameter
        in model.named_parameters()
        if ".tpsa." in name
    )

    ceca_parameter_count = sum(
        parameter.numel()
        for name, parameter
        in model.named_parameters()
        if ".ceca." in name
    )

    total_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    batch = next(iter(loader))

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

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    torch.cuda.reset_peak_memory_stats(
        device
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
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

    loss_values = {
        "total": total_loss.item(),
        "ce": losses["ce_loss"].item(),
        "focal": (
            losses["focal_loss"].item()
        ),
        "dice": (
            losses["dice_loss"].item()
        ),
    }

    for loss_name, value in (
        loss_values.items()
    ):
        if not math.isfinite(value):
            raise ValueError(
                f"{loss_name}损失异常："
                f"{value}"
            )

    scaler.scale(
        total_loss
    ).backward()

    scaler.unscale_(optimizer)

    stage_gradient_sums = {}

    for stage_index, stage in enumerate(
        model.segformer.stages,
        start=1,
    ):
        if not isinstance(
            stage,
            TPSAStageWrapper,
        ):
            raise RuntimeError(
                f"Stage {stage_index}"
                "不是TPSA包装器"
            )

        gradient_sum = (
            gradient_absolute_sum(
                stage.tpsa.parameters()
            )
        )

        stage_gradient_sums[
            stage_index
        ] = gradient_sum

        if gradient_sum <= 0.0:
            raise RuntimeError(
                f"Stage {stage_index} TPSA"
                "没有获得有效梯度"
            )

    ceca_gradient_sum = (
        gradient_absolute_sum(
            model.decode_head
            .ceca
            .parameters()
        )
    )

    if ceca_gradient_sum <= 0.0:
        raise RuntimeError(
            "CECA没有获得有效梯度"
        )

    classifier_gradient_sum = (
        gradient_absolute_sum(
            model.decode_head
            .classifier
            .parameters()
        )
    )

    if classifier_gradient_sum <= 0.0:
        raise RuntimeError(
            "分类器没有获得有效梯度"
        )

    scaler.step(optimizer)
    scaler.update()

    peak_memory_gb = (
        torch.cuda.max_memory_allocated(
            device
        )
        / 1024**3
    )

    implementation_path = (
        EXPERIMENT_DIR
        / "combined_implementation.json"
    )

    save_combined_implementation_record(
        model=model,
        output_path=implementation_path,
    )

    print("=" * 68)
    print("三模块联合单步训练测试通过")
    print(
        f"输入形状："
        f"{tuple(pixel_values.shape)}"
    )
    print(
        f"标签形状："
        f"{tuple(labels.shape)}"
    )
    print(
        f"输出形状："
        f"{tuple(outputs.logits.shape)}"
    )

    print("\nLz损失：")
    print(
        f"  总损失："
        f"{loss_values['total']:.6f}"
    )
    print(
        f"  CE："
        f"{loss_values['ce']:.6f}"
    )
    print(
        f"  Focal："
        f"{loss_values['focal']:.6f}"
    )
    print(
        f"  Dice："
        f"{loss_values['dice']:.6f}"
    )

    print("\n参数量：")
    print(
        f"  TPSA新增参数量："
        f"{tpsa_parameter_count}"
    )
    print(
        f"  CECA新增参数量："
        f"{ceca_parameter_count}"
    )
    print(
        f"  模型总参数量："
        f"{total_parameter_count}"
    )

    print("\nTPSA梯度：")

    for stage_index, gradient_sum in (
        stage_gradient_sums.items()
    ):
        stage = (
            model.segformer
            .stages[stage_index - 1]
        )

        print(
            f"  Stage {stage_index}: "
            f"Top-k={stage.tpsa.top_k}, "
            f"梯度和={gradient_sum:.6f}"
        )

    print(
        "\nCECA梯度绝对值总和："
        f"{ceca_gradient_sum:.6f}"
    )

    print(
        "分类器梯度绝对值总和："
        f"{classifier_gradient_sum:.6f}"
    )

    print(
        f"峰值显存："
        f"{peak_memory_gb:.2f} GB"
    )

    print(
        f"实现记录："
        f"{implementation_path}"
    )

    print(
        "单模块训练权重加载：否"
    )
    print("=" * 68)


if __name__ == "__main__":
    main()

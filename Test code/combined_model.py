from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation

from ceca_model import (
    CECASegformerDecodeHead,
    attach_ceca,
    get_ceca_implementation_record,
)
from tpsa_model import (
    TPSAStageWrapper,
    attach_tpsa,
    get_tpsa_implementation_record,
)


def build_combined_model(
    model_name_or_path: str | Path,
    num_labels: int = 2,
    local_files_only: bool = True,
    state_dict_path: str | Path | None = None,
) -> SegformerForSemanticSegmentation:
    """
    从ADE20K预训练SegFormer-B2开始构建联合模型。

    安装顺序：
    1. TPSA安装到四个编码器Stage输出之后。
    2. CECA安装到解码器融合特征与分类器之间。
    3. 可选地严格加载联合模型状态字典。

    本函数不会加载任何单模块训练权重。
    """

    model = (
        SegformerForSemanticSegmentation
        .from_pretrained(
            str(model_name_or_path),
            num_labels=num_labels,
            id2label={
                0: "background",
                1: "branch",
            },
            label2id={
                "background": 0,
                "branch": 1,
            },
            ignore_mismatched_sizes=True,
            local_files_only=local_files_only,
        )
    )

    model = attach_tpsa(
        model=model,
        num_branches=4,
        top_k_per_stage=(4, 3, 2, 2),
        enabled_stages=(1, 2, 3, 4),
        excitation_reduction=16,
        residual_scale_init=0.1,
    )

    model = attach_ceca(
        model=model,
        gamma=2.0,
        b=1.0,
    )

    if state_dict_path is not None:
        state_dict_path = Path(
            state_dict_path
        )

        if not state_dict_path.exists():
            raise FileNotFoundError(
                "找不到联合模型状态字典："
                f"{state_dict_path}"
            )

        state_dict = torch.load(
            state_dict_path,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    return model


def validate_combined_structure(
    model: SegformerForSemanticSegmentation,
) -> None:
    """
    检查TPSA和CECA是否同时正确安装。
    """

    if not hasattr(
        model.segformer,
        "stages",
    ):
        raise AttributeError(
            "SegFormer主干缺少stages"
        )

    if len(model.segformer.stages) != 4:
        raise ValueError(
            "编码器Stage数量不是4："
            f"{len(model.segformer.stages)}"
        )

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
                "没有正确安装TPSA"
            )

    if not isinstance(
        model.decode_head,
        CECASegformerDecodeHead,
    ):
        raise TypeError(
            "解码头没有正确安装CECA"
        )


def get_combined_implementation_record(
    model: SegformerForSemanticSegmentation,
) -> dict:
    validate_combined_structure(model)

    return {
        "experiment": (
            "TPSA_CECA_Lz_combination"
        ),
        "initialization": {
            "base_model": (
                "ADE20K_pretrained_"
                "SegFormer_B2"
            ),
            "segmentation_head": (
                "new_two_class_head"
            ),
            "tpsa": (
                "fresh_initialization"
            ),
            "ceca": (
                "fresh_initialization"
            ),
            "single_module_checkpoint_loaded": (
                False
            ),
        },
        "model_forward_order": [
            "SegFormer encoder stage",
            "TPSA stage enhancement",
            "next encoder stage",
            "SegFormer decoder fusion",
            "CECA channel enhancement",
            "dropout and classifier",
        ],
        "tpsa": (
            get_tpsa_implementation_record(
                model
            )
        ),
        "ceca": (
            get_ceca_implementation_record(
                model
            )
        ),
        "loss": {
            "name": "Lz",
            "formula": (
                "1*CE + 2*Focal + 2*Dice"
            ),
            "focal_alpha": 1.0,
            "focal_gamma": 2.0,
            "dice_scope": (
                "branch_foreground_only"
            ),
            "dice_epsilon": 1e-6,
            "weight_normalization": False,
        },
    }


def save_combined_implementation_record(
    model: SegformerForSemanticSegmentation,
    output_path: str | Path,
) -> None:
    record = (
        get_combined_implementation_record(
            model
        )
    )

    Path(output_path).write_text(
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

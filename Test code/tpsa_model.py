from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
from transformers import (
    SegformerForSemanticSegmentation,
)


class TPSAModule(nn.Module):
    """
    TPSA分支稀疏化模块。

    实现流程：
    1. 沿通道维度均分为S个分支。
    2. 各分支通过1×1卷积投影回完整通道数。
    3. 对每个分支进行全局平均池化。
    4. 拼接描述向量并通过Excitation计算分支分数。
    5. Softmax归一化。
    6. 仅保留Top-k分支并重新归一化。
    7. 分支加权求和。
    8. 通过残差形式增强原始特征。
    """

    def __init__(
        self,
        channels: int,
        num_branches: int = 4,
        top_k: int = 2,
        excitation_reduction: int = 16,
        residual_scale_init: float = 0.1,
    ) -> None:
        super().__init__()

        self.channels = int(channels)
        self.num_branches = int(num_branches)
        self.top_k = int(top_k)
        self.excitation_reduction = int(
            excitation_reduction
        )

        if self.channels <= 0:
            raise ValueError(
                "channels必须大于0"
            )

        if self.num_branches <= 0:
            raise ValueError(
                "num_branches必须大于0"
            )

        if (
            self.channels
            % self.num_branches
            != 0
        ):
            raise ValueError(
                "通道数必须能够被分支数整除："
                f"{self.channels}不能被"
                f"{self.num_branches}整除"
            )

        if not (
            1
            <= self.top_k
            <= self.num_branches
        ):
            raise ValueError(
                "top_k必须位于1到分支数量之间"
            )

        self.branch_channels = (
            self.channels
            // self.num_branches
        )

        # 每个通道分组分别投影为完整通道特征，
        # 使不同分支可以执行加权求和。
        self.branch_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=(
                            self.branch_channels
                        ),
                        out_channels=self.channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        self.channels
                    ),
                    nn.GELU(),
                )
                for _ in range(
                    self.num_branches
                )
            ]
        )

        excitation_hidden_size = max(
            self.channels
            // self.excitation_reduction,
            self.num_branches * 2,
        )

        self.excitation = nn.Sequential(
            nn.Linear(
                self.channels
                * self.num_branches,
                excitation_hidden_size,
                bias=True,
            ),
            nn.GELU(),
            nn.Linear(
                excitation_hidden_size,
                self.num_branches,
                bias=True,
            ),
        )

        self.global_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        # 残差缩放用于减少新模块对预训练特征的
        # 初始破坏。该参数参与训练。
        self.residual_scale = nn.Parameter(
            torch.tensor(
                float(residual_scale_init),
                dtype=torch.float32,
            )
        )

        # 仅用于检查和记录最近一次前向传播。
        self.last_soft_weights = None
        self.last_sparse_weights = None
        self.last_topk_indices = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for branch in self.branch_projections:
            convolution = branch[0]
            batch_norm = branch[1]

            nn.init.kaiming_normal_(
                convolution.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

            nn.init.ones_(
                batch_norm.weight
            )

            nn.init.zeros_(
                batch_norm.bias
            )

        first_linear = self.excitation[0]
        last_linear = self.excitation[2]

        nn.init.kaiming_uniform_(
            first_linear.weight,
            nonlinearity="linear",
        )
        nn.init.zeros_(
            first_linear.bias
        )

        # 使用较小初始分支分数，避免一开始过度偏向
        # 某个分支。
        nn.init.normal_(
            last_linear.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(
            last_linear.bias
        )

    def forward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(
                "TPSA输入必须为[B,C,H,W]，"
                f"实际形状为{feature.shape}"
            )

        if (
            feature.shape[1]
            != self.channels
        ):
            raise ValueError(
                "TPSA输入通道不一致："
                f"预期{self.channels}，"
                f"实际{feature.shape[1]}"
            )

        channel_groups = torch.chunk(
            feature,
            chunks=self.num_branches,
            dim=1,
        )

        branch_features = []

        for projection, channel_group in zip(
            self.branch_projections,
            channel_groups,
        ):
            branch_feature = projection(
                channel_group
            )

            branch_features.append(
                branch_feature
            )

        descriptors = [
            self.global_pool(
                branch_feature
            ).flatten(1)
            for branch_feature
            in branch_features
        ]

        joint_descriptor = torch.cat(
            descriptors,
            dim=1,
        )

        branch_logits = self.excitation(
            joint_descriptor
        )

        soft_weights = torch.softmax(
            branch_logits,
            dim=1,
        )

        if (
            self.top_k
            < self.num_branches
        ):
            _, topk_indices = torch.topk(
                soft_weights,
                k=self.top_k,
                dim=1,
                largest=True,
                sorted=True,
            )

            selection_mask = torch.zeros_like(
                soft_weights
            )

            selection_mask.scatter_(
                dim=1,
                index=topk_indices,
                value=1.0,
            )

            sparse_weights = (
                soft_weights
                * selection_mask
            )

            sparse_weights = (
                sparse_weights
                / sparse_weights.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1e-8)
            )
        else:
            topk_indices = torch.arange(
                self.num_branches,
                device=feature.device,
            ).unsqueeze(0).expand(
                feature.shape[0],
                -1,
            )

            sparse_weights = soft_weights

        stacked_features = torch.stack(
            branch_features,
            dim=1,
        )

        weighted_feature = (
            stacked_features
            * sparse_weights[
                :,
                :,
                None,
                None,
                None,
            ]
        ).sum(dim=1)

        enhanced_feature = (
            feature
            + self.residual_scale
            * weighted_feature
        )

        self.last_soft_weights = (
            soft_weights.detach()
        )

        self.last_sparse_weights = (
            sparse_weights.detach()
        )

        self.last_topk_indices = (
            topk_indices.detach()
        )

        return enhanced_feature


class TPSAStageWrapper(nn.Module):
    """
    包装原始SegformerStage，并在其输出后应用TPSA。
    """

    def __init__(
        self,
        original_stage: nn.Module,
        channels: int,
        num_branches: int,
        top_k: int,
        excitation_reduction: int,
        residual_scale_init: float,
    ) -> None:
        super().__init__()

        self.original_stage = original_stage

        self.tpsa = TPSAModule(
            channels=channels,
            num_branches=num_branches,
            top_k=top_k,
            excitation_reduction=(
                excitation_reduction
            ),
            residual_scale_init=(
                residual_scale_init
            ),
        )

    def forward(
        self,
        *args,
        **kwargs,
    ):
        output = self.original_stage(
            *args,
            **kwargs,
        )

        if isinstance(
            output,
            torch.Tensor,
        ):
            return self.tpsa(output)

        # 为兼容可能返回元组的其他Transformers版本。
        if isinstance(output, tuple):
            if len(output) == 0:
                raise RuntimeError(
                    "SegFormer Stage返回空元组"
                )

            enhanced_first = self.tpsa(
                output[0]
            )

            return (
                enhanced_first,
                *output[1:],
            )

        if isinstance(output, list):
            if len(output) == 0:
                raise RuntimeError(
                    "SegFormer Stage返回空列表"
                )

            result = list(output)
            result[0] = self.tpsa(
                result[0]
            )

            return result

        raise TypeError(
            "无法处理SegFormer Stage输出类型："
            f"{type(output)}"
        )


def attach_tpsa(
    model: SegformerForSemanticSegmentation,
    num_branches: int = 4,
    top_k_per_stage: Sequence[int] = (
        4,
        3,
        2,
        2,
    ),
    enabled_stages: Iterable[int] = (
        1,
        2,
        3,
        4,
    ),
    excitation_reduction: int = 16,
    residual_scale_init: float = 0.1,
) -> SegformerForSemanticSegmentation:
    if not hasattr(
        model.segformer,
        "stages",
    ):
        raise AttributeError(
            "当前SegFormer主干没有stages属性"
        )

    stages = model.segformer.stages

    hidden_sizes = list(
        model.config.hidden_sizes
    )

    if len(stages) != len(hidden_sizes):
        raise ValueError(
            "Stage数量与hidden_sizes不一致"
        )

    if (
        len(top_k_per_stage)
        != len(stages)
    ):
        raise ValueError(
            "top_k_per_stage长度必须与"
            "Stage数量一致"
        )

    enabled_stage_set = {
        int(stage_number)
        for stage_number
        in enabled_stages
    }

    for stage_index in range(
        len(stages)
    ):
        stage_number = stage_index + 1

        if (
            stage_number
            not in enabled_stage_set
        ):
            continue

        if isinstance(
            stages[stage_index],
            TPSAStageWrapper,
        ):
            continue

        stages[stage_index] = TPSAStageWrapper(
            original_stage=(
                stages[stage_index]
            ),
            channels=int(
                hidden_sizes[stage_index]
            ),
            num_branches=int(
                num_branches
            ),
            top_k=int(
                top_k_per_stage[
                    stage_index
                ]
            ),
            excitation_reduction=int(
                excitation_reduction
            ),
            residual_scale_init=float(
                residual_scale_init
            ),
        )

    return model


def build_tpsa_model(
    model_name_or_path: str | Path,
    num_labels: int = 2,
    local_files_only: bool = True,
    state_dict_path: (
        str | Path | None
    ) = None,
) -> SegformerForSemanticSegmentation:
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
            local_files_only=(
                local_files_only
            ),
        )
    )

    model = attach_tpsa(model)

    if state_dict_path is not None:
        state_dict = torch.load(
            Path(state_dict_path),
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    return model


def get_tpsa_implementation_record(
    model: SegformerForSemanticSegmentation,
) -> dict:
    stage_records = []

    for stage_index, stage in enumerate(
        model.segformer.stages,
        start=1,
    ):
        if not isinstance(
            stage,
            TPSAStageWrapper,
        ):
            stage_records.append(
                {
                    "stage": stage_index,
                    "enabled": False,
                }
            )
            continue

        module = stage.tpsa

        stage_records.append(
            {
                "stage": stage_index,
                "enabled": True,
                "channels": module.channels,
                "num_branches": (
                    module.num_branches
                ),
                "branch_channels": (
                    module.branch_channels
                ),
                "top_k": module.top_k,
                "excitation_reduction": (
                    module.excitation_reduction
                ),
                "residual_scale_initial": 0.1,
            }
        )

    return {
        "module": "TPSA",
        "placement": (
            "after_each_segformer_stage_output"
        ),
        "enabled_stages": [
            1,
            2,
            3,
            4,
        ],
        "branch_generation": (
            "channel_split_then_1x1_"
            "projection_to_full_channels"
        ),
        "descriptor": (
            "global_average_pooling_"
            "for_each_branch"
        ),
        "excitation": (
            "two_layer_mlp_with_gelu"
        ),
        "weight_normalization": "softmax",
        "selection": "hard_top_k",
        "post_top_k_normalization": True,
        "fusion": "weighted_sum",
        "output": (
            "input_plus_learnable_scale_"
            "times_weighted_feature"
        ),
        "stage_settings": stage_records,
        "source_unspecified_assumptions": [
            "enabled encoder stages",
            "number of branches S",
            "per-stage top-k values",
            "SPC projection implementation",
            "excitation architecture",
            "top-k renormalization",
            "residual connection and scale",
        ],
    }


def save_implementation_record(
    model: SegformerForSemanticSegmentation,
    output_path: str | Path,
) -> None:
    record = get_tpsa_implementation_record(
        model
    )

    Path(output_path).write_text(
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

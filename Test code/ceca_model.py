from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import SegformerForSemanticSegmentation


class CECAModule(nn.Module):
    """
    CECA通道增强模块。

    输入：
        fused_feature：解码器融合特征Ff
        shallow_feature：浅层细节特征Fl
        deep_feature：深层语义特征Fh
    """

    def __init__(
        self,
        channels: int,
        gamma: float = 2.0,
        b: float = 1.0,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError("channels必须为正整数")

        self.channels = int(channels)
        self.gamma = float(gamma)
        self.b = float(b)

        kernel_size = self.calculate_kernel_size(
            channels=self.channels,
            gamma=self.gamma,
            b=self.b,
        )

        self.kernel_size = kernel_size
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 对应论文图中的2C -> C通道投影。
        self.channel_projection = nn.Linear(
            2 * self.channels,
            self.channels,
            bias=True,
        )

        # 在通道序列上进行局部相关性建模。
        self.channel_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    @staticmethod
    def calculate_kernel_size(
        channels: int,
        gamma: float,
        b: float,
    ) -> int:
        value = int(
            abs(
                (
                    math.log2(channels)
                    + b
                )
                / gamma
            )
        )

        kernel_size = (
            value
            if value % 2 == 1
            else value + 1
        )

        return max(kernel_size, 1)

    def reset_parameters(self) -> None:
        """
        将2C->C投影初始化为浅层与深层描述向量的平均，
        使初始状态更稳定、可重复。
        """
        with torch.no_grad():
            self.channel_projection.weight.zero_()
            self.channel_projection.bias.zero_()

            identity = torch.eye(
                self.channels,
                dtype=self.channel_projection.weight.dtype,
            )

            self.channel_projection.weight[
                :,
                : self.channels,
            ].copy_(0.5 * identity)

            self.channel_projection.weight[
                :,
                self.channels :,
            ].copy_(0.5 * identity)

            self.channel_conv.weight.fill_(
                1.0 / self.kernel_size
            )

    def forward(
        self,
        fused_feature: torch.Tensor,
        shallow_feature: torch.Tensor,
        deep_feature: torch.Tensor,
    ) -> torch.Tensor:
        expected_shape = fused_feature.shape

        if shallow_feature.shape != expected_shape:
            raise ValueError(
                "浅层特征与融合特征尺寸不一致："
                f"{shallow_feature.shape} != {expected_shape}"
            )

        if deep_feature.shape != expected_shape:
            raise ValueError(
                "深层特征与融合特征尺寸不一致："
                f"{deep_feature.shape} != {expected_shape}"
            )

        shallow_descriptor = self.global_pool(
            shallow_feature
        ).flatten(1)

        deep_descriptor = self.global_pool(
            deep_feature
        ).flatten(1)

        joint_descriptor = torch.cat(
            [
                shallow_descriptor,
                deep_descriptor,
            ],
            dim=1,
        )

        projected_descriptor = (
            self.channel_projection(
                joint_descriptor
            )
        )

        channel_weights = self.channel_conv(
            projected_descriptor.unsqueeze(1)
        ).squeeze(1)

        channel_weights = self.sigmoid(
            channel_weights
        ).unsqueeze(-1).unsqueeze(-1)

        enhanced_feature = (
            fused_feature * channel_weights
        )

        return enhanced_feature


class CECASegformerDecodeHead(nn.Module):
    """
    在SegFormer原解码头的融合特征与分类器之间加入CECA。

    原解码头已有参数会原样保留，新增参数仅来自CECA。
    """

    REQUIRED_ATTRIBUTES = (
        "linear_projections",
        "linear_fuse",
        "batch_norm",
        "activation",
        "dropout",
        "classifier",
    )

    def __init__(
        self,
        original_decode_head: nn.Module,
        decoder_hidden_size: int,
        num_encoder_blocks: int,
        gamma: float = 2.0,
        b: float = 1.0,
    ) -> None:
        super().__init__()

        missing_attributes = [
            name
            for name in self.REQUIRED_ATTRIBUTES
            if not hasattr(
                original_decode_head,
                name,
            )
        ]

        if missing_attributes:
            raise AttributeError(
                "当前Transformers版本的SegFormer解码头"
                "结构与预期不一致，缺少："
                + ", ".join(missing_attributes)
            )

        # 直接复用原始预训练解码头模块及参数。
        self.linear_projections = original_decode_head.linear_projections
        self.linear_fuse = (
            original_decode_head.linear_fuse
        )
        self.batch_norm = (
            original_decode_head.batch_norm
        )
        self.activation = (
            original_decode_head.activation
        )
        self.dropout = original_decode_head.dropout
        self.classifier = (
            original_decode_head.classifier
        )

        self.num_encoder_blocks = int(
            num_encoder_blocks
        )

        self.ceca = CECAModule(
            channels=int(decoder_hidden_size),
            gamma=gamma,
            b=b,
        )

    def forward(
        self,
        encoder_hidden_states,
    ) -> torch.Tensor:
        if (
            len(encoder_hidden_states)
            != self.num_encoder_blocks
        ):
            raise ValueError(
                "编码器特征数量异常："
                f"预期{self.num_encoder_blocks}，"
                f"实际{len(encoder_hidden_states)}"
            )

        batch_size = (
            encoder_hidden_states[-1].shape[0]
        )

        target_size = (
            encoder_hidden_states[0].shape[-2:]
        )

        projected_states = []

        for stage_index, (
            encoder_hidden_state,
            mlp,
        ) in enumerate(
            zip(
                encoder_hidden_states,
                self.linear_projections,
            ),
            start=1,
        ):
            if encoder_hidden_state.ndim != 4:
                raise ValueError(
                    f"第{stage_index}阶段特征不是4维张量："
                    f"{encoder_hidden_state.shape}"
                )

            height, width = (
                encoder_hidden_state.shape[-2:]
            )

            projected_state = mlp(
                encoder_hidden_state
            )

            projected_state = (
                projected_state
                .permute(0, 2, 1)
                .reshape(
                    batch_size,
                    -1,
                    height,
                    width,
                )
            )

            projected_state = F.interpolate(
                projected_state,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

            projected_states.append(
                projected_state
            )

        # 保持原SegFormer解码头的逆序拼接方式。
        fused_feature = self.linear_fuse(
            torch.cat(
                projected_states[::-1],
                dim=1,
            )
        )

        fused_feature = self.batch_norm(
            fused_feature
        )

        fused_feature = self.activation(
            fused_feature
        )

        # 本次复现的明确映射：
        # Fl = 第1阶段投影特征
        # Fh = 第4阶段投影特征
        shallow_feature = projected_states[0]
        deep_feature = projected_states[-1]

        enhanced_feature = self.ceca(
            fused_feature=fused_feature,
            shallow_feature=shallow_feature,
            deep_feature=deep_feature,
        )

        enhanced_feature = self.dropout(
            enhanced_feature
        )

        logits = self.classifier(
            enhanced_feature
        )

        return logits


def attach_ceca(
    model: SegformerForSemanticSegmentation,
    gamma: float = 2.0,
    b: float = 1.0,
) -> SegformerForSemanticSegmentation:
    if isinstance(
        model.decode_head,
        CECASegformerDecodeHead,
    ):
        return model

    model.decode_head = CECASegformerDecodeHead(
        original_decode_head=model.decode_head,
        decoder_hidden_size=int(
            model.config.decoder_hidden_size
        ),
        num_encoder_blocks=int(
            model.config.num_encoder_blocks
        ),
        gamma=gamma,
        b=b,
    )

    return model


def build_ceca_model(
    model_name_or_path: str | Path,
    num_labels: int = 2,
    local_files_only: bool = True,
    state_dict_path: str | Path | None = None,
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
            local_files_only=local_files_only,
        )
    )

    model = attach_ceca(model)

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


def get_ceca_implementation_record(
    model: SegformerForSemanticSegmentation,
) -> dict:
    if not isinstance(
        model.decode_head,
        CECASegformerDecodeHead,
    ):
        raise TypeError(
            "当前模型尚未安装CECA解码头"
        )

    module = model.decode_head.ceca

    return {
        "module": "CECA",
        "placement": (
            "after_decoder_fusion_activation_"
            "before_dropout_and_classifier"
        ),
        "shallow_feature": (
            "projected_and_upsampled_encoder_stage_1"
        ),
        "deep_feature": (
            "projected_and_upsampled_encoder_stage_4"
        ),
        "descriptor_operation": (
            "adaptive_global_average_pooling"
        ),
        "descriptor_fusion": (
            "concatenate_shallow_and_deep"
        ),
        "projection": "linear_2C_to_C",
        "adaptive_kernel_formula": (
            "t=int(abs((log2(C)+b)/gamma)); "
            "k=t if odd else t+1"
        ),
        "channels": module.channels,
        "gamma": module.gamma,
        "b": module.b,
        "kernel_size": module.kernel_size,
        "activation": "sigmoid",
        "output": (
            "fused_feature_times_channel_weights"
        ),
        "source_unspecified_assumptions": [
            "encoder stage mapping for Fl and Fh",
            "complete phi(C) kernel formula",
            "initialization method",
        ],
    }


def save_implementation_record(
    model: SegformerForSemanticSegmentation,
    output_path: str | Path,
) -> None:
    record = get_ceca_implementation_record(
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

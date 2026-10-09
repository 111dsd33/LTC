import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor

from combined_model import (
    build_combined_model,
    validate_combined_structure,
)
from train_baseline import (
    PROJECT_DIR,
    WinterJujubeDataset,
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

TEMP_STATE_PATH = (
    EXPERIMENT_DIR
    / "checkpoints"
    / "combined_save_reload_test.pt"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def main() -> None:
    set_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "没有检测到可用的CUDA显卡"
        )

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
            / "val.txt"
        ),
        processor=processor,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    batch = next(iter(loader))

    pixel_values = batch[
        "pixel_values"
    ].to(
        device,
        non_blocking=True,
    )

    print("创建第一个三模块联合模型……")

    model_1 = build_combined_model(
        model_name_or_path=MODEL_NAME,
        num_labels=2,
        local_files_only=True,
        state_dict_path=None,
    )

    validate_combined_structure(model_1)

    model_1.to(device)
    model_1.eval()

    logits_1 = model_1(
        pixel_values=pixel_values
    ).logits.detach().cpu()

    TEMP_STATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model_1.state_dict(),
        TEMP_STATE_PATH,
    )

    print(
        "联合模型状态字典已保存："
        f"{TEMP_STATE_PATH}"
    )

    print(
        "重新创建联合模型并严格加载状态字典……"
    )

    model_2 = build_combined_model(
        model_name_or_path=MODEL_NAME,
        num_labels=2,
        local_files_only=True,
        state_dict_path=TEMP_STATE_PATH,
    )

    validate_combined_structure(model_2)

    model_2.to(device)
    model_2.eval()

    logits_2 = model_2(
        pixel_values=pixel_values
    ).logits.detach().cpu()

    maximum_output_difference = (
        logits_1 - logits_2
    ).abs().max().item()

    if maximum_output_difference > 1e-5:
        raise RuntimeError(
            "保存前后输出不一致，最大差值："
            f"{maximum_output_difference}"
        )

    state_1 = model_1.state_dict()
    state_2 = model_2.state_dict()

    if state_1.keys() != state_2.keys():
        raise RuntimeError(
            "保存前后状态字典键不一致"
        )

    maximum_parameter_difference = 0.0

    for key in state_1:
        parameter_difference = (
            state_1[key].detach().cpu()
            - state_2[key].detach().cpu()
        ).abs().max().item()

        maximum_parameter_difference = max(
            maximum_parameter_difference,
            parameter_difference,
        )

    if maximum_parameter_difference > 0.0:
        raise RuntimeError(
            "严格加载后参数存在差异，最大差值："
            f"{maximum_parameter_difference}"
        )

    tpsa_parameter_count = sum(
        parameter.numel()
        for name, parameter
        in model_2.named_parameters()
        if ".tpsa." in name
    )

    ceca_parameter_count = sum(
        parameter.numel()
        for name, parameter
        in model_2.named_parameters()
        if ".ceca." in name
    )

    total_parameter_count = sum(
        parameter.numel()
        for parameter
        in model_2.parameters()
    )

    print("=" * 68)
    print("三模块联合模型保存—加载测试通过")
    print(
        f"状态字典路径："
        f"{TEMP_STATE_PATH}"
    )
    print(
        f"输出形状："
        f"{tuple(logits_2.shape)}"
    )
    print(
        "保存前后最大输出差值："
        f"{maximum_output_difference:.10f}"
    )
    print(
        "保存前后最大参数差值："
        f"{maximum_parameter_difference:.10f}"
    )
    print(
        f"TPSA参数量："
        f"{tpsa_parameter_count}"
    )
    print(
        f"CECA参数量："
        f"{ceca_parameter_count}"
    )
    print(
        f"模型总参数量："
        f"{total_parameter_count}"
    )
    print("TPSA结构检查：成功")
    print("CECA结构检查：成功")
    print("严格加载：成功")
    print("=" * 68)


if __name__ == "__main__":
    main()

import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerImageProcessor

from tpsa_model import build_tpsa_model
from train_baseline import (
    PROJECT_DIR,
    WinterJujubeDataset,
    calculate_metrics,
    update_confusion_matrix,
)


MODEL_NAME = (
    "nvidia/"
    "segformer-b2-finetuned-ade-512-512"
)

EXPERIMENT_DIR = (
    PROJECT_DIR
    / "07_experiments"
    / "segformer_b2_tpsa_effbs10"
)

CHECKPOINT_DIR = (
    EXPERIMENT_DIR
    / "checkpoints"
)

MODEL_STATE_PATH = (
    CHECKPOINT_DIR
    / "formal_training_best_state.pt"
)

PROCESSOR_DIR = (
    CHECKPOINT_DIR
    / "processor"
)

BEST_METRICS_PATH = (
    CHECKPOINT_DIR
    / "best_validation_metrics.json"
)

RESULT_DIR = (
    EXPERIMENT_DIR
    / "test_results"
)

JSON_RESULT_PATH = (
    RESULT_DIR
    / "tpsa_test_metrics.json"
)

TEXT_RESULT_PATH = (
    RESULT_DIR
    / "tpsa_test_metrics.txt"
)


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "没有检测到可用的CUDA显卡"
        )

    if not MODEL_STATE_PATH.exists():
        raise FileNotFoundError(
            "找不到TPSA最佳模型状态字典："
            f"{MODEL_STATE_PATH}"
        )

    if not PROCESSOR_DIR.exists():
        raise FileNotFoundError(
            "找不到图像处理器目录："
            f"{PROCESSOR_DIR}"
        )

    device = torch.device("cuda")
    batch_size = 4

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation_metrics = {}

    if BEST_METRICS_PATH.exists():
        best_validation_metrics = json.loads(
            BEST_METRICS_PATH.read_text(
                encoding="utf-8"
            )
        )

    print("正在加载最佳TPSA模型……")
    print(f"状态字典：{MODEL_STATE_PATH}")

    processor = (
        SegformerImageProcessor
        .from_pretrained(
            PROCESSOR_DIR,
            local_files_only=True,
        )
    )

    test_dataset = WinterJujubeDataset(
        split_file=(
            PROJECT_DIR
            / "03_数据集划分"
            / "test.txt"
        ),
        processor=processor,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model = build_tpsa_model(
        model_name_or_path=MODEL_NAME,
        num_labels=2,
        local_files_only=True,
        state_dict_path=MODEL_STATE_PATH,
    )

    model.to(device)
    model.eval()

    confusion_matrix = torch.zeros(
        (2, 2),
        dtype=torch.int64,
    )

    total_loss = 0.0
    total_samples = 0

    evaluation_start = time.time()

    print("=" * 65)
    print("开始评估锁定后的TPSA模型")
    print(f"测试集数量：{len(test_dataset)}")
    print(f"测试批量大小：{batch_size}")
    print(f"运行设备：{device}")
    print("=" * 65)

    progress = tqdm(
        test_loader,
        desc="测试",
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
            enabled=True,
        ):
            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

            loss = outputs.loss

        resized_logits = F.interpolate(
            outputs.logits.float(),
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        predictions = resized_logits.argmax(
            dim=1
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

        total_loss += (
            loss.item()
            * current_batch_size
        )

    test_loss = (
        total_loss / total_samples
    )

    metrics = calculate_metrics(
        confusion_matrix
    )

    evaluation_seconds = (
        time.time() - evaluation_start
    )

    result = {
        "experiment": (
            "segformer_b2_tpsa_effbs10"
        ),
        "model": "SegFormer-B2 + TPSA",
        "ceca_enabled": False,
        "tpsa_enabled": True,
        "lz_enabled": False,
        "training_effective_batch_size": 10,
        "test_batch_size": batch_size,
        "test_images": len(test_dataset),
        "best_validation_epoch": (
            best_validation_metrics.get(
                "epoch"
            )
        ),
        "best_validation_branch_iou": (
            best_validation_metrics.get(
                "branch_iou"
            )
        ),
        "best_validation_stage_scales": (
            best_validation_metrics.get(
                "stage_scales"
            )
        ),
        "test_loss": float(test_loss),
        "background_iou": float(
            metrics["background_iou"]
        ),
        "branch_iou": float(
            metrics["branch_iou"]
        ),
        "miou": float(
            metrics["miou"]
        ),
        "mpa": float(
            metrics["mpa"]
        ),
        "branch_precision": float(
            metrics["branch_precision"]
        ),
        "branch_recall": float(
            metrics["branch_recall"]
        ),
        "branch_f1": float(
            metrics["branch_f1"]
        ),
        "branch_dice": float(
            metrics["branch_dice"]
        ),
        "confusion_matrix": (
            confusion_matrix.tolist()
        ),
        "evaluation_seconds": float(
            evaluation_seconds
        ),
    }

    JSON_RESULT_PATH.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    matrix = confusion_matrix.tolist()

    text_result = f"""SegFormer-B2 + TPSA 测试结果

最佳验证轮次：{result["best_validation_epoch"]}
最佳验证集枝条IoU：{result["best_validation_branch_iou"]}

测试集数量：{result["test_images"]}
测试批量大小：{result["test_batch_size"]}
测试损失：{result["test_loss"]:.6f}

背景IoU：{result["background_iou"]:.4f}
枝条IoU：{result["branch_iou"]:.4f}
mIoU：{result["miou"]:.4f}
mPA：{result["mpa"]:.4f}
枝条Precision：{result["branch_precision"]:.4f}
枝条Recall：{result["branch_recall"]:.4f}
枝条F1：{result["branch_f1"]:.4f}
枝条Dice：{result["branch_dice"]:.4f}

混淆矩阵：
[[{matrix[0][0]}, {matrix[0][1]}],
 [{matrix[1][0]}, {matrix[1][1]}]]

评估用时：{evaluation_seconds:.2f}秒
"""

    TEXT_RESULT_PATH.write_text(
        text_result,
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("TPSA模型测试结果")
    print("=" * 65)
    print(f"测试损失：{test_loss:.6f}")

    print(
        "背景IoU："
        f"{metrics['background_iou']:.4f}"
    )

    print(
        "枝条IoU："
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
        "枝条F1："
        f"{metrics['branch_f1']:.4f}"
    )

    print(
        "枝条Dice："
        f"{metrics['branch_dice']:.4f}"
    )

    print("混淆矩阵：")
    print(confusion_matrix.numpy())

    print(
        f"评估用时："
        f"{evaluation_seconds:.2f}秒"
    )

    print("=" * 65)
    print(f"结果已保存至：{RESULT_DIR}")


if __name__ == "__main__":
    main()

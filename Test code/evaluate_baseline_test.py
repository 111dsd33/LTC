import json
import time
from pathlib import Path

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
    / "segformer_b2_baseline"
)

MODEL_DIR = (
    EXPERIMENT_DIR
    / "checkpoints"
    / "continue_to_200_best"
)

TEST_SPLIT = (
    PROJECT_DIR
    / "03_数据集划分"
    / "test.txt"
)

OUTPUT_DIR = (
    EXPERIMENT_DIR
    / "test_results"
)

BATCH_SIZE = 4
NUM_WORKERS = 0


@torch.inference_mode()
def evaluate(
    model,
    dataloader,
    device,
):
    model.eval()

    confusion_matrix = torch.zeros(
        (2, 2),
        dtype=torch.int64,
    )

    total_loss = 0.0
    total_samples = 0

    progress = tqdm(
        dataloader,
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
            enabled=device.type == "cuda",
        ):
            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

        logits = F.interpolate(
            outputs.logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        prediction = logits.argmax(dim=1)

        update_confusion_matrix(
            confusion_matrix,
            prediction,
            labels,
        )

        batch_size = pixel_values.size(0)

        total_loss += (
            outputs.loss.item()
            * batch_size
        )

        total_samples += batch_size

    average_loss = (
        total_loss / total_samples
    )

    metrics = calculate_metrics(
        confusion_matrix
    )

    return (
        average_loss,
        metrics,
        confusion_matrix,
    )


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"找不到最佳基线模型：{MODEL_DIR}"
        )

    if not TEST_SPLIT.exists():
        raise FileNotFoundError(
            f"找不到测试集名单：{TEST_SPLIT}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "没有检测到可用的CUDA显卡"
        )

    device = torch.device("cuda")

    print("正在加载最佳基线模型……")
    print("模型目录：", MODEL_DIR)

    processor = (
        SegformerImageProcessor
        .from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
    )

    model = (
        SegformerForSemanticSegmentation
        .from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
    )

    model.to(device)

    test_dataset = WinterJujubeDataset(
        split_file=TEST_SPLIT,
        processor=processor,
    )

    if len(test_dataset) != 93:
        raise ValueError(
            f"测试集应为93张，"
            f"实际为{len(test_dataset)}张"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print("=" * 60)
    print("开始评估锁定后的基线模型")
    print(f"测试集数量：{len(test_dataset)}")
    print(f"批量大小：{BATCH_SIZE}")
    print(f"运行设备：{device}")
    print("=" * 60)

    start_time = time.time()

    test_loss, metrics, matrix = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    elapsed_seconds = (
        time.time() - start_time
    )

    results = {
        "model": "SegFormer-B2 baseline",
        "model_directory": str(MODEL_DIR),
        "test_image_count": len(test_dataset),
        "test_loss": test_loss,
        "background_iou": (
            metrics["background_iou"]
        ),
        "branch_iou": (
            metrics["branch_iou"]
        ),
        "miou": metrics["miou"],
        "background_pa": (
            metrics["background_pa"]
        ),
        "branch_pa": (
            metrics["branch_pa"]
        ),
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
        "confusion_matrix": (
            matrix.tolist()
        ),
        "evaluation_seconds": (
            elapsed_seconds
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        OUTPUT_DIR
        / "baseline_test_metrics.json"
    )

    json_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text_lines = [
        "SegFormer-B2 Baseline Test Results",
        "=" * 50,
        f"Test images: {len(test_dataset)}",
        f"Test loss: {test_loss:.6f}",
        (
            "Background IoU: "
            f"{metrics['background_iou']:.4f}"
        ),
        (
            "Branch IoU: "
            f"{metrics['branch_iou']:.4f}"
        ),
        f"mIoU: {metrics['miou']:.4f}",
        (
            "Background PA: "
            f"{metrics['background_pa']:.4f}"
        ),
        (
            "Branch PA: "
            f"{metrics['branch_pa']:.4f}"
        ),
        f"mPA: {metrics['mpa']:.4f}",
        (
            "Branch Precision: "
            f"{metrics['branch_precision']:.4f}"
        ),
        (
            "Branch Recall: "
            f"{metrics['branch_recall']:.4f}"
        ),
        (
            "Branch F1: "
            f"{metrics['branch_f1']:.4f}"
        ),
        (
            "Branch Dice: "
            f"{metrics['branch_dice']:.4f}"
        ),
        "",
        "Confusion Matrix:",
        str(matrix.numpy()),
        "",
        (
            "Evaluation time: "
            f"{elapsed_seconds:.2f} seconds"
        ),
    ]

    text_path = (
        OUTPUT_DIR
        / "baseline_test_metrics.txt"
    )

    text_path.write_text(
        "\n".join(text_lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("基线模型测试结果")
    print("=" * 60)
    print(f"测试损失：{test_loss:.6f}")
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
    print("混淆矩阵：")
    print(matrix.numpy())
    print(
        f"评估用时："
        f"{elapsed_seconds:.2f}秒"
    )
    print("=" * 60)
    print(f"结果已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

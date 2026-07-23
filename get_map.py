import csv
import os
import tempfile
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_map
from detector import Detector

# ================================
# User Config (edit here only)
# ================================
USE_KFOLD = True  #True False

SINGLE_WEIGHT_PATH = "logs/fold_1/best_epoch_weights.pth"
SINGLE_VAL_ANNOTATION_PATH = "val_fold1.txt"

classes_path = "model_data/voc_classes.txt"
PROJECT_NAME = "RaDet_ablation_full"

KFOLD_WEIGHT_PATHS = [
    "logs/fold_1/ep193.pth",
    "logs/fold_2/ep210.pth",
    "logs/fold_3/ep193.pth",
    "logs/fold_4/ep196.pth",
    "logs/fold_5/ep240.pth",
]
KFOLD_VAL_ANNOTATION_PATHS = [
    "val_fold1.txt",
    "val_fold2.txt",
    "val_fold3.txt",
    "val_fold4.txt",
    "val_fold5.txt",
]

# get_map confidence is independent from predict.py
map_confidence = 0.3
nms_iou = 0.3
letterbox_image = False
use_cuda = True

MINOVERLAP = 0.5
score_threhold = 0.4
draw_plot = True

MAP_SUMMARY_MD_NAME = f"{PROJECT_NAME}_our_results_summary_0.5.md"
MAP_SUMMARY_CSV_NAME = f"{PROJECT_NAME}_our_results_summary_0.5.csv"
summary_md_path = MAP_SUMMARY_MD_NAME
summary_csv_path = MAP_SUMMARY_CSV_NAME

# Optional: override any Detector defaults (architecture, input size, etc.)
detector_extra_kwargs = {"ablation_mode": "full"}


def ensure_dir(path):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def prepare_map_dirs(base_dir):
    """Prepare the folder layout required by the mAP calculator."""
    ensure_dir(base_dir)
    ensure_dir(os.path.join(base_dir, "ground-truth"))
    ensure_dir(os.path.join(base_dir, "detection-results"))


def load_annotation_entries(annotation_txt, class_names):
    """Load image paths and ground-truth boxes from an annotation txt file."""
    if not os.path.exists(annotation_txt):
        raise FileNotFoundError(f"Annotation txt not found: {annotation_txt}")

    entries = []
    with open(annotation_txt, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            image_path = parts[0]
            if not os.path.exists(image_path):
                print(f"[Warn] image missing in annotation list: {image_path}")
                continue

            image_id = f"{idx:06d}_{os.path.splitext(os.path.basename(image_path))[0]}"
            gt_boxes = []
            for box in parts[1:]:
                fields = box.split(",")
                if len(fields) < 5:
                    continue
                left = int(float(fields[0]))
                top = int(float(fields[1]))
                right = int(float(fields[2]))
                bottom = int(float(fields[3]))
                cls_id = int(float(fields[4]))
                if cls_id < 0 or cls_id >= len(class_names):
                    continue
                gt_boxes.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_names[cls_id],
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                    }
                )

            entries.append({"image_id": image_id, "image_path": image_path, "gt_boxes": gt_boxes})
    return entries


def append_summary(records, title):
    """Append evaluation records to the CSV and Markdown summary files."""
    if not records:
        return

    header = ["timestamp", "script", "task", "item", "value", "note"]
    file_exists = os.path.exists(summary_csv_path)
    with open(summary_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k, "") for k in header})

    with open(summary_md_path, "a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n")
        f.write("| timestamp | script | task | item | value | note |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in records:
            f.write(
                f"| {row.get('timestamp','')} | {row.get('script','')} | {row.get('task','')} | "
                f"{row.get('item','')} | {row.get('value','')} | {row.get('note','')} |\n"
            )


def build_detector(weight_path):
    """Create a Detector instance for one weight file."""
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight not found: {weight_path}")
    kwargs = dict(
        model_path=weight_path,
        classes_path=classes_path,
        confidence=map_confidence,
        nms_iou=nms_iou,
        letterbox_image=letterbox_image,
        cuda=use_cuda,
    )
    kwargs.update(detector_extra_kwargs)
    return Detector(**kwargs)


def write_gt_txt(gt_path, gt_boxes):
    """Write one mAP ground-truth text file."""
    with open(gt_path, "w", encoding="utf-8") as f:
        for box in gt_boxes:
            f.write(
                "%s %s %s %s %s\n"
                % (box["class_name"], box["left"], box["top"], box["right"], box["bottom"])
            )


def write_det_txt(det_path, detections, class_names):
    """Write one mAP detection-result text file."""
    valid_names = set(class_names)
    with open(det_path, "w", encoding="utf-8") as f:
        for det in detections:
            if det["class_name"] not in valid_names:
                continue
            score = str(det["score"])
            f.write(
                "%s %s %s %s %s %s\n"
                % (
                    det["class_name"],
                    score[:6],
                    det["left"],
                    det["top"],
                    det["right"],
                    det["bottom"],
                )
            )


def run_map_calc(out_dir, class_names):
    """Run mAP calculation and return aggregate metrics."""
    map_value, ap_dict, class_metrics = get_map(
        MINOVERLAP,
        draw_plot=draw_plot,
        score_threhold=score_threhold,
        path=out_dir,
        return_class_metrics=True,
    )
    return float(map_value), ap_dict, class_metrics


def get_fold_jobs():
    """Return all k-fold jobs as (fold_id, weight_path, annotation_path)."""
    if len(KFOLD_WEIGHT_PATHS) != len(KFOLD_VAL_ANNOTATION_PATHS):
        raise ValueError("KFOLD_WEIGHT_PATHS and KFOLD_VAL_ANNOTATION_PATHS must have the same length.")

    return [
        (idx, weight, ann_path)
        for idx, (weight, ann_path) in enumerate(zip(KFOLD_WEIGHT_PATHS, KFOLD_VAL_ANNOTATION_PATHS), start=1)
    ]


def evaluate_fold_model(model_path, entries, class_names, eval_tag):
    """Evaluate one weight file against the loaded validation entries."""
    with tempfile.TemporaryDirectory(prefix=f"{eval_tag}_map_") as out_dir:
        prepare_map_dirs(out_dir)
        model = build_detector(model_path)

        for entry in tqdm(entries, desc=f"Eval {eval_tag}"):
            image_id = entry["image_id"]
            image_path = entry["image_path"]
            gt_boxes = entry["gt_boxes"]

            try:
                image = Image.open(image_path)
            except Exception as e:
                print(f"[Skip] {image_path}: {e}")
                continue

            _, detections = model.predict_boxes(image, class_names=class_names)
            write_det_txt(
                os.path.join(out_dir, "detection-results", f"{image_id}.txt"),
                detections,
                class_names,
            )
            write_gt_txt(os.path.join(out_dir, "ground-truth", f"{image_id}.txt"), gt_boxes)

        metric, ap_dict, class_metrics = run_map_calc(out_dir, class_names)
        del model
        return metric, ap_dict, class_metrics


def append_class_metrics_records(records, timestamp, task, prefix, class_metrics):
    """Append per-class F1, recall, and precision rows to summary records."""
    for cls, metrics in class_metrics.items():
        records.append(
            {
                "timestamp": timestamp,
                "script": "get_map.py",
                "task": task,
                "item": f"{prefix}_{cls}_F1",
                "value": f"{metrics.get('f1', 0.0)*100:.2f}%",
                "note": f"raw={metrics.get('f1', 0.0):.6f}",
            }
        )
        records.append(
            {
                "timestamp": timestamp,
                "script": "get_map.py",
                "task": task,
                "item": f"{prefix}_{cls}_Recall",
                "value": f"{metrics.get('recall', 0.0)*100:.2f}%",
                "note": f"raw={metrics.get('recall', 0.0):.6f}",
            }
        )
        records.append(
            {
                "timestamp": timestamp,
                "script": "get_map.py",
                "task": task,
                "item": f"{prefix}_{cls}_Precision",
                "value": f"{metrics.get('precision', 0.0)*100:.2f}%",
                "note": f"raw={metrics.get('precision', 0.0):.6f}",
            }
        )


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    class_names, _ = get_classes(classes_path)
    records = []
    records.extend(
        [
            {
                "timestamp": timestamp,
                "script": "get_map.py",
                "task": "map_config",
                "item": "USE_KFOLD",
                "value": str(USE_KFOLD),
                "note": "True=all folds; False=single model",
            },
            {
                "timestamp": timestamp,
                "script": "get_map.py",
                "task": "map_config",
                "item": "SINGLE_WEIGHT_PATH",
                "value": SINGLE_WEIGHT_PATH,
                "note": "used only when USE_KFOLD=False",
            },
            {
                "timestamp": timestamp,
                "script": "get_map.py",
                "task": "map_config",
                "item": "SINGLE_VAL_ANNOTATION_PATH",
                "value": SINGLE_VAL_ANNOTATION_PATH,
                "note": "used only when USE_KFOLD=False",
            },
            {
                "timestamp": timestamp,
                "script": "get_map.py",
                "task": "map_config",
                "item": "MINOVERLAP",
                "value": str(MINOVERLAP),
                "note": "",
            },
            {
                "timestamp": timestamp,
                "script": "get_map.py",
                "task": "map_config",
                "item": "score_threhold",
                "value": str(score_threhold),
                "note": "",
            },
            {
                "timestamp": timestamp,
                "script": "get_map.py",
                "task": "map_config",
                "item": "map_confidence",
                "value": str(map_confidence),
                "note": "",
            },
            {
                "timestamp": timestamp,
                "script": "get_map.py",
                "task": "map_config",
                "item": "nms_iou",
                "value": str(nms_iou),
                "note": "",
            },
        ]
    )

    if USE_KFOLD:
        fold_jobs = get_fold_jobs()
        fold_metrics = []
        fold_ap_by_class = {cls: [] for cls in class_names}
        fold_f1_by_class = {cls: [] for cls in class_names}
        fold_recall_by_class = {cls: [] for cls in class_names}
        fold_precision_by_class = {cls: [] for cls in class_names}
        for idx, weight, ann_path in fold_jobs:
            entries = load_annotation_entries(ann_path, class_names)
            metric, ap_dict, class_metrics = evaluate_fold_model(
                model_path=weight,
                entries=entries,
                class_names=class_names,
                eval_tag=f"fold_{idx}",
            )
            fold_metrics.append(metric)
            records.append(
                {
                    "timestamp": timestamp,
                    "script": "get_map.py",
                    "task": "map_kfold",
                    "item": f"fold_{idx}_metric",
                    "value": f"{metric:.6f}",
                    "note": f"weight={weight}; val={ann_path}",
                }
            )
            for cls, ap in ap_dict.items():
                records.append(
                    {
                        "timestamp": timestamp,
                        "script": "get_map.py",
                        "task": "map_kfold",
                        "item": f"fold_{idx}_{cls}_AP",
                        "value": f"{ap:.6f}",
                        "note": "",
                    }
                )
            for cls in class_names:
                fold_ap_by_class[cls].append(float(ap_dict.get(cls, 0.0)))
                metrics = class_metrics.get(cls, {})
                fold_f1_by_class[cls].append(float(metrics.get("f1", 0.0)))
                fold_recall_by_class[cls].append(float(metrics.get("recall", 0.0)))
                fold_precision_by_class[cls].append(float(metrics.get("precision", 0.0)))
            append_class_metrics_records(
                records=records,
                timestamp=timestamp,
                task="map_kfold",
                prefix=f"fold_{idx}",
                class_metrics=class_metrics,
            )

        records.append(
            {
                "timestamp": timestamp,
                "script": "get_map.py",
                "task": "map_kfold",
                "item": "fold_mean_metric",
                "value": f"{float(np.mean(fold_metrics)):.6f}",
                "note": f"std={float(np.std(fold_metrics)):.6f}",
            }
        )
        for cls in class_names:
            ap_values = np.array(fold_ap_by_class.get(cls, []), dtype=np.float64)
            f1_values = np.array(fold_f1_by_class.get(cls, []), dtype=np.float64)
            recall_values = np.array(fold_recall_by_class.get(cls, []), dtype=np.float64)
            precision_values = np.array(fold_precision_by_class.get(cls, []), dtype=np.float64)
            if ap_values.size > 0:
                records.append(
                    {
                        "timestamp": timestamp,
                        "script": "get_map.py",
                        "task": "map_kfold",
                        "item": f"fold_mean_{cls}_AP",
                        "value": f"{float(np.mean(ap_values)):.6f}",
                        "note": f"std={float(np.std(ap_values)):.6f}",
                    }
                )
                records.append(
                    {
                        "timestamp": timestamp,
                        "script": "get_map.py",
                        "task": "map_kfold",
                        "item": f"fold_mean_{cls}_F1",
                        "value": f"{float(np.mean(f1_values)):.6f}",
                        "note": f"std={float(np.std(f1_values)):.6f}",
                    }
                )
                records.append(
                    {
                        "timestamp": timestamp,
                        "script": "get_map.py",
                        "task": "map_kfold",
                        "item": f"fold_mean_{cls}_Recall",
                        "value": f"{float(np.mean(recall_values)):.6f}",
                        "note": f"std={float(np.std(recall_values)):.6f}",
                    }
                )
                records.append(
                    {
                        "timestamp": timestamp,
                        "script": "get_map.py",
                        "task": "map_kfold",
                        "item": f"fold_mean_{cls}_Precision",
                        "value": f"{float(np.mean(precision_values)):.6f}",
                        "note": f"std={float(np.std(precision_values)):.6f}",
                    }
                )
    else:
        entries = load_annotation_entries(SINGLE_VAL_ANNOTATION_PATH, class_names)
        metric, ap_dict, class_metrics = evaluate_fold_model(
            model_path=SINGLE_WEIGHT_PATH,
            entries=entries,
            class_names=class_names,
            eval_tag="single",
        )
        records.append(
            {
                "timestamp": timestamp,
                "script": "get_map.py",
                "task": "map_single",
                "item": "single_metric",
                "value": f"{metric:.6f}",
                "note": f"weight={SINGLE_WEIGHT_PATH}; val={SINGLE_VAL_ANNOTATION_PATH}",
            }
        )
        for cls, ap in ap_dict.items():
            records.append(
                {
                    "timestamp": timestamp,
                    "script": "get_map.py",
                    "task": "map_single",
                    "item": f"single_{cls}_AP",
                    "value": f"{ap:.6f}",
                    "note": "",
                }
            )
        append_class_metrics_records(
            records=records,
            timestamp=timestamp,
            task="map_single",
            prefix="single",
            class_metrics=class_metrics,
        )

    append_summary(records, f"Map Eval {timestamp}")
    print(f"Done. Summary appended to: {summary_md_path}, {summary_csv_path}")


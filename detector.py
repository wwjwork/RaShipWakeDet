import colorsys
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.detector import RaShipWakeDet
from utils.utils import (
    cvtColor,
    get_anchors,
    get_classes,
    preprocess_input,
    resize_image,
    show_config,
)
from utils.utils_bbox import DecodeBox


class Detector(object):
    """Inference helper for the custom detector."""

    _defaults = {
        # Paths
        "model_path": "logs/loss_2025_03_21_23_14_27/ep125-loss0.042-val_loss0.049.pth",
        "classes_path": "model_data/voc_classes.txt",
        "anchors_path": "model_data/detector_anchors.txt",

        # Model decoding config
        "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        "input_shape": [640, 640],

        # RaShipWakeDet config
        "rashipwakedet_variant": "s2",
        "ablation_mode": "full",
        "neck_channels": (128, 256, 512),

        # Post-process config
        "confidence": 0.5,
        "nms_iou": 0.3,
        "letterbox_image": False,

        # Runtime
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        """Return a named default configuration value."""
        if n in cls._defaults:
            return cls._defaults[n]
        return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        """Create decoder state, colors, and the RaShipWakeDet model instance."""
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(
            self.anchors,
            self.num_classes,
            (self.input_shape[0], self.input_shape[1]),
            self.anchors_mask,
        )

        hsv_tuples = [(x / self.num_classes, 1.0, 1.0) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors)
        )

        self.generate()
        show_config(**self._defaults)

    def generate(self):
        """Build model and load weights."""
        self.net = RaShipWakeDet(
            self.anchors_mask,
            self.num_classes,
            variant=self.rashipwakedet_variant,
            input_shape=self.input_shape,
            pretrained=False,
            neck_channels=self.neck_channels,
            ablation_mode=self.ablation_mode,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        self.net = self.net.eval()
        print(f"{self.model_path} model, and classes loaded.")

    def _inference(self, image):
        """Run one forward pass and decode boxes."""
        image = cvtColor(image)
        image_shape = np.array(np.shape(image)[0:2])

        image_data = resize_image(
            image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image
        )
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)), 0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(
                torch.cat(outputs, 1),
                self.num_classes,
                self.input_shape,
                image_shape,
                self.letterbox_image,
                conf_thres=self.confidence,
                nms_thres=self.nms_iou,
            )

            if results[0] is None:
                return (
                    image,
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32),
                )

            top_label = np.array(results[0][:, 6], dtype="int32")
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
            return image, top_boxes, top_conf, top_label

    def _build_detections(self, image, top_boxes, top_conf, top_label, class_names=None):
        """Convert decoded tensors into a list of detection dicts."""
        allowed = set(class_names) if class_names is not None else None
        w, h = image.size
        detections = []

        for i, c in enumerate(top_label):
            cls_id = int(c)
            cls_name = self.class_names[cls_id]
            if allowed is not None and cls_name not in allowed:
                continue

            top, left, bottom, right = top_boxes[i]
            top = max(0, int(np.floor(top)))
            left = max(0, int(np.floor(left)))
            bottom = min(h, int(np.floor(bottom)))
            right = min(w, int(np.floor(right)))

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "score": float(top_conf[i]),
                    "top": top,
                    "left": left,
                    "bottom": bottom,
                    "right": right,
                }
            )
        return detections

    def predict_boxes(self, image, class_names=None):
        """Return (RGB image, detections list)."""
        image, top_boxes, top_conf, top_label = self._inference(image)
        detections = self._build_detections(
            image, top_boxes, top_conf, top_label, class_names=class_names
        )
        return image, detections

    def detect_image(self, image, crop=False, count=False, detections=None):
        """Draw detection results on image.

        If `detections` is provided, it draws these detections directly.
        Otherwise it runs inference first.
        """
        if detections is None:
            image, top_boxes, top_conf, top_label = self._inference(image)
            detections = self._build_detections(image, top_boxes, top_conf, top_label)
        else:
            image = cvtColor(image)

        if len(detections) == 0:
            return image

        top_label = np.array([int(det["class_id"]) for det in detections], dtype="int32")
        top_boxes = np.array(
            [[det["top"], det["left"], det["bottom"], det["right"]] for det in detections],
            dtype=np.float32,
        )
        top_conf = np.array([float(det["score"]) for det in detections], dtype=np.float32)

        font = ImageFont.truetype(
            font="model_data/simhei.ttf",
            size=np.floor(3e-2 * image.size[1] + 0.5).astype("int32"),
        )
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if crop:
            for i, _ in enumerate(top_boxes):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype("int32"))
                left = max(0, np.floor(left).astype("int32"))
                bottom = min(image.size[1], np.floor(bottom).astype("int32"))
                right = min(image.size[0], np.floor(right).astype("int32"))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(
                    os.path.join(dir_save_path, "crop_" + str(i) + ".png"),
                    quality=95,
                    subsampling=0,
                )
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top).astype("int32"))
            left = max(0, np.floor(left).astype("int32"))
            bottom = min(image.size[1], np.floor(bottom).astype("int32"))
            right = min(image.size[0], np.floor(right).astype("int32"))

            label = f"{predicted_class} {score:.2f}"
            draw = ImageDraw.Draw(image)
            bbox = draw.textbbox((0, 0), label, font=font)
            label_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            encoded_label = label.encode("utf-8")
            print(encoded_label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for t in range(thickness):
                draw.rectangle([left + t, top + t, right - t, bottom - t], outline=self.colors[int(c)])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[int(c)])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        """Measure average single-image inference time."""
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(
            image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image
        )
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)), 0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            self.bbox_util.non_max_suppression(
                torch.cat(outputs, 1),
                self.num_classes,
                self.input_shape,
                image_shape,
                self.letterbox_image,
                conf_thres=self.confidence,
                nms_thres=self.nms_iou,
            )

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                self.bbox_util.non_max_suppression(
                    torch.cat(outputs, 1),
                    self.num_classes,
                    self.input_shape,
                    image_shape,
                    self.letterbox_image,
                    conf_thres=self.confidence,
                    nms_thres=self.nms_iou,
                )

        t2 = time.time()
        return (t2 - t1) / test_interval

    def get_heatmap_mask(self, image):
        """Return (RGB image, uint8 heatmap mask)."""

        def sigmoid(x):
            """Apply a NumPy sigmoid to heatmap logits."""
            return 1.0 / (1.0 + np.exp(-x))

        image = cvtColor(image)
        image_data = resize_image(
            image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image
        )
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)), 0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)

        mask = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, 3, -1, h, w]), [0, 3, 4, 1, 2])[0]
            score = np.max(sigmoid(sub_output[..., 4]), -1)
            score = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score = (score * 255).astype("uint8")
            mask = np.maximum(mask, normed_score)

        return image, mask.astype("uint8")

    def detect_heatmap(self, image, heatmap_save_path):
        """Save and display an overlay heatmap image."""
        import matplotlib.pyplot as plt

        image, mask = self.get_heatmap_mask(image)
        plt.imshow(image, alpha=1)
        plt.axis("off")
        plt.imshow(mask, alpha=0.5, interpolation="nearest", cmap="jet")
        plt.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches="tight", pad_inches=-0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        """Write detection result txt for mAP calculation."""
        det_path = os.path.join(map_out_path, "detection-results", image_id + ".txt")
        _, detections = self.predict_boxes(image, class_names=class_names)
        with open(det_path, "w", encoding="utf-8") as f:
            for det in detections:
                score = str(det["score"])
                f.write(
                    "%s %s %s %s %s %s\n"
                    % (
                        det["class_name"],
                        score[:6],
                        str(det["left"]),
                        str(det["top"]),
                        str(det["right"]),
                        str(det["bottom"]),
                    )
                )
        return



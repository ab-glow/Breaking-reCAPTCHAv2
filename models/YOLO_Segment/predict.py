from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

CLASSES = ["bicycle", "bridge", "bus", "car", "chimney", "crosswalk", "hydrant", "motorcycle", "other", "palm", "stairs", "traffic"]
"""{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
      6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
        12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
          19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
              31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                  41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
                    48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                      54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                          66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                            72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
                              77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
"""
MAPPING = [(1, 0), (5, 2), (2, 3), (10, 6), (3,7), (9, 12)] # mapping from COCO to captcha classes
# Load a pretrained YOLOv8n model
model = YOLO('yolov8n-seg.pt')
LAST_PLOT = None   # annotated image for the live preview

# (min detection confidence, min tile coverage), strict first, then more forgiving
PASSES = [(0.50, 0.08), (0.30, 0.05), (0.15, 0.02)]


def supports(class_number):
  return any(m[1] == class_number for m in MAPPING)


def _grid_from_masks(result, class_id_searched, conf_min, min_cover, w, h, grid_size=4):
  grid = np.zeros((grid_size, grid_size))
  if result.masks is None:
    return grid
  mask = np.zeros((h, w), dtype=np.uint8)
  for k, cls in enumerate(result.boxes.cls):
    if int(cls.item()) != class_id_searched:
      continue
    if float(result.boxes.conf[k].item()) < conf_min:
      continue
    for segment in result.masks[k].xy:
      segment = np.array(segment, dtype=np.int32)
      if len(segment) >= 3:
        cv2.fillPoly(mask, [segment], 1)
  cell_h, cell_w = h / grid_size, w / grid_size
  for i in range(grid_size):
    for j in range(grid_size):
      cell = mask[int(i * cell_h):int((i + 1) * cell_h), int(j * cell_w):int((j + 1) * cell_w)]
      if cell.size and float(cell.mean()) >= min_cover:
        grid[i, j] = 1
  return grid


def predict(class_number, image_path):
  global LAST_PLOT
  LAST_PLOT = None

  class_id_searched = None
  for m in MAPPING:
    if m[1] == class_number:
      class_id_searched = m[0]
      break
  if class_id_searched is None:
    print(f"4x4: class {class_number} has no COCO equivalent, nothing to detect")
    return False, np.zeros((4, 4))

  img = Image.open(image_path)
  w, h = img.size

  # run once at the lowest confidence, then filter for each pass
  results = model.predict(image_path, save=False, imgsz=320, conf=PASSES[-1][0], visualize=False)
  result = results[0]
  try:
    LAST_PLOT = result.plot()
  except Exception:
    LAST_PLOT = None

  for conf_min, min_cover in PASSES:
    grid = _grid_from_masks(result, class_id_searched, conf_min, min_cover, w, h)
    if grid.sum() > 0:
      print(f"4x4: pass conf>={conf_min}, cover>={min_cover} -> {int(grid.sum())} tiles")
      return True, grid

  print("4x4: nothing found even on the most forgiving pass")
  return False, np.zeros((4, 4))
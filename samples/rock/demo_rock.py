import cv2
import os
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import numpy as np
from skimage.measure import find_contours

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from samples.rock.rock import RockCocoDataset, RockTrainConfig


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, save_path,
                      scores=None, figsize=(16, 16)):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    fig, ax = plt.subplots(1, figsize=figsize)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    masked_image = image.astype(np.uint8).copy()
    for i in range(N):
        color = (1, 0, 0)

        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))
    fig.savefig(save_path, dpi=300)


if __name__ == "__main__":

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Demo images')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the ROCK-COCO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default="logs",
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    class InferenceConfig(RockTrainConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

    config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=args.logs)

    dataset_val = RockCocoDataset()
    val_type = "test"
    dataset_val.load_rock(args.dataset, val_type)
    dataset_val.prepare()

    model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)
    os.makedirs(osp.join(args.logs, "debug"), exist_ok=True)

    for image_id in dataset_val.image_ids:
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, config, 
                                  image_id, use_mini_mask=False)

        results = model.detect([original_image], verbose=1)
        r = results[0]

        file_name = osp.split(dataset_val.image_info[image_id]['path'])[-1]
        save_path = osp.splitext(osp.join(args.logs, "debug", file_name))[0] +\
                ".png"

        display_instances(original_image, r['rois'], 
            r['masks'], r['class_ids'], save_path, r['scores'])


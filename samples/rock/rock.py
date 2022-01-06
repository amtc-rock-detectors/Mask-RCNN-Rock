"""
Mask R-CNN
Configurations and data loading code for MS COCO.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco
    # Train a new model starting from ImageNet weights.
    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5
    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last
    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import json
import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
from skimage import measure

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from samples.rock.rock_coco_eval import RockCOCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class RockTrainConfig(Config):

    NAME = "rock_coco"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    BATCH_SIZE = 8

    STEPS_PER_EPOCH = 1006 // BATCH_SIZE

    IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + NUM_CLASSES

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 334 // BATCH_SIZE

    LEARNING_RATE = [1e-4, 5e-5, 1e-5]
    DECAY_STEPS = (np.array([100, 200]) * STEPS_PER_EPOCH).tolist()


############################################################
#  Dataset
############################################################

class RockCocoDataset(utils.Dataset):

    def load_rock(self, dataset_dir, subset, class_ids=None,
                  class_map=None, return_coco=False):

        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """

        coco = COCO("{}/annotations/{}.json".format(dataset_dir, subset))

        image_dir = "{}/{}".format(dataset_dir, subset)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds(catNms=["rock"]))

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("rock_coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "rock_coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = annotation['category_id']
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(RockCocoDataset, self).load_mask(image_id)


    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "rock_coco"),
                "bbox": np.int64([bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]).tolist(),
                "score": float(score),
                "segmentation": []
            }
                           
            for contour in measure.find_contours(mask, 0.5):
                result['segmentation'].append(np.int64(np.flip(contour, axis=1).ravel()).tolist())
                           
            results.append(result)
        
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None, areaRng=None, save_results=""):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()
    
    if len(save_results) > 0:
        with open(os.path.join(save_results, 'results.json'), 'w') as f:
            f.write("")

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    if len(save_results) > 0:
        with open(os.path.join(save_results, 'results.json'), 'w') as f:
            f.write(json.dumps(results))

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    assert eval_type == "bbox" or eval_type == "segm"
    eval_type = {"use_gt_bbox": eval_type == "bbox", "use_gt_poly": eval_type == "segm"}
    cocoEval = RockCOCOeval(coco, coco_results, areaRng, **eval_type, areaType="segmentation")
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on ROCK COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the ROCK-COCO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--min_area', type=float, default=0,
                        help='minimum ellipse area to be considered as detection')
    parser.add_argument('--max_area', type=float, default=1e10,
                        help='maximum ellipse area to be considered as detection')
    parser.add_argument('--small_min', type=float, default=0,
                        help='Minimum area for small ellipse')
    parser.add_argument('--small_max', type=float, default=-1,
                        help='Maximum area for small ellipse')
    parser.add_argument('--medium_min', type=float, default=-1,
                        help='Minimum area for medium ellipse')
    parser.add_argument('--medium_max', type=float, default=-1,
                        help='Minimum area for medium ellipse')
    parser.add_argument('--large_min', type=float, default=-1,
                        help='Minimum area for medium ellipse')
    parser.add_argument('--large_max', type=float, default=1e10,
                        help='Minimum area for medium ellipse')
    parser.add_argument('--eval_type', type=str, default='segm',
                        help='bbox | segm')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = RockTrainConfig()
    else:
        class InferenceConfig(RockTrainConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
        areaRng = {'small': [args.small_min, args.small_max], 
                   'medium': [args.medium_min, args.medium_max,], 
                   'large': [args.large_min, args.large_max],
                   'all': [args.small_min, args.large_max]}

        if np.any([np.any(np.array(areaRng[key]) < 0) for key in areaRng.keys()]):
            areaRng = None    

    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    if not args.model == "coco":
        model.load_weights(model_path, by_name=True)
    else:
        model.load_weights(model_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = RockCocoDataset()
        dataset_train.load_rock(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = RockCocoDataset()
        val_type = "val"
        dataset_val.load_rock(args.dataset, val_type)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Sequential([
                        imgaug.augmenters.Fliplr(0.5),
                        imgaug.augmenters.Affine(
                            scale=(0.6, 1.4), # scale images to 80-120% of their size, individually per axis
                            translate_percent=(-0.2, 0.2), # translate by -20 to +20 percent (per axis)
                            order=1, # use nearest neighbour or bilinear interpolation (fast)
                            cval=0)
                        ])
                        

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    decay_steps=config.DECAY_STEPS,
                    epochs=500,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Fine tune all layers
        # print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=(np.array(config.LEARNING_RATE) / 50).tolist(),
                    decay_steps=(np.array(config.DECAY_STEPS) + 600 * config.STEPS_PER_EPOCH).tolist(),
                    epochs=1200,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate" or args.command == "test":
        # Validation dataset
        dataset_val = RockCocoDataset()
        val_type = "val" if args.command == "evaluate" else "test"
        coco = dataset_val.load_rock(args.dataset, val_type, return_coco=True)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, args.eval_type, limit=int(args.limit),
                      image_ids=None, areaRng=areaRng, save_results=os.path.split(model_path)[0])

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate' or 'test'".format(args.command))

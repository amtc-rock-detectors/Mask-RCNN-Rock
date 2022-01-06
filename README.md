# Mask R-CNN-Rock for Rock Detection and Segmentation

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone. It builds on Matterport's implementation of [Mask-RCNN](https://github.com/matterport/Mask_RCNN). Further implementation details on Mask R-CNN can be found in the original repository.

Results of Rock Detection and Segmentation were compared with the ones of using ellipses and bounding boxes [Rocky-CenterNet](https://github.com/amtc-rock-detectors/Rocky-CenterNet). Results can also be seen in our paper "Detecting Rocks in Challenging Mining Environments using Convolutional Neural Networks and Ellipses as an alternative to Bounding Boxes" to be published in the "Expert Systems with Applications" Journal.


# Training on Rock Dataset
We're providing pre-trained weights for our Rock's Dataset to make it easier to start [link](https://drive.google.com/file/d/1NpnGvb4O98Ta92C_sppWIko-p27yoF9I/view?usp=sharing). You can use those weights as a starting point to train your own variation on the network. Training and evaluation code is in `samples/rock/rock.py`. You can run it directly from the command line. We found best results starting the rtraining process with coco weights, as such:

```
# Train a new model starting from pre-trained COCO weights
python3 samples/rock/rock.py train --dataset=/path/to/rock_dataset/ --model=coco

# Continue training a model that you had trained earlier
python3 samples/coco/coco.py train --dataset=/path/to/rock_dataset/ --model=/path/to/weights.h5
```

You can also run the Rock Dataset evaluation code with:
```
# Run COCO evaluation on the trained model (val set)
python3 samples/rock/rock.py evaluate --dataset=/path/to/rock_dataset/ --model=/path/to/weights.h5

# Run COCO evaluation on the trained model (test set)
python3 samples/rock/rock.py test --dataset=/path/to/rock_dataset/ --model=/path/to/weights.h5
```

The training schedule, learning rate, and other parameters should be set in `samples/rock/rock.py`.

# Main Results
## Rock Segmentation on Hammer Rock Test Set


|  Metric      |MRCNN-Rock vs - polygon|MRCNN-Rock vs - bbox|
|--------------|-----------------------|--------------------|
|AP (all)      |         71.4          |        72.1        |
|AP (small)    |         64.9          |        67.4        |
|AP (medium)   |         73.7          |        74.6        |
|AP (large)    |         73.5          |        72.2        |
|AR (all)      |         77.0          |        77.7        |
|AR (small)    |         72.6          |        75.4        |
|AR (medium)   |         77.8          |        79.4        |
|AR (large)    |         78.3          |        76.9        |

## Rock Segmentation on Scaled Front Dataset


|  Metric      |MRCNN-Rock vs - ellipse|MRCNN-Rock vs - bbox|
|--------------|-----------------------|--------------------|
|AP (all)      |         53.7          |        59.1        |
|AP (small)    |         51.0          |        57.2        |
|AP (medium)   |         64.0          |        65.9        |
|AP (large)    |         83.4          |        76.7        |
|AR (all)      |         59.8          |        65.7        |
|AR (small)    |         58.2          |        64.9        |
|AR (medium)   |         67.0          |        69.4        |
|AR (large)    |         83.3          |        76.7        |


## Citation
If you use the Rock-Dataset or the results of our work, please cite:
```
    @article{loncomilla2022detecting,
      title={Detecting Rocks in Challenging Mining Environments using Convolutional Neural Networks and Ellipses as an alternative to Bounding Boxes},
      author={Loncomilla, Patricio and Samtani, Pavan and Ruiz-del-Solar, Javier},
      journal={Expert Systems with Applications},
      publisher={Elsevier},
      year={2022}
    }
```

## Requirements
Python 3.6, TensorFlow 1.14, Keras 2.1.5 and other common packages listed in `requirements.txt`.

### Rock Dataset Requirements:
To train or test on the Rock Dataset:
* pycocotools (installation instructions below)
* [Rocks Dataset](https://datos.uchile.cl/dataset.xhtml?persistentId=doi:10.34691/FK2/1GQBHK)

If you use Docker, the Dockerfile attached in this repository has been tested.

## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ```
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. (Optional) To train or test on the Rock's Dataset install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

## Repository forked from: Mask R-CNN for Object Detection and Segmentation
Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow:          
> Waleed Abdulla       
> Github (https://github.com/matterport/Mask_RCNN)

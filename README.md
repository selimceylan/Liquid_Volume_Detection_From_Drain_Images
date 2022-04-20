# Liquid_Volume_Detection_From_Drain_Images
Liquid volume detection code with Mask R-CNN on Keras and Tensorflow. This project uses the mask pixels of objects for compute liquid volumes. Thanks to Mask R-CNN, masks of objects can be obtained easily.
## Mask R-CNN for Object Detection and Segmentation
The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.\
![image](https://user-images.githubusercontent.com/86148100/164163216-988168b9-d975-491d-a881-584348dc7d47.png)

The architecture of Mask R-CNN:
![image](https://user-images.githubusercontent.com/86148100/164163474-29b237ce-b4ae-4453-9afd-8b86a0bb336e.png)

Using Mask R-CNN pixel-wise masks for every object in an image can automatically segment and construct.\
Unlike the Fast/Faster R-CNN
## Why to Use Mask R-CNN
## Citation
Use this bibtex to cite this repository:
> @misc{matterport_maskrcnn_2017,\
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},\
  author={Waleed Abdulla},\
  year={2017},\
  publisher={Github},\
  journal={GitHub repository},\
  howpublished={\url{https://github.com/matterport/Mask_RCNN }},\
  }

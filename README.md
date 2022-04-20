# Liquid_Volume_Detection_From_Drain_Images
Liquid volume detection code with Mask R-CNN on Keras and Tensorflow. This project uses the mask pixels of objects for compute liquid volumes. Thanks to Mask R-CNN, masks of objects can be obtained easily.
## Mask R-CNN for Object Detection and Segmentation
The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.\
![image](https://user-images.githubusercontent.com/86148100/164163216-988168b9-d975-491d-a881-584348dc7d47.png)

The architecture of Mask R-CNN:
![image](https://user-images.githubusercontent.com/86148100/164163474-29b237ce-b4ae-4453-9afd-8b86a0bb336e.png)

Using Mask R-CNN pixel-wise masks for every object in an image can automatically segment and construct.\
Mask R-CNN was built using Faster R-CNN. While Faster R-CNN has 2 outputs for each candidate object, a class label and a bounding-box offset, Mask R-CNN is the addition of a third branch that outputs the object mask.
## How to Detect Liquid Volume in Drain 
Our model detect blood part and empty part separately in drain and gives masks of these detections. The software takes total volume of drain as an input for calculate this:\
blood_mask_pixels + empty_mask_pixels = total_pixels_of_drain\
blood_volume_in_ml = (blood_mask_pixels * total_volume_in_ml) / total_pixels_of_drain
## Restrictions 
When these conditions provided, model detect volume with %1 error rate.
- Drain position should be on the middle of image.
- Drain and camera distance should be roughly arm distance.
- User should know the total volume of drain.

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

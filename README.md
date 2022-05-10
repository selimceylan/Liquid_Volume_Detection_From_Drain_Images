# Liquid_Volume_Detection_From_Drain_Images
Liquid volume detection code with Mask R-CNN on Keras and Tensorflow. This project uses the mask pixels of objects for compute liquid volumes. Thanks to Mask R-CNN, masks of objects can be obtained easily.

## Mask R-CNN for Object Detection and Segmentation
The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.\
![image](https://user-images.githubusercontent.com/86148100/164163216-988168b9-d975-491d-a881-584348dc7d47.png)

The architecture of Mask R-CNN:
![image](https://user-images.githubusercontent.com/86148100/164163474-29b237ce-b4ae-4453-9afd-8b86a0bb336e.png)

Using Mask R-CNN, pixel-wise masks for every object in an image can automatically segment and construct.\
Mask R-CNN was built using Faster R-CNN. While Faster R-CNN has 2 outputs for each candidate object, a class label and a bounding-box offset, Mask R-CNN is the addition of a third branch that outputs the object mask.

## How to Detect Liquid Volume in Drain 
Our model detect blood part and empty part separately in drain and gives masks of these detections. Total pixel sizes of these masks saved and sum of these saved as total_pixels_of_drain. The software takes total volume of drain ("total_volume_in_ml") as an input for calculate the blood volume in milliliter.\
blood_mask_pixels + empty_mask_pixels = total_pixels_of_drain\
blood_volume_in_ml = (blood_mask_pixels * total_volume_in_ml) / total_pixels_of_drain

## Restrictions 
When these conditions provided, model measures volume with %3 error rate.
- Drain position should be on the middle of image.
- Drain and camera distance should be roughly arm distance.
- User should know the total volume of drain.

## Advantages
- Model can measure liquid for different intensities of blood.
- Model can work independent from kind of drain, only total volume needed.
- Background of the drain images doesn't change the result. 

## Usage with network
Streamlit api code (streamlit_app.py) added to use the model on web. Model path must adjusted in code, image path will be taken from user. Last result displays on web and volume exists at top right side of image.
![main](https://user-images.githubusercontent.com/86148100/167581118-1c27de41-f0a3-46de-85f7-4c2651956881.png)

Result page with good result.


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

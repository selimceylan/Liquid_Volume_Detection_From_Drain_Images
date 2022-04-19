import random
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import cv2
from matplotlib import patches
from skimage.measure import find_contours
import colorsys
from matplotlib.patches import Polygon
import mrcnn.model as modellib
import Blood


def Inference(CUSTOM_MODEL_PATH,MODEL_DIR,IMAGE_DIR,total_volume):
    # K.clear_session()

    def apply_mask(image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    def random_colors(N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    class InferenceConfig(Blood.DrainConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()


    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(CUSTOM_MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')

    # #Update the class names in the order mentioned in the custom.py file
    class_names = ['BG', 'Blood', 'Empty']

    # Load a random image from the images folder
    #file_names = next(os.walk(IMAGE_DIR))[2]
    #image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    #image = skimage.io.imread(os.path.join(IMAGE_DIR, "71.jpeg"))
    image = skimage.io.imread(IMAGE_DIR)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                            class_names, r['scores'])

    positive = [1,2]

    for i in range(r['masks'].shape[-1]):
        mask = r['masks'][:, :, i]
        image[mask] = 255
        image[~mask] = 0
        unique, counts = np.unique(image, return_counts=True)
        mask_area = counts[1] / (counts[0] + counts[1])
        positive[i] = counts[1]
        # print(counts[1])

    mask = r['masks']
    mask = mask.astype(int)
    #print(mask.shape)
    #positive = np.array()




    for i in range(mask.shape[2]):
        temp = skimage.io.imread(IMAGE_DIR)
        for j in range(temp.shape[2]):
            temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
        plt.figure(figsize=(8,8))
        # plt.imshow(temp)
        # plt.show()

    h,w = temp.shape[1:3]
    empty = positive[0] // (w*h)
    blood = positive[1] // (w*h)
    #print((w*h))

    #print(positive[0])
    #print(positive[1])

    #print(blood,"Blood")
    #print(empty,"Empty")

    total = blood + empty
    totalInMl = total_volume #200 for drain 1

    # totalInMl = 550  #for drain2

    bloodInMl = (blood*totalInMl)//total
    print("Blood ml is:",bloodInMl)


    boxes = r['rois']
    masks = r['masks']
    class_ids = r['class_ids']
    class_names = class_names
    scores = r['scores']

    title=""
    figsize=(16, 16)
    ax=None
    show_mask=True
    show_bbox=True
    colors=None
    captions=None

    image = skimage.io.imread(IMAGE_DIR)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
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
    org = (150, 150)
    color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 3

    masked_image = cv2.putText(masked_image.astype(np.uint8),str(bloodInMl)+' ml',org,font,1,color,thickness)

    masked_image=masked_image.astype(np.uint8)
    return masked_image




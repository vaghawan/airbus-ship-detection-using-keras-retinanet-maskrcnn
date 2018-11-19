import keras
import sys
sys.path.append("../..")
# import keras_retinanet
from keras_maskrcnn import models
from keras_maskrcnn.utils.visualization import draw_mask, draw_mask_overlap
from keras_retinanet.utils.visualization import draw_box, draw_caption, draw_annotations
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
import skimage
from skimage.morphology import binary_opening, disk, label,binary_closing,binary_dilation

from skimage.measure import find_contours
from skimage.measure import label as label_lib
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import dilation, erosion
import scipy
import skimage.color
import skimage.io
import pandas as pd
import csv, datetime

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())



def draw_test(image, boxes, masks, scores, labels=None, color=None, binarize_threshold=0.5):
    """ Draws a mask in a given box.

    Args
        image              : Three dimensional image to draw on.
        box                : Vector of at least 4 values (x1, y1, x2, y2) representing a box in the image.
        mask               : A 2D float mask which will be reshaped to the size of the box, binarized and drawn over the image.
        color              : Color to draw the mask with. If the box has 5 values, the last value is assumed to be the label and used to construct a default color.
        binarize_threshold : Threshold used for binarizing the mask.
        scores             : A 1D Numpy array of scores
    """
    resulting_masks = []
    kept_scores = []
    kept_labels = []
    kept_boxes = []
    
    if labels is None:
        labels = [None for _ in range(boxes.shape[0])]
    

    for box, mask, label, score in zip(boxes, masks, labels, scores):
        
        # resize to fit the box
        if label != -1.:
            kept_boxes.append(box)
            box = box.astype(int)
            mask = cv2.resize(mask, (box[2] - box[0], box[3] - box[1]))

            # binarize the mask
            mask = (mask > binarize_threshold).astype(np.uint8)
            # print(mask, mask.shape)
            # draw the mask in the image
            mask_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
            mask_image[box[1]:box[3], box[0]:box[2]] = mask
            mask = mask_image
            resulting_masks.append(mask)
            kept_scores.append(score)
            kept_labels.append(label)
            #resulting_masks = np.append(resulting_masks, mask, axis=0)
    if len(resulting_masks) <1 :
        resulting_masks = np.zeros((image.shape[0], image.shape[1], 0), np.uint8)
        kept_scores = np.array([])
        kept_labels = np.array([])
        kept_boxes = np.array([])
    else:   
        resulting_masks = np.asarray(resulting_masks)
        resulting_masks = resulting_masks.reshape(resulting_masks.shape[0],768, 768, 1 )
        kept_scores = np.asarray(kept_scores)
        kept_labels = np.asarray(kept_labels)
        kept_boxes = np.asarray(kept_boxes)
        
    print(resulting_masks.shape)
    if resulting_masks.shape[-1] == None:
        print("here")
        resulting_masks = resulting_masks.reshape(0, 768, 768, 1)
    
    return resulting_masks, kept_scores, kept_labels, kept_boxes




def postprocess_masks(result, image, min_nuc_size=0):

    """Clean overlaps between bounding boxes, fill small holes, smooth boundaries"""
    print(np.where(result["masks"][0]==1))
    height, width = image.shape[:2]

    # If there is no mask prediction do the following
    print("inside post-process", result['masks'].shape)
    if result['masks'].shape[0] == 0:
        print("we were supposed to be here")
        result['masks'] = np.zeros([height, width, 1])
        result['masks'][0, 0, 0] = 1
        result['scores'] = np.ones(1)
        result['class_ids'] = np.zeros(1)

    keep_ind = np.where(np.sum(result['masks'], axis=(0, 1)) > min_nuc_size)[0]
    
    if len(keep_ind) < result['masks'].shape[-1]:
        # print('Deleting',len(result['masks'])-len(keep_ind), ' empty result['masks']')
        result['masks'] = result['masks'][..., keep_ind]
        result['scores'] = result['scores'][keep_ind]
        result['rois'] = result['rois'][keep_ind]
        result['class_ids'] = result['class_ids'][keep_ind]

    sort_ind = np.argsort(result['scores'])[::-1]
    
    result['masks'] = result['masks'][..., sort_ind]
    overlap = np.zeros([height, width])

    # Removes overlaps from masks with lower score
    for mm in range(result['masks'].shape[-1]):
        # Fill holes inside the mask
        mask = binary_fill_holes(result['masks'][..., mm]).astype(np.uint8)
        # Smoothen edges using dilation and erosion
        mask = erosion(dilation(mask))
        # Delete overlaps
        overlap += mask
        
        mask[overlap > 1] = 0
        
        out_label = label_lib(mask)
        
        # Remove all the pieces if there are more than one pieces
        if out_label.max() > 1:
            mask[()] = 0
            print('removed something here')
        result['masks'][..., mm] = mask
    
    keep_ind = np.where(np.sum(result['masks'], axis=(0, 1)) > min_nuc_size)[0]
    
    if len(keep_ind) < result['masks'].shape[-1]:
        result['masks'] = result['masks'][..., keep_ind]
        result['scores'] = result['scores'][keep_ind]
        result['rois'] = result['rois'][keep_ind]
        result['class_ids'] = result['class_ids'][keep_ind]

    return result



def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)



def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    # y1 = np.maximum(box[0], boxes[:, 0])
    # y2 = np.minimum(box[2], boxes[:, 2])
    # x1 = np.maximum(box[1], boxes[:, 1])
    # x2 = np.minimum(box[3], boxes[:, 3])

    #New for us: 
    y1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[3], boxes[:, 3])
    x1 = np.maximum(box[0], boxes[:, 0])
    x2 = np.minimum(box[2], boxes[:, 2])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    # y1 = boxes[:, 0]
    # x1 = boxes[:, 1]
    # y2 = boxes[:, 2]
    # x2 = boxes[:, 3]
    
    #New for us:
    y1 = boxes[:, 1]
    x1 = boxes[:, 0]
    y2 = boxes[:, 3]
    x2 = boxes[:, 2]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    red = image*[1,1,0]
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set

    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        mask = mask.astype(int)
        #print(rle_encode(mask))
        splash = np.where(mask, red, image).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash 


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b

    return ' '.join(str(x) for x in run_lengths)


def prob_to_rles(masks, height, width):

    if masks.sum() < 1:
        masks = np.zeros([height, width, 1])
        # print('no masks')
        masks[0, 0, 0] = 1

    if np.any(masks.sum(axis=-1) > 1):
        print('Overlap', masks.shape)

    for mm in range(masks.shape[-1]):
        yield rle_encoding(masks[..., mm].astype(np.int32)), np.sum(masks[..., mm].astype(np.int32)==1)
        
def test_on_single_image(model, imagepath, labels_names:dict, SCORE_THRES= 0.2, IOU_THRES = 0.5):
	
    image = read_image_bgr(imagepath)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    boxes  = outputs[-4][0]
    scores = outputs[-3][0]
    labels = outputs[-2][0]
    masks  = outputs[-1][0]

    # correct for image scale
    boxes /= scale

    # visualize detections
    #print(boxes.shape, scores.shape, masks.shape)
    masks, scores, labels, boxes = draw_test(draw, boxes, masks, scores, labels, color=label_color(0))
    if boxes.size !=0:
        keep_ind = non_max_suppression(boxes, scores, IOU_THRES)
        masks = masks[keep_ind, :, :]
        scores = scores[keep_ind]
        labels = labels[keep_ind]
        rois = boxes[keep_ind]
        result = {"masks":masks, "scores":scores, "class_ids":labels, "rois":rois}
        idxtokeep = np.where(result['scores']>SCORE_THRES)[0]
        result['masks'] = masks[idxtokeep,:, :]
        result['scores'] = scores[idxtokeep]
        result['class_ids'] = labels[idxtokeep]
        result['rois'] = rois[idxtokeep]

        image_arr = skimage.io.imread(imagepath)

        masks_resulted = []
        if result['masks'].size !=0:
            firstmask = result['masks'][0]

            result['masks'] = result['masks'][1:]
            for box, score, label, mask in zip(result['rois'], result['scores'], result['class_ids'], result['masks']):
                color = label_color(label)
                b = box.astype(int)
                #draw_box(draw, b, color=color)

                firstmask = np.append(firstmask, mask, axis=2)
                mask = mask[:,:,label]
                #draw_mask_overlap(draw, b, mask)

                caption = "{} {:.3f}".format(labels_to_names[label], score)

            
            

            print("concatinated mask", firstmask.shape)
            result['masks'] = firstmask
            print(result['scores'])
            splash = color_splash(image_arr, result['masks'])
            skimage.io.imshow(splash)
            plt.show()
            plt.close()

        else:
            print("the result were removed due to the thresholding.")

    else:
        print("no instance found.")



def generate_result(model, imagedir, labels_names:dict, csv_path:str, output_image_path:str=None, SCORE_THRES= 0.2, IOU_THRES = 0.5):
    
    already_tested = list(pd.read_csv(csv_path)["ImageId"].unique())
    allimagesindir = list(os.listdir(imagedir))
    yettotest = list(set(allimagesindir) - set(already_tested)) if (len(allimagesindir)>len(already_tested)) else list(set(already_tested) - set(allimagesindir))
    print(yettotest) 
    print("number of images yet to  test is:", len(yettotest), len(already_tested))

    count = 0
    for image_path in yettotest:
        img_path = imagedir+image_path
        # load image
        image = read_image_bgr(img_path)
        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)
        # process image
        start = time.time()
        outputs = model.predict_on_batch(np.expand_dims(image, axis=0))

        print("processing time: ", time.time() - start)
        

        boxes  = outputs[-4][0]
        scores = outputs[-3][0]
        labels = outputs[-2][0]
        masks  = outputs[-1][0]

        boxes /= scale

        #This is for the preparation of submission csv file.
        imagelist=[]
        encodelist=[]
        overlappedrm = []
        scorelist = []
        lengthlist = []
        
        masks, scores, labels, boxes = draw_test(draw, boxes, masks, scores, labels, color=label_color(0))
        if boxes.size !=0:
            keep_ind = non_max_suppression(boxes, scores, IOU_THRES)
            masks = masks[keep_ind, :, :]
            scores = scores[keep_ind]
            labels = labels[keep_ind]
            rois = boxes[keep_ind]
            result = {"masks":masks, "scores":scores, "class_ids":labels, "rois":rois}
            idxtokeep = np.where(result['scores']>SCORE_THRES)[0]
            result['masks'] = masks[idxtokeep,:, :]
            result['scores'] = scores[idxtokeep]
            result['class_ids'] = labels[idxtokeep]
            result['rois'] = rois[idxtokeep]

            image_arr = skimage.io.imread(img_path)

            masks_resulted = []

            if result['masks'].size != 0:
                firstmask = result['masks'][0]

                result['masks'] = result['masks'][1:]
                for box, score, label, mask in zip(result['rois'], result['scores'], result['class_ids'], result['masks']):
                    color = label_color(label)
                    b = box.astype(int)
                    firstmask = np.append(firstmask, mask, axis=2)
                    mask = mask[:,:,label]

                
                result['masks'] = firstmask
                print("kept scores.. " , result["scores"])
                if output_image_path:
                    splash = color_splash(image_arr, result['masks'])
                    file_name = image_path+"splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
                    skimage.io.imsave(output_image_path+file_name, splash)
                    
                

            else:
                print("the result were removed due to the score thresholding.")

            height, width = image_arr.shape[:2]

            if result["scores"].size == 0:
                imagelist.append(image_path)
                encodelist.append('')
                print("the mask is blank")
                scorelist.append(0.0)
                lengthlist.append(0)

            else:
                masks = result["masks"].astype(int)
                
                encode = list(prob_to_rles(masks, height, width))
                
                
                scores = list(result['scores'])
                
                if encode !=None:
                    for en, score in zip(encode, scores):
                        imagelist.append(image_path)
                        encodelist.append(en[0])
                        scorelist.append(score)
                        lengthlist.append(en[1])
                        #overlappedrm.append(rmoverlapped)
                else:
                    imagelist.append(image_path)
                    encodelist.append('')
                    scorelist.append(0.0)
                    lengthlist.append(0)

        else:
            imagelist.append(image_path)
            encodelist.append('')
            print("the mask is blank")
            scorelist.append(0.0)
            lengthlist.append(0)

        with open(csv_path, 'a') as outcsv:

            fieldnames = ['ImageId', 'EncodedPixels', 'Score', 'Length']
            writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
            writer.writerows([{"ImageId":img, "EncodedPixels":enc, "Score":scr, "Length":lengt} for img, enc, scr, lengt in zip(imagelist, encodelist, scorelist, lengthlist)])



if __name__=='__main__':
    # adjust this to point to your downloaded/trained model
    #model_path = os.path.join('..', 'snapshots', 'resnet50_csv_44.h5')
    model_path = '/var/www/keras-maskrcnn/snapshots/resnet50_csv_63.h5'
    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')
    print(model.summary())
    SCORE_THRES = 0.2
    IOU_THRES = 0.5
    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'ship'}

    img_path = '/var/www/keras-maskrcnn/examples/tet/5dcc2a406.jpg'
    # load image
    test_on_single_image(model, img_path, labels_to_names)

    #generate_result(model, "/var/www/mask-rcnn/data/all/test_v2/", labels_to_names, "/var/www/mask-rcnn/data/all/ensemble170-32.csv", output_image_path="/var/www/airbus-competition-using-keras-maskrcnn/examples/splash/")

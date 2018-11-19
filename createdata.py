import numpy as np
import pandas as pd
from PIL import Image
import os

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)





def load_mask(data):

    """Generate instance masks for an image.
   Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    # If not a ship dataset image, delegate to parent class.
    
    height =  width =  768
    rle = [data["EncodedPixels"]]
    mask = np.zeros((height,width,len(rle)))
    for p,m in enumerate(rle):
        all_masks = np.zeros((height,width))
        all_masks += rle_decode(m)
        mask[:, :, p] = all_masks

    return mask.astype(np.bool)


def rleToMask(rleString,height,width):
    rows,cols = height,width
    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1,2)
    img = np.zeros(rows*cols,dtype=np.uint8)
    for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(cols,rows)
    img = img.T
    return img


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


if __name__=='__main__':
    from tqdm import tqdm_notebook
    #Load Images
    masks = pd.read_csv('data/train_ship_segmentations_v2.csv')
    print(masks.shape[0], 'masks found')
    print(masks['ImageId'].value_counts().shape[0])
    masks.head()
    images_with_ship = masks.ImageId[masks.EncodedPixels.isnull()==False]
    images_with_ship = np.unique(images_with_ship.values)
    print('There are ' +str(len(images_with_ship)) + ' image files with masks')
    data = pd.DataFrame(columns=["ImageId", "x1", "y1", "x2", "y2", "classname", "mask"])
    masks.fillna('', inplace=True)
    grouped = masks.groupby(["ImageId"])
    count = 0
    for index, row in tqdm_notebook(masks.iterrows()):
        if(row["EncodedPixels"]==''):
            rowtoappend = {"ImageId":row["ImageId"], "x1":'', "y1":'', "x2":'', "y2":'', "classname":'', 'mask':''}
            #rowtoappend = [row["ImageId"].values.tolist()[0], '', '', '','', '']
            data.loc[count] = rowtoappend
            count+=1
        else:
            #print(row["EncodedPixels"], type(row["EncodedPixels"]))
            number_of_masks = masks[masks["ImageId"]==row["ImageId"]]
            rle = [row["EncodedPixels"]]
            
            imgdata = rleToMask(row["EncodedPixels"], 768, 768)
            im = Image.fromarray(imgdata)
            i = 0
            while os.path.exists("data/masks/"+row["ImageId"]+"%s.png" % i):
                i += 1
            im.save("data/masks/"+row["ImageId"]+"%s.png" % i)
            m = load_mask(row)
            box = extract_bboxes(m)
            box = box[0]
            
            rowtoappend = {"ImageId":row["ImageId"], "x1":box[1], "y1":box[0], "x2":box[3], 
                           "y2":box[2], "classname":"ship", 'mask':"data/masks/"+row["ImageId"]+"%s.png" % i}
            data.loc[count] = rowtoappend
            count+=1
    data.to_csv("data/annotation.csv", index=None)

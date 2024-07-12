import torch
from PIL import Image
from skimage import color
from skimage.transform import resize
import glob
import numpy as np
import csv
import cv2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

img_size = 28 # Image size
path_main = str('./Data/')
img_show1 = False; img_show2 = False

def Image_converter(image):
    image = color.rgb2gray(image)*255
    image = resize(image, (img_size, img_size), anti_aliasing=True)
    image_resized = torch.from_numpy(image).to(device)
    return image_resized

data_name = path_main + 'rfi_annotations_28by28.csv'
f = open(data_name, 'w', encoding='UTF8', newline='')
writer = csv.writer(f)

for classes in range(0,133):
    try: # Load only the image that have annotations
        path = path_main + str(classes) + str('/*.jpg')
        image_path = glob.glob(path, recursive=True)
        all_data = [] # For each class
        for img in image_path:
            image = Image.open(img)

            '''
            # Just rotate the images
            if image_ori.size[0]<image_ori.size[1]: # larger images vertically
                image = image_ori
            else: # larger images horizontally
                image = Image.fromarray(cv2.rotate(asarray(image_ori), cv2.ROTATE_90_CLOCKWISE), "RGB")
            image = asarray(image)
            filename = path_main + str(classes) + img[40:]
            # To save the rotated images
            cv2.imwrite(filename, image) '''


            '''
            # Complete white border
            background  = Image.new(mode = "RGB", size = (img_size, img_size), color = (255, 255, 255))
            if image_ori.size[0]<image_ori.size[1]: # larger images vertically
                background.paste(image_ori, (int((img_size/2)-(image_ori.size[0]/2)), int((img_size/2)-(image_ori.size[1]/2))))
                if img_show1 == True:
                    background = asarray(background)
                    cv2.imshow("Img", background)
                    cv2.waitKey(100)

            else: # larger images horizontally
                background.paste(image_ori, (int((img_size/2)-(image_ori.size[0]/2)), int((img_size/2)-(image_ori.size[1]/2))))
                # background = background.rotate(-90)
                if img_show2 == True:
                    background = asarray(background)
                    cv2.imshow("Img", background)
                    cv2.waitKey(100)
            image = background
            '''

            '''
            # White border on the sides
            if image_ori.size[0]!=image_ori.size[1]:
                max_coord = max(image_ori.size)
                background  = Image.new(mode = "RGB", size = (max_coord, max_coord), color = (255, 255, 255))
                if image_ori.size[0]<image_ori.size[1]: # larger images vertically
                    background.paste(image_ori, (int(max(image_ori.size)/2-min(image_ori.size)/2), 0))
                    if img_show1 == True:
                        background = asarray(background)
                        cv2.imshow("Img", background)
                        cv2.waitKey(100)

                else: # larger images horizontally
                    background.paste(image_ori, (0, int(max(image_ori.size)/2-min(image_ori.size)/2)))
                    background = background.rotate(-90)
                    if img_show2 == True:
                        background = asarray(background)
                        cv2.imshow("Img", background)
                        cv2.waitKey(100)
                image = background
            else:
                image = image_ori
            '''

            image_conv = Image_converter(image).flatten()
            image_class = torch.tensor(classes).reshape(1)
            data = (np.array(torch.cat([image_class, image_conv]))).astype(int)
            writer.writerow(data)
            all_data.append(data)
    except OSError as error:
        print(error) # When there is no folder

f.close()

if (img_show1 or img_show2) == True:
    cv2.destroyAllWindows()

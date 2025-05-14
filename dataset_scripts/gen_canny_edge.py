import cv2
import threading
from diffusers.utils import load_image
from PIL import Image
import glob
import numpy as np
import os

thread_count = 8
low_threshold = 100
high_threshold = 200

def cvt_canny_edge(cvt_list,dest):
    for item in cvt_list:
        canny_image = np.array(load_image(item))
        canny_image = cv2.Canny(canny_image, low_threshold, high_threshold)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        canny_image = Image.fromarray(canny_image)
        new_fldr = item.split("/")[-2]
        new_pth = new_fldr + "/" + item.split("/")[-1]
        os.makedirs(dest+new_fldr,exist_ok=True)
        canny_image.save(dest+new_pth,"PNG")

if __name__ == "__main__":
    parent_loc = "/work/zura-storage/Data/DrivingSceneDDM/images/"
    full_list = glob.glob(parent_loc+"**/*.png")
    each_thread_ct = int(len(full_list)/thread_count)
    # print(full_list)
    th_list = []
    for i in range(thread_count):
        end_ct = max(len(full_list),(i+1)*each_thread_ct) if i == thread_count-1 else (i+1)*each_thread_ct
        th = threading.Thread(target=cvt_canny_edge,args=(full_list[i*each_thread_ct:end_ct],"/work/zura-storage/Data/DrivingSceneDDM/canny/",))
        th_list.append(th)
        th_list[-1].start()
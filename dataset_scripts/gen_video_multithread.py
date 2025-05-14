import threading
import os
import cv2
import glob

def create_vid_fldr(fldr,imgs_total,out_loc):
    gta_imgs = list(os.listdir(fldr))
    gta_imgs.sort()
    resolution_og = cv2.imread(os.path.join(fldr,gta_imgs[0])).shape
    resolution = (resolution_og[1],resolution_og[0])
    for i in range(len(gta_imgs) - imgs_total):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_name = "{:09d}.mp4".format(i)
        the_name = os.path.join(out_loc,vid_name)
        out = cv2.VideoWriter(the_name, fourcc, seq_fps, resolution)
        for j in range(imgs_total):
            frame = cv2.imread(os.path.join(fldr,gta_imgs[i+j]))
            out.write(frame)
        out.release()

def create_vid_list(img_list,out_loc,vid_name,resolution):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    the_name = os.path.join(out_loc,vid_name)
    out = cv2.VideoWriter(the_name, fourcc, seq_fps, resolution)
    for img in img_list:
        frame = cv2.imread(img)
        out.write(frame)
    out.release()

dataset_loc = "/work/zura-storage/Data/DSDDM_XL/"
seq_len = 15 # in seconds
seq_fps = 10 # frames per sec
# resolution = (640,360)
thread_count = 8
threads_handle = []

imgs_total = seq_fps*seq_fps
# full_list = glob.glob(dataset_loc+"**/*.png")
# each_thread_ct = int(len(full_list)/thread_count)
# th_list = []

for dataset in os.listdir(os.path.join(dataset_loc,"images")):
    if dataset == "tfkt":
        continue
    full_pth = os.path.join(dataset_loc,"images",dataset)
    out_pth = os.path.join(dataset_loc,"videos",dataset)
    if not os.path.exists(out_pth):
        os.makedirs(out_pth)
    threads_handle.append(threading.Thread(target=create_vid_fldr,args=(full_pth,imgs_total,out_pth,)))
    threads_handle[-1].start()
    print(full_pth)

for th in threads_handle:
    th.join()

os.makedirs("./completed")
    
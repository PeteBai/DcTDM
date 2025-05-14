import threading
import os
import cv2
import glob

def create_vid_fldr(fldr,imgs_total,out_loc):
    gta_imgs = list(os.listdir(fldr))
    gta_imgs.sort()
    resolution_og = cv2.imread(os.path.join(fldr,gta_imgs[0])).shape
    resolution = (resolution_og[1],resolution_og[0])
    assert len(gta_imgs) % 30 == 0
    for i in range(int(len(gta_imgs) / 30)):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_name = "{:09d}.mp4".format(i)
        the_name = os.path.join(out_loc,vid_name)
        out = cv2.VideoWriter(the_name, fourcc, seq_fps, resolution)
        for j in range(imgs_total):
            frame = cv2.imread(os.path.join(fldr,gta_imgs[i+j]))
            out.write(frame)
        out.release()

dataset_loc = "/mnt/data/dataset/new_images/"
seq_len = 15 # in seconds
d = 10 # frames per sec
# resolution = (640,360)
threads_handle = []

imgs_total = 30
for dataset in os.listdir(os.path.join(dataset_loc,"cityscape")):
    full_pth = os.path.join(dataset_loc,"cityscape",dataset)
    out_pth = os.path.join(dataset_loc,"videos",dataset)
    if not os.path.exists(out_pth):
        os.makedirs(out_pth)
    threads_handle.append(threading.Thread(target=create_vid_fldr,args=(full_pth,imgs_total,out_pth,)))
    threads_handle[-1].start()
    print(full_pth)

for th in threads_handle:
    th.join()
    
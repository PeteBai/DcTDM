import os

dataset_pth = "/mnt/data/dataset/new_images/depth"
for ct in os.listdir(dataset_pth):
    ct_pth = os.path.join(dataset_pth,ct)
    all_depth = list(os.listdir(ct_pth))
    all_depth.sort()
    for idx,depth in enumerate(all_depth):
        new_name = "{:06d}.npy".format(idx)
        new_path = os.path.join(ct_pth,new_name)
        os.rename(os.path.join(ct_pth,depth),os.path.join(ct_pth,new_name))

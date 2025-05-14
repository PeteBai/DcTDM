# generate train.txt as 
# video.mp4 depth_folder depth_start frame_count description
import os
# print(os.listdir(os.path.join(".","videos")))

do_short = False

desc_edict = {
    "carla_01":"A vehicle driving in-road,carla style.",
    "carla_02":"A vehicle driving in-road,carla style.",
    "carla_03":"A vehicle driving in-road,carla style.",
    "carla_04":"A vehicle driving in-road,carla style.",
    "carla_05":"A vehicle driving in-road,carla style.",
    "carla_06":"A vehicle driving in-road,carla style.",
    "carla_07":"A vehicle driving in-road,carla style.",
    "carla_08":"A vehicle driving in-road,carla style.",
    "gta_01":"A vehicle driving in-road, gta style.",
    "gta_02":"A vehicle driving in-road, gta style.",
    "gta_03":"A vehicle driving in-road, gta style.",
    "gta_04":"A vehicle driving in-road, gta style.",
    "gta_05":"A vehicle driving off-road, gta style.",
    "kitti_01":"A vehicle driving in-road, kitti style.",
    "kitti_02":"A vehicle driving in-road, kitti style.",
    "kitti_03":"A vehicle driving in-road, kitti style.",
    "kitti_04":"A vehicle driving in-road, kitti style.",
    "kitti_05":"A vehicle driving in-road, kitti style.",
    "kitti_06":"A vehicle driving in-road, kitti style.",
    "kitti_07":"A vehicle driving in-road, kitti style.",
    "kitti_08":"A vehicle driving in-road, kitti style.",
    "nus_01":"A vehicle driving in-road, nuscenes style.",
    "nus_02":"A vehicle driving in-road, nuscenes style.",
    "nus_03":"A vehicle driving in-road, nuscenes style.",
    "nus_04":"A vehicle driving in-road, nuscenes style.",
    "nus_05":"A vehicle driving in-road, nuscenes style.",
    "nus_06":"A vehicle driving in-road, nuscenes style.",
    "nus_07":"A vehicle driving in-road, nuscenes style.",
    "nus_08":"A vehicle driving in-road, nuscenes style.",
    "tfkt":"A vehicle driving in-road, saves style."
}

if not do_short:
    f = open("train_style.txt","w")
    for dataset in os.listdir(os.path.join(".","videos")):
        vids = list(os.listdir(os.path.join(".","videos",dataset)))
        vids.sort()
        depth_zero_ct = len(list(os.listdir(os.path.join("depth",dataset)))[0].split(".")[0])
        for vid in vids:
            depth_ct = int(vid.split(".")[0])
            new_depth_name = str(depth_ct).zfill(depth_zero_ct) + ".npy"
            the_line = "{} {} {} {} {}\n".format(os.path.join("videos",dataset,vid),os.path.join("depth",dataset),new_depth_name,15,desc_edict[dataset])
            f.write(the_line)
    for ct_ct in os.listdir(os.path.join(".","new_images","videos")):
        vids = list(os.listdir(os.path.join(".","new_images","videos",ct_ct)))
        vids.sort()
        for vid in vids:
            depth_ct = int(vid.split(".")[0])
            new_depth_name = str(depth_ct).zfill(6) + ".npy"
            desc = "A vehicle driving in-road, cityscape style."
            the_line = "{} {} {} {} {}\n".format(os.path.join("new_images","videos",ct_ct,vid),os.path.join("new_images","depth",ct_ct),new_depth_name,15,desc)
            f.write(the_line)  
else:
    f = open("train_short.txt","w")
    # test code here
    for ct_ct in os.listdir(os.path.join(".","new_images","videos")):
        vids = list(os.listdir(os.path.join(".","new_images","videos",ct_ct)))
        vids.sort()
        for vid in vids:
            depth_ct = int(vid.split(".")[0]) * 30
            new_depth_name = str(depth_ct).zfill(6) + ".npy"
            desc = "A vehicle driving in-road in a real-world city."
            the_line = "{} {} {} {} {}\n".format(os.path.join("new_images","videos",ct_ct,vid),os.path.join("new_images","depth",ct_ct),new_depth_name,15,desc)
            f.write(the_line)  
f.close()

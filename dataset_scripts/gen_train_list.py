# generate train.txt as 
# video.mp4 depth_folder depth_start frame_count description
import os
# print(os.listdir(os.path.join(".","videos")))

do_short = False
txt_file_loc = "."
dataset_loc = "/work/zura-storage/Data/DrivingSceneDDM/"

desc_edict = {
    "carla_01":"A vehicle driving in-road in a small town with a river and several bridges; rainly, cloudy town surroundings.",
    "carla_02":"A vehicle driving in-road in a small town with a mixture of residential and commercial buildings; sunny afternoon suburban surroundings.",
    "carla_03":"A vehicle driving in-road in a large urban with a roundabout and large junctions; sunny noon urban surroundings.",
    "carla_04":"A vehicle driving in-road on a highway embedded mountains and trees; foggy, early morning forest surroundings.",
    "carla_05":"A vehicle driving in-road in a squared grid town with cross junctions and a bridge with multiple lanes per direction; sunny urban surroundings.",
    "carla_06":"A vehicle driving in-road on long many lane highways with many highway entrances and exits; sunny early evening forest surroundings.",
    "carla_07":"A vehicle driving in-road in a rural environment with narrow roads, corn, barns and hardly any traffic lights; sunny late afternoon town surrounding.",
    "carla_08":"A vehicle driving in-road in a downtown urban environment with skyscrapers, residential buildings and an ocean promenade; sunny noon urban surroundings.",
    "gta_01":"A vehicle driving in-road in a busy metropolitan environment with skyscrapers; sunny noon city surroundings.",
    "gta_02":"A vehicle driving in-road in a busy city environment with mostly residential buildings; earily evening city surroundings.",
    "gta_03":"A vehicle driving in-road in a busy metropolitan environment with skyscrapers; sunny noon city surroundings.",
    "gta_04":"A vehicle driving in-road near a busy city airport with billboards, road signs, hotels and bridges; sunny noon urban surroundings.",
    "gta_05":"A vehicle driving off-road on a mountain with sparse trees and uneven terrians, sunny noon mountain surroundings.",
    "kitti_01":"A vehicle driving in-road in a real-world residential neighborhood, consists mainly with houses with garden and side parking cars; sunny noon urban surroundings.",
    "kitti_02":"A vehicle driving in-road in a real-world residential neighborhood, consists mainly with condos and apartments with side parking cars and trees; sunny noon urban surroundings.",
    "kitti_03":"A vehicle driving in-road on a real-world urban road with light traffic, with signs and dense trees on two sides; sunny noon urban surroundings.",
    "kitti_04":"A vehicle driving in-road in a real-world highly populated city downtown with many pedestrains and bike riders, with shops and retail windows; sunny noon city surroundings.",
    "kitti_05":"A vehicle driving in-road in a real-world residential neighborhood, consists mainly with apartments and dense green trees and plants; sunny afternoon urban surroundings.",
    "kitti_06":"A vehicle driving in-road in a real-world residential neighborhood, consists a mix of houses and apartments with parked cars; sunny noon urban surroundings.",
    "kitti_07":"A vehicle driving in-road on a real-world two-sided highway with almost no traffic, sunny noon suburban surroundings.",
    "kitti_08":"A vehicle driving in-road on a real-world one-sided highway with very heavy traffic (many cars), sunny noon suburban surroundings.",
    "nus_01":"A vehicle driving in-road in a real-world less concentrated part of a city, cloudy noon city surroundings.",
    "nus_02":"A vehicle driving in-road in a real-world less concentrated part of a city, cloudy noon city surroundings.",
    "nus_03":"A vehicle driving in-road in a real-world city with moderate traffic, cloudy noon city surroundings.",
    "nus_04":"A vehicle driving in-road in a real-world city with moderate traffic, sunny late afternoon city surroundings.",
    "nus_05":"A vehicle driving in-road in a real-world city with light traffic, sunny afternoon city surroundings.",
    "nus_06":"A vehicle driving in-road in a real-world city with light traffic, sunny afternoon city surroundings.",
    "nus_07":"A vehicle performing a turn in front of two pedestrains in a real-world city with light traffic, sunny morning city surroundings.",
    "nus_08":"A vehicle driving in-road in a real-world city with dense trees on both sides, sunny noon city surroundings.",
    "tfkt":"A vehicle driving in a real-world alike in-road in a busy metropolitan environment with skyscrapers; sunny noon city surroundings."
}

if not do_short:
    f = open(os.path.join(txt_file_loc,"train.txt"),"w")
    for dataset in os.listdir(os.path.join(dataset_loc,"videos")):
        vids = list(os.listdir(os.path.join(dataset_loc,"videos",dataset)))
        vids.sort()
        depth_zero_ct = len(list(os.listdir(os.path.join(dataset_loc,"depth",dataset)))[0].split(".")[0])
        for vid in vids:
            depth_ct = int(vid.split(".")[0])
            new_depth_name = str(depth_ct).zfill(depth_zero_ct) + ".npy"
            new_canny_name = str(depth_ct).zfill(depth_zero_ct) + ".png"
            the_line = "{} {} {} {} {} {} {}\n".format(os.path.join("videos",dataset,vid),os.path.join("depth",dataset),new_depth_name,os.path.join("canny",dataset),new_canny_name,15,desc_edict[dataset])
            f.write(the_line)
    # for ct_ct in os.listdir(os.path.join(".","new_images","videos")):
    #     vids = list(os.listdir(os.path.join(".","new_images","videos",ct_ct)))
    #     vids.sort()
    #     for vid in vids:
    #         depth_ct = int(vid.split(".")[0])
    #         new_depth_name = str(depth_ct).zfill(6) + ".npy"
    #         desc = "A vehicle driving in a real-world city."
    #         the_line = "{} {} {} {} {}\n".format(os.path.join("new_images","videos",ct_ct,vid),os.path.join("new_images","depth",ct_ct),new_depth_name,15,desc)
    #         f.write(the_line)  
else: # for cityscapes
    f = open(os.path.join(txt_file_loc,"train_short.txt"),"w")
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

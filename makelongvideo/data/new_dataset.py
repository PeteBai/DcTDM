import os
import decord
decord.bridge.set_bridge('torch')
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from torch.utils.data import Dataset
from einops import rearrange
from diffusers import AutoencoderKL
from PIL import Image

import random

class MakeDepthVideoDataset(Dataset):
    def __init__(
            self,
            video_dir: str,
            train_list: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
            sample_frame_rates: str = None,
            tokenizer=None,
            pretrained_model_path : str = None
    ):
        self.video_dir = video_dir
        f = open(os.path.join(self.video_dir, train_list))
        self.videolist = [line.strip() for line in f.readlines()]
        random.shuffle(self.videolist)
        f.close()
        self.tokenizer = tokenizer

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rates = [sample_frame_rate]
        self.pretrained_model_path = pretrained_model_path
        self.vae_scale_factor = self.get_vae_scale_factor()
        self.p = transforms.Compose([transforms.Resize(self.height//self.vae_scale_factor,interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)])

        if sample_frame_rates is not None:
            self.sample_frame_rates = [int(n) for n in sample_frame_rates.split(',')]

    def __len__(self):
        return len(self.videolist)

    def get_vae_scale_factor(self):
        vae = AutoencoderKL.from_pretrained(self.pretrained_model_path, subfolder="vae")
        return 2 ** (len(vae.config.block_out_channels) - 1)


    def getvr(self, index, sample_frame_rate):
        # video.mp4 depth_folder depth_start frame_count description
        vr = None

        line = self.videolist[index]
        info = line.split(" ")
        video_path = os.path.join(self.video_dir, info[0])
        prompt = " ".join(info[6:])
        depth_loop_ct = 100
        # depth_loop_ct = 30 if info[1][0] == "n" else 100 # cityscape videos only 30 fps
        filename_len = len(info[2].split(".")[0]) # numebr of leading 0
        start_ct = int(info[2].split(".")[0])
        depth_pth = []
        canny_arr = []
        for i in range(depth_loop_ct):
            name_to_open = str(start_ct+i).zfill(filename_len) + ".npy"
            canny_to_open = str(start_ct+i).zfill(filename_len) + ".png"
            one_depth = np.load(os.path.join(self.video_dir,info[1],name_to_open))
            one_canny = Image.open(os.path.join(self.video_dir,info[3],canny_to_open)).convert('L')
            # print(np.max(np.array(one_canny)))
            depth_pth.append(one_depth)
            canny_arr.append(one_canny)
        depth_pth = np.array(depth_pth)
        canny_arr = np.array(canny_arr)

        sample_frame_rates = self.sample_frame_rates
        sample_frame_rate = sample_frame_rates[random.randint(0,len(sample_frame_rates)-1)]

        scale = 1.0

        # load and sample video frames
        try:
            vr = decord.VideoReader(video_path)

            fps = vr.get_avg_fps()
            scale = 1 if fps <= 30 else fps//24
            # print(fps,len(vr),depth_pth.shape)
            depth_tensor = torch.from_numpy(depth_pth).reshape(len(vr), *one_depth.shape).to(torch.float32)
            canny_arr = torch.from_numpy(canny_arr).reshape(len(vr), *one_canny.size).to(torch.float32)
            
            # assume every video has length>=n_sample_frames 
            min_frame_rate = max(len(vr)//self.n_sample_frames, sample_frame_rates[0])
            sample_frame_rate = min(min_frame_rate, sample_frame_rate)
        except Exception as e:
            print(e, " illegal file", video_path)

        ################## Commit 08-20 #################
        # if sample_frame_rate == sample_frame_rates[0] and random.random() < 0.5:
            # return vr, depth_tensor, prompt, sample_frame_rate
        if len(sample_frame_rates) == 1 or \
                sample_frame_rate == sample_frame_rates[0] and random.random() < 0.5:
            return vr, depth_tensor, canny_arr, prompt, int(sample_frame_rate*scale)
        # return vr, depth_tensor, "{} ...{}x".format(prompt, sample_frame_rate), sample_frame_rate
        return vr, depth_tensor, canny_arr, "{} ...{}x".format(prompt, sample_frame_rate), int(sample_frame_rate*scale)
        ################## Commit CONCLUDE ##############

    def __getitem__(self, index):
        idx = index
        sample_frame_rate = 2

        while True:
            vr, depth_tensor, canny_arr, prompt, sample_frame_rate = self.getvr(idx, sample_frame_rate)

            if vr is None or len(vr) < self.sample_start_idx+1 \
                    or ((len(vr)-self.sample_start_idx)//sample_frame_rate) < self.n_sample_frames+3:
                idx = random.randint(0, len(self.videolist)-1)
                continue

            break

        ###
        framelst = list(range(self.sample_start_idx, len(vr), sample_frame_rate))
        firstidx = random.randint(0,len(framelst)-self.n_sample_frames)
        sample_index = framelst[firstidx:firstidx+self.n_sample_frames]
        ###
        
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")
        f, c, h, w = video.shape
        neww = (self.height*w)//h
        selected_depth = torch.index_select(depth_tensor, 0, torch.tensor(sample_index))
        selected_canny = torch.index_select(canny_arr, 0, torch.tensor(sample_index))
        if neww <= self.width:
            video = F.interpolate(video, (self.height,self.width), mode='bilinear')
            # selected_depth = F.interpolate(selected_depth.unsqueeze(1), (self.height,self.width), mode='bilinear')
            # depth = self.p(selected_depth)
        else:
            video = F.interpolate(video, (self.height,neww), mode='bilinear')
            selected_depth = F.interpolate(selected_depth.unsqueeze(1), (self.height,neww), mode='bilinear')
            selected_canny = F.interpolate(selected_canny.unsqueeze(1), (self.height,neww), mode='bilinear')
            ################## Commit 08-20 #################
            # startpos = random.randint(0,neww-self.width-1)
            offlist = [(neww-self.width)//4, (neww-self.width)//2, (neww-self.width)*3//4]
            startpos = random.choice(offlist)
            ################## Commit CONCLUDE #################
            
            video = video[:,:,:,startpos:startpos+self.width]
            selected_depth = selected_depth[:,:,:,startpos:startpos+self.width]
            selected_canny = selected_canny[:,:,:,startpos:startpos+self.width]
            # depth = self.p(selected_depth)
        # print(video.shape,selected_depth.shape,selected_canny.shape)

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "depth_values": selected_depth,
            "canny_values": selected_canny,
            "prompt_ids": self.tokenizer(
                                prompt,
                                max_length=self.tokenizer.model_max_length,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt"
                            ).input_ids[0]
        }

        return example

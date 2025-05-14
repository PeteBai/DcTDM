import os
import sys
import random
import argparse
from makelongvideo.pipelines.pipeline_makelongvideo import MakeLongVideoPipeline
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from makelongvideo.models.unet import UNet3DConditionModel
from makelongvideo.models.multicontrolnet import MultiControlNetModel
from makelongvideo.models.controlnet3d import ControlNetModel
from makelongvideo.models.train_multi_controlnet import *
from makelongvideo.util import save_videos_grid, ddim_inversion, safetensor_load
import torch
import decord
decord.bridge.set_bridge('torch')
from einops import rearrange
import torch.nn.functional as F
import datetime
from tqdm import tqdm

def randstr(l=16):
  s =''
  chars ='ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'

  for i in range(l):
    s += chars[random.randint(0, len(chars)-1)]

  return  s

def get_edit():
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
    return desc_edict

parser = argparse.ArgumentParser(description='Make Long Video')
parser.add_argument('--prompt', type=str, default=None, required=True, help='prompt')
parser.add_argument('--negprompt', type=str, default=None, help='negtive prompt')
#parser.add_argument('--negprompt', type=str, default='vague, static', help='negtive prompt')
#parser.add_argument('--negprompt', type=str, default='blurred', help='negtive prompt')
parser.add_argument('--n_frames',type=int, default=24, help='total frames generated')
parser.add_argument('--depth_start_file',type=str,required=True,help='depth start file, should be named as a 6 digit number')
parser.add_argument('--canny_start_file',type=str,required=True,help='canny start file, should be named as a 6 digit number')
parser.add_argument('--speed', type=int, default=None, help='playback speed')
parser.add_argument('--inv_latent_path', type=str, default=None, help='inversion latent path')
parser.add_argument('--sample_video_path', type=str, default=None, help='sample video path')
parser.add_argument('--guidance_scale', type=float, default=12.5, help='guidance scale')
#parser.add_argument('--guidance_scale', type=float, default=17., help='guidance scale')
parser.add_argument('--save', action='store_true', default=False, help='save parameters')
parser.add_argument('--batch',type=int,default=None,help="generate videos in batch mode, the number is the # of videos generated")
parser.add_argument('--width', type=int, default=256, help='width')
parser.add_argument('--height', type=int, default=256, help='height')
parser.add_argument('--out_folder', type=str, help='name of the sub folder in result')
parser.add_argument('--full_edit', action='store_true', default=False)
parser.add_argument('--use_corr_depth',action='store_true',default=False)
parser.add_argument('--config_file',type=str,default=None)
parser.add_argument('--run_all',action="store_true",default=False)
parser.add_argument('--do_gif',action="store_true",default=False)
parser.add_argument('--do_depth',action="store_true",default=False)
parser.add_argument('--do_canny',action="store_true",default=False)
parser.add_argument('--my_model_path', type=str, help='trained ckpt', required=True)

args = parser.parse_args()

pretrained_model_path = "./checkpoints/stable-diffusion-v1-5/"

controlnet = []
if args.do_depth:
    controlnet.append(ControlNetModel.from_pretrained_2d(pretrained_model_path="./checkpoints/ControlNet/models/control_sd15_depth"))
if args.do_canny:
    controlnet.append(ControlNetModel.from_pretrained_2d(pretrained_model_path="./checkpoints/ControlNet/models/control_sd15_canny"))

my_model_path = args.my_model_path
unet = UNet3DConditionModel.from_pretrained(my_model_path+"/../", subfolder='unet', torch_dtype=torch.float16).to('cuda')
state_dict = unet.state_dict()
if args.do_depth and args.do_canny:
    controlnet[0] = safetensor_load(os.path.join(my_model_path, 'model.safetensors'), controlnet[0]).to(device="cuda", dtype=torch.float16)
    controlnet[1] = safetensor_load(os.path.join(my_model_path, 'model_1.safetensors'), controlnet[1]).to(device="cuda", dtype=torch.float16)
elif not args.do_depth and not args.do_canny:
    pass
else:
    controlnet[0] = safetensor_load(os.path.join(my_model_path, 'model.safetensors'), controlnet[0]).to(device="cuda",dtype=torch.float16)

#print(state_dict['up_blocks.2.attentions.0.transformer_blocks.0.temporal_rel_pos_bias.net.2.weight'])
#print(state_dict['up_blocks.2.attentions.2.transformer_blocks.0.attn_temp.to_q.weight'])

if args.save:
    print(state_dict)
    sys.exit(0)

pipeline = MakeLongVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
# pipeline.enable_xformers_memory_efficient_attention()
pipeline.enable_vae_slicing()

ddim_inv_latent = None
n_sample_frames = args.n_frames

if args.sample_video_path is not None:
    noise_scheduler = DDPMScheduler.from_pretrained(my_model_path, subfolder="scheduler")

    ddim_inv_scheduler = DDIMScheduler.from_pretrained(my_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(50)

    sample_start_idx = 0
    sample_frame_rate = 2
    
    vr = decord.VideoReader(args.sample_video_path, width=args.width, height=args.height)
    framelst = list(range(sample_start_idx, len(vr), sample_frame_rate))
    sample_index = framelst[0:n_sample_frames]
    video = vr.get_batch(sample_index)
    pixel_values = rearrange(video, "(b f) h w c -> b f c h w", f=n_sample_frames) / 127.5 - 1.0
    b, f, c, h, w = pixel_values.shape
    video_length = pixel_values.shape[1]
    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
    pixel_values = pixel_values.to('cuda', dtype=torch.float16)
    with torch.no_grad():
        latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215
    '''
    ###
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # Sample a random timestep for each video
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    latents = noise_scheduler.add_noise(latents, noise, timesteps)
    ###
    '''
    ddim_inv_latent = ddim_inversion(
        pipeline, ddim_inv_scheduler, video_latent=latents, num_inv_steps=50, prompt=""
        )[-1].to(torch.float16)
elif args.inv_latent_path is not None:
    ddim_inv_latent = torch.load(args.inv_latent_path).to(torch.float16)
#else:
elif False:
    ddim_inv_latent = torch.randn([1, 4, 24, 64, 64]).to(torch.float16)
    #ddim_inv_latent = torch.randn([1, 4, 1, 64, 64]).repeat_interleave(24,dim=2)

prompt = "{} ...{}x".format(args.prompt, args.speed) if args.speed is not None else args.prompt

print('prompt:', prompt)
if args.batch is not None:
    str_seq = args.depth_start_file.split("/")[-1].split(".")[0]
    for i in range(args.batch):
        now = datetime.datetime.now()
        curr_pos = int(str_seq) + i * n_sample_frames # no overlapping at all
        len_str_seq = len(str_seq)
        depth_name = str(curr_pos).zfill(len_str_seq) + ".npy"
        canny_name = str(curr_pos).zfill(len_str_seq) + ".png"
        full_depth_name = "/".join(args.depth_start_file.split("/")[:-1]) + "/" + depth_name
        full_canny_name = "/".join(args.canny_start_file.split("/")[:-1]) + "/" + canny_name
        video = pipeline(prompt, latents=ddim_inv_latent, video_length=n_sample_frames\
        , height=args.height, width=args.width, num_inference_steps=50, guidance_scale=args.guidance_scale, \
        negative_prompt=args.negprompt, start_depth=full_depth_name, start_canny=full_canny_name, \
            use_depth=args.do_depth, controlnet_joint_modal=controlnet).videos
        out_full_path = os.path.join("./outputs/results/icra",args.out_folder)

        if not os.path.exists(out_full_path):
            os.makedirs(out_full_path)

        fps = 24//args.speed if args.speed is not None else 12
        date_time_str = now.strftime("%m%d_%H%M%S")
        if args.speed is not None:
            resultfile = f"./outputs/results/{args.out_folder}/{args.prompt[:16]}-{randstr(6)}-{args.speed}x.gif"
        else:
            resultfile = f"{out_full_path}/{args.prompt[:16]}-{date_time_str}"

        save_videos_grid(video, resultfile, fps=fps,do_gif=args.do_gif)

else:
    out_full_path = os.path.join("./outputs/results",args.out_folder)
    if not os.path.exists(out_full_path):
        os.makedirs(out_full_path, exist_ok=True)
    if args.full_edit and not args.run_all:
        for key, value in get_edit().items():
            if args.use_corr_depth:
                depth_batch_file = "/work/zura-storage/Data/DrivingSceneDDM/depth/{}/{}.npy"
                canny_batch_file = "/work/zura-storage/Data/DrivingSceneDDM/canny/{}/{}.png"
                file_name = "0000000100" if key[:5] == "kitti" else "000100"
                final_name = depth_batch_file.format(key,file_name)
                final_canny_name = canny_batch_file.format(key,file_name)
            else:
                final_name = args.depth_start_file
                final_canny_name = args.canny_start_file
            video = pipeline(value, latents=ddim_inv_latent, video_length=n_sample_frames\
                , height=args.height, width=args.width, num_inference_steps=50, guidance_scale=args.guidance_scale, \
                negative_prompt=args.negprompt, start_depth=final_name, start_canny=final_canny_name,use_depth=args.do_depth ,controlnet_joint_modal=controlnet).videos
            print(value)
            if not os.path.exists(out_full_path):
                os.mkdir(out_full_path)

            fps = 24//args.speed if args.speed is not None else 12

            if args.speed is not None:
                resultfile = f"./outputs/results/{args.out_folder}/{key}.gif"
            else:
                resultfile = f"./outputs/results/{args.out_folder}/{key}"

            save_videos_grid(video, resultfile, fps=fps)
    elif args.full_edit and args.run_all:
        total = 0
        for key, value in get_edit().items():
            parent_fldr = "/work/zura-storage/Data/DrivingSceneDDM/{}/{}/"
            img_list = list(os.listdir(parent_fldr.format("depth", key)))
            canny_list = list(os.listdir(parent_fldr.format("canny", key)))
            img_len = len(img_list)
            img_list.sort()
            print(key)
            for i in tqdm(range(img_len // args.n_frames)):
                start_depth = parent_fldr.format("depth",key) + img_list[i*args.n_frames]
                start_canny = parent_fldr.format("canny",key) + canny_list[i*args.n_frames]
                video = pipeline(value, latents=ddim_inv_latent, video_length=n_sample_frames\
                    , height=args.height, width=args.width, num_inference_steps=50, guidance_scale=args.guidance_scale, \
                    negative_prompt=args.negprompt, start_depth=start_depth, start_canny=start_canny,use_depth=args.do_depth,controlnet_joint_modal=controlnet).videos
                fps = 24//args.speed if args.speed is not None else 12
                if args.speed is not None:
                    resultfile = "./outputs/results/{}/{}/{:03d}".format(args.out_folder,key,total)
                else:
                    resultfile = "./outputs/results/{}/{}/{:03d}".format(args.out_folder,key,total)
                total += 1
                save_videos_grid(video, resultfile, fps=fps, do_gif=args.do_gif)  
                
    elif args.config_file is not None:
        with open(args.config_file,'r') as f:
            content = f.readlines()
        for line in content:
            now = datetime.datetime.now()
            (depth_loc,description) = line.split("x-x")
            print(depth_loc,description)
            frame_idx = depth_loc.split("/")[-1].split(".")[0]
            dataset_name = depth_loc.split("/")[-2]
            canny_loc = "/".join(depth_loc.split("/")[:-3]) + "/canny/" + dataset_name + "/" + frame_idx + ".png"
            video = pipeline(description, latents=ddim_inv_latent, video_length=n_sample_frames\
                , height=args.height, width=args.width, num_inference_steps=50, guidance_scale=args.guidance_scale, \
                negative_prompt=args.negprompt, start_depth=depth_loc, start_canny=canny_loc, use_depth=args.do_depth,controlnet_joint_modal=controlnet).videos
            fps = 24//args.speed if args.speed is not None else 12
            date_time_str = now.strftime("%m%d_%H%M%S")

            if args.speed is not None:
                resultfile = f"./outputs/results/{args.out_folder}/{date_time_str}.gif"
            else:
                resultfile = f"./outputs/results/{args.out_folder}/{date_time_str}"
            
            save_videos_grid(video, resultfile, fps=fps)
    else:
        video = pipeline(prompt, latents=ddim_inv_latent, video_length=n_sample_frames\
        , height=args.height, width=args.width, num_inference_steps=50, guidance_scale=args.guidance_scale, \
        negative_prompt=args.negprompt, start_depth=args.depth_start_file, start_canny=args.canny_start_file, \
            use_depth=args.do_depth, controlnet_joint_modal=controlnet).videos

        fps = 24//args.speed if args.speed is not None else 12

        if args.speed is not None:
            resultfile = f"./outputs/results/{args.out_folder}/{args.prompt[:16]}-{randstr(6)}-{args.speed}x.gif"
        else:
            resultfile = f"./outputs/results/{args.out_folder}/{args.prompt[:16]}-{randstr(6)}"

        save_videos_grid(video, resultfile, fps=fps)

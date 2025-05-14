from add_midas import AddMiDaS
from PIL import Image
from tqdm import tqdm
from midas.api import MiDaSInference
from einops import repeat, rearrange
import numpy as np
import torch
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas_trafo = AddMiDaS(model_type="dpt_hybrid")
parent_loc = r"/mnt/data/dataset/new_images/cityscape"
num_samples = 1

for dataset in os.listdir(parent_loc):
    imgs = list(os.listdir(os.path.join(parent_loc, dataset)))
    imgs.sort()
    for img_loc in tqdm(imgs):
        image = Image.open(os.path.join(parent_loc,dataset,img_loc))
        image = np.array(image.convert("RGB"))
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        with torch.no_grad(),\
                    torch.autocast("cuda"):
            batch = {
                "jpg": image
            }
            batch = midas_trafo(batch)
            batch["jpg"] = rearrange(batch["jpg"], 'h w c -> 1 c h w')
            batch["jpg"] = repeat(batch["jpg"].to(device=device),
                                "1 ... -> n ...", n=num_samples)
            batch["midas_in"] = repeat(torch.from_numpy(batch["midas_in"][None, ...]).to(
                device=device), "1 ... -> n ...", n=num_samples)
            # print(batch)
            depth_model = MiDaSInference("dpt_hybrid")
            cc = depth_model(batch["midas_in"])

            depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                        keepdim=True)
            cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.
            new_name = img_loc.split(".")[0] + ".npy"
            new_path = os.path.join(parent_loc,"..","depth",dataset)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            np.save(os.path.join(new_path,new_name),cc.cpu().detach().numpy()[0][0])
# c_cat.append(cc)
# c_cat = torch.cat(c_cat, dim=1)
# cond = {"c_concat": [c_cat], "c_crossattn": [c]}
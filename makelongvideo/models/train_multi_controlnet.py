def multi_control_forward(
    sample,
    timestep,
    encoder_hidden_states,
    controlnet_cond,
    conditioning_scale,
    controlnets,
    class_labels=None,
    timestep_cond=None,
    attention_mask=None,
    added_cond_kwargs=None,
    cross_attention_kwargs=None,
    guess_mode=False,
    return_dict=True):

    for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, controlnets)):
        # print(scale)
        down_samples, mid_sample = controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=image,
            conditioning_scale=scale,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs=cross_attention_kwargs,
            guess_mode=guess_mode,
            return_dict=return_dict,
        )
        # merge samples
        if i == 0:
            down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
        else:
            down_block_res_samples = [
                samples_prev + samples_curr
                for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
            ]
            mid_block_res_sample += mid_sample
    return down_block_res_samples, mid_block_res_sample

def enable_xformer_control(controlnets):
    for idx in range(len(controlnets)):
        controlnets[idx].enable_xformers_memory_efficient_attention()
    return controlnets

def enable_gradient_checkpointing_control(controlnets):
    for idx in range(len(controlnets)):
        controlnets[idx].enable_gradient_checkpointing()
    return controlnets

def get_optimizer_param_control(controlnets):
    trainable_modules = ("attn1.to_out","attn_temp","norm_temp","conv_temp")
    # trainable_modules = ("attn1.to_q")
    optimize_params = []
    for controlnet in controlnets:
        for name, module in controlnet.named_modules():
            if any(ele in name for ele in trainable_modules):
                print(name)
                optimize_params += list(module.parameters())
                for params in module.parameters():
                    params.requires_grad = True
    return optimize_params
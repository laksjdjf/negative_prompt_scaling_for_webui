#negative scaler script for webui

import torch
import numpy as np

import modules.scripts as scripts
import gradio as gr

import importlib

import modules.processing
from modules.processing import StableDiffusionProcessing
from modules import prompt_parser,devices

from ldm.modules.diffusionmodules.util import noise_like
import ldm.models.diffusion.ddim as ddim
import ldm.models.diffusion.plms as plms

SCALE_MAX = 100

class OrgDDIMSampler(ddim.DDIMSampler):
    pass

class OrgPLMSSampler(plms.PLMSSampler):
    pass

class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "negative prompt scaling"

    def ui(self, is_img2img):
        with gr.Row():
            gr.Markdown(
                """
                Set sampler to DDIM or PLMS(^q^)
                """)
        with gr.Row():
            enable_negative_scale = gr.Checkbox(label='Enable negative scale', value=False)
            use_guidance_scale = gr.Checkbox(label='Use same value as guidance scale (ignore below slider)', value=False) #改行ってどうするの？
        with gr.Row():    
            negative_scale = gr.Slider(0, SCALE_MAX, value=7,step=0.5,label='negative_scale')
    
        with gr.Row():
            gr.Markdown(
                """
                Linearly increasing(decreasing) of guidance(negative) scale.

                If negative scale is disabled, generation time remains the same.
                """)
        with gr.Row():
            enable_scaling= gr.Checkbox(label='Enable scale scaling', value=False)
            use_guidance_scale_scaling = gr.Checkbox(label='Use same value as guidance scale (ignore negative slider)', value=False)
        with gr.Row():    
            initial_guidance_scale = gr.Slider( 1, SCALE_MAX, value=7,step=0.5,label='guidance_scale of initial step')
            last_guidance_scale = gr.Slider( 1, SCALE_MAX, value=7,step=0.5,label='guidance_scale of last step')
        with gr.Row():    
            initial_negative_scale = gr.Slider( 1, SCALE_MAX, value=7,step=0.5,label='negative_scale of initial step')
            last_negative_scale = gr.Slider( 1, SCALE_MAX, value=7,step=0.5,label='negative_scale of last step')
        with gr.Row():
            latent_scale = gr.Slider(0.9,1.0,value=0.99,step=0.001,label='latent scale')

        return [enable_negative_scale,use_guidance_scale ,negative_scale,enable_scaling,use_guidance_scale_scaling,initial_guidance_scale,last_guidance_scale,initial_negative_scale,last_negative_scale,latent_scale]
    
    def run(self, p: StableDiffusionProcessing,enable_negative_scale,use_guidance_scale ,negative_scale,enable_scaling,use_guidance_scale_scaling,initial_guidance_scale,last_guidance_scale,initial_negative_scale,last_negative_scale,latent_scale):
        if not ( enable_negative_scale or enable_scaling ):
            return modules.processing.process_images(p)

        cfg_repeat = 3 if enable_negative_scale else 2
        ##################################################################################################################
        #samplerの関数書き換え用クラス、関数の中にクラスとかきもいし読みづらいけど、run内のローカル変数を使いたい。global変数うまくつかえばいいのかな？
        ##################################################################################################################
        #monkey patch CFG in ddimsampler
        #https://github.com/Stability-AI/stablediffusion/blob/main/ldm/models/diffusion/ddim.py
        class WrapDDIMSampler(ddim.DDIMSampler):
            def __init__(self, model, schedule="linear", **kwargs):
                super().__init__(model, schedule="linear", **kwargs)
                #反映できているか確認しておく
                if enable_scaling:
                    print(f"using scale scaling:{guidance_scale_schedule[-1]} to {guidance_scale_schedule[0]} !!!!!!!!!!!!")
                    if enable_negative_scale:
                        print(f"using negative scale scaling:{negative_scale_schedule[-1]} to {negative_scale_schedule[0]} !!!!!!!!!!!!")
                else:
                    print(f"using negative_scale:{negative_scale} !!!!!!!!!!!!")
            @torch.no_grad()
            def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                                  temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                                  unconditional_guidance_scale=1., unconditional_conditioning=None,
                                  dynamic_threshold=None):
                b, *_, device = *x.shape, x.device
                
                if enable_scaling:
                    guidance_scale_step = guidance_scale_schedule[index]
                    negative_scale_step = negative_scale_schedule[index]
                else:
                    guidance_scale_step = unconditional_guidance_scale
                    negative_scale_step = negative_scale
                if unconditional_conditioning is None:
                    model_output = self.model.apply_model(x, t, c)
                else:
                    #入力の拡張を2⇒3へ
                    x_in = torch.cat([x] * cfg_repeat)
                    t_in = torch.cat([t] * cfg_repeat)
                    if enable_negative_scale:
                        assert isinstance(unconditional_conditioning, dict)
                        c_in = dict()

                        #なんかぐちゃぐちゃなのでだれかたすけて
                        #そもそも"c_concat"になにが入っているか分からない、condにもuncondにも同じものいれていたのでこうしてるけどいいのかな？）
                        uc_dict = {"c_concat":c["c_concat"],"c_crossattn":[uc_t[0].cond.unsqueeze(0).repeat(p.batch_size,1,1) for uc_t in uc]}
                        for k in c:
                            if isinstance(c[k], list):
                                c_in[k] = [torch.cat([
                                    unconditional_conditioning[k][i],
                                    c[k][i],uc_dict[k][i]]) for i in range(len(c[k]))]
                            else:
                                c_in[k] = torch.cat([
                                        unconditional_conditioning[k],
                                        c[k],uc_dict[k][0]])
                    else:
                        assert isinstance(unconditional_conditioning, dict)
                        c_in = dict()
                        for k in c:
                            if isinstance(c[k], list):
                                c_in[k] = [torch.cat([
                                    unconditional_conditioning[k][i],
                                    c[k][i]]) for i in range(len(c[k]))]
                            else:
                                c_in[k] = torch.cat([
                                        unconditional_conditioning[k],
                                        c[k]])

                    #このプログラムの核部分
                    if enable_negative_scale:
                        model_negative, model_t, model_uncond = self.model.apply_model(x_in, t_in, c_in).chunk(3)
                        model_output = model_uncond + guidance_scale_step * (model_t - model_uncond) - negative_scale_step * (model_negative - model_uncond)
                    else:
                        model_negative, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                        model_output = model_negative + guidance_scale_step * (model_t - model_negative)

                #これ以降何起きてるか知らない。
                if self.model.parameterization == "v":
                    e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
                else:
                    e_t = model_output

                if score_corrector is not None:
                    assert self.model.parameterization == "eps", 'not implemented'
                    e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

                alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
                alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
                sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
                sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
                # select parameters corresponding to the currently considered timestep
                a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
                a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
                sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
                sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

                # current prediction for x_0
                if self.model.parameterization != "v":
                    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
                else:
                    pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

                if quantize_denoised:
                    pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

                if dynamic_threshold is not None:
                    raise NotImplementedError()

                # direction pointing to x_t
                dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
                noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
                if noise_dropout > 0.:
                    noise = torch.nn.functional.dropout(noise, p=noise_dropout)
                x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
                x_prev *= latent_scale
                return x_prev, pred_x0


        ##########################################################################################################
        #PLMSSampler https://github.com/Stability-AI/stablediffusion/blob/main/ldm/models/diffusion/plms.py
        class WrapPLMSSampler(plms.PLMSSampler):
            def __init__(self, model, schedule="linear", **kwargs):
                super().__init__(model, schedule="linear", **kwargs)
                #反映できているか確認しておく
                if enable_scaling:
                    print(f"using scale scaling:{guidance_scale_schedule[-1]} to {guidance_scale_schedule[0]} !!!!!!!!!!!!")
                    if enable_negative_scale:
                        print(f"using negative scale scaling:{negative_scale_schedule[-1]} to {negative_scale_schedule[0]} !!!!!!!!!!!!")
                else:
                    print(f"using negative_scale:{negative_scale} !!!!!!!!!!!!")
            @torch.no_grad()
            def p_sample_plms(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                              temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                              unconditional_guidance_scale=1., unconditional_conditioning=None, old_eps=None, t_next=None,
                              dynamic_threshold=None):
                b, *_, device = *x.shape, x.device
                if enable_scaling:
                    guidance_scale_step = guidance_scale_schedule[index]
                    negative_scale_step = negative_scale_schedule[index]
                else:
                    guidance_scale_step = unconditional_guidance_scale
                    negative_scale_step = negative_scale
                def get_model_output(x, t):
                    if unconditional_conditioning is None:
                        e_t = self.model.apply_model(x, t, c)
                    else:
                        #入力の拡張を2⇒3にする
                        x_in = torch.cat([x] * cfg_repeat)
                        t_in = torch.cat([t] * cfg_repeat)
                        #動けばいいやの精神でddimから丸写し
                        if enable_negative_scale:
                            assert isinstance(unconditional_conditioning, dict)
                            c_in = dict()

                            #なんかぐちゃぐちゃなのでだれかたすけて
                            #そもそも"c_concat"になにが入っているか分からない、condにもuncondにも同じものいれていたのでこうしてるけどいいのかな？）
                            uc_dict = {"c_concat":c["c_concat"],"c_crossattn":[uc_t[0].cond.unsqueeze(0).repeat(p.batch_size,1,1) for uc_t in uc]}
                            for k in c:
                                if isinstance(c[k], list):
                                    c_in[k] = [torch.cat([
                                        unconditional_conditioning[k][i],
                                        c[k][i],uc_dict[k][i]]) for i in range(len(c[k]))]
                                else:
                                    c_in[k] = torch.cat([
                                            unconditional_conditioning[k],
                                            c[k],uc_dict[k][0]])
                        else:
                            assert isinstance(unconditional_conditioning, dict)
                            c_in = dict()
                            for k in c:
                                if isinstance(c[k], list):
                                    c_in[k] = [torch.cat([
                                        unconditional_conditioning[k][i],
                                        c[k][i]]) for i in range(len(c[k]))]
                                else:
                                    c_in[k] = torch.cat([
                                            unconditional_conditioning[k],
                                            c[k]])
                            
                        #このプログラムの核部分
                        if enable_negative_scale:
                            e_t_negative, e_t, e_t_uncond = self.model.apply_model(x_in, t_in, c_in).chunk(3)
                            e_t = e_t_uncond + guidance_scale_step * (e_t - e_t_uncond) - negative_scale_step * (e_t_negative - e_t_uncond)
                        else:
                            e_t_negative, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                            e_t = e_t_negative + guidance_scale_step * (e_t - e_t_negative)

                    if score_corrector is not None:
                        assert self.model.parameterization == "eps"
                        e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

                    return e_t

                alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
                alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
                sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
                sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

                def get_x_prev_and_pred_x0(e_t, index):
                    # select parameters corresponding to the currently considered timestep
                    a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
                    a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
                    sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
                    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

                    # current prediction for x_0
                    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
                    if quantize_denoised:
                        pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
                    if dynamic_threshold is not None:
                        pred_x0 = norm_thresholding(pred_x0, dynamic_threshold)
                    # direction pointing to x_t
                    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
                    noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
                    if noise_dropout > 0.:
                        noise = torch.nn.functional.dropout(noise, p=noise_dropout)
                    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
                    x_prev *= latent_scale
                    return x_prev, pred_x0

                e_t = get_model_output(x, t)
                if len(old_eps) == 0:
                    # Pseudo Improved Euler (2nd order)
                    x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
                    e_t_next = get_model_output(x_prev, t_next)
                    e_t_prime = (e_t + e_t_next) / 2
                elif len(old_eps) == 1:
                    # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
                    e_t_prime = (3 * e_t - old_eps[-1]) / 2
                elif len(old_eps) == 2:
                    # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
                    e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
                elif len(old_eps) >= 3:
                    # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
                    e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

                x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

                return x_prev, pred_x0, e_t

        ###############################################################################################################
        ###############################################################################################################

        #私みたいなめんどくさがりのためにg=nにしたいときはnを設定しなくてもいいようにしておく        
        if use_guidance_scale:
            negative_scale = p.cfg_scale

        #逆拡散過程から見ているのでlast→initialとする
        guidance_scale_schedule = np.linspace(last_guidance_scale, initial_guidance_scale, p.steps)
        if use_guidance_scale_scaling:
            negative_scale_schedule = guidance_scale_schedule
        else:
            negative_scale_schedule = np.linspace(last_negative_scale, initial_negative_scale, p.steps)

        #unconditionのテキスト埋め込みベクトル
        with devices.autocast():
            uc = prompt_parser.get_learned_conditioning(p.sd_model, [""], p.steps)

        #DDIMSamplerを改造する
        ddim.DDIMSampler = WrapDDIMSampler
        plms.PLMSSampler = WrapPLMSSampler
        #モジュールのリロードで無理やり上の変更を反映
        importlib.reload(modules.processing)
        result = modules.processing.process_images(p)

        #元に戻っていることをお祈りする
        ddim.DDIMSampler = OrgDDIMSampler
        plms.PLMSSampler = OrgPLMSSampler
        importlib.reload(modules.processing)

        return result

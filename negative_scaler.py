#negative scaler script for webui

import torch

import modules.scripts as scripts
import gradio as gr

import importlib

import modules.processing
from modules.processing import StableDiffusionProcessing
from modules import prompt_parser,devices

from ldm.modules.diffusionmodules.util import noise_like
import ldm.models.diffusion.ddim as ddim

class OrgDDIMSampler(ddim.DDIMSampler):
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
                Set sampler to DDIM (^q^)
                """)
        with gr.Row():
            enabled = gr.Checkbox(label='Enable', value=False)
            use_guidance_scale = gr.Checkbox(label='Use same value as guidance scale (ignore below slider)', value=False) #改行ってどうするの？
        with gr.Row():    
            negative_scale = gr.Slider(0, 50, value=7,step=0.5,label='negative_scale')
        return [enabled,use_guidance_scale ,negative_scale]
    
    def run(self, p: StableDiffusionProcessing, enabled,use_guidance_scale,negative_scale):
        if not enabled:
            return modules.processing.process_images(p)

        #monkey patch CFG in ddimsampler
        #https://github.com/Stability-AI/stablediffusion/blob/main/ldm/models/diffusion/ddim.py
        class WrapDDIMSampler(ddim.DDIMSampler):
            def __init__(self, model, schedule="linear", **kwargs):
                super().__init__(model, schedule="linear", **kwargs)
                #反映できているか確認しておく
                print(f"using negative_scale:{negative_scale}!!!!!!!!!!!!!!!!!!!!!")
            @torch.no_grad()
            def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                                  temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                                  unconditional_guidance_scale=1., unconditional_conditioning=None,
                                  dynamic_threshold=None):
                b, *_, device = *x.shape, x.device
                #print(f"using negative_scale:{negative_scale} !!!!!!!!!!!11!!!")
                if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                    model_output = self.model.apply_model(x, t, c)
                else:

                    #入力の拡張を2⇒3へ
                    x_in = torch.cat([x] * 3)
                    t_in = torch.cat([t] * 3)
                    if isinstance(c, dict):
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
                    elif isinstance(c, list):

                        #こっちになることがあったりするのかもわからない
                        c_in = list()
                        assert isinstance(unconditional_conditioning, list)
                        for i in range(len(c)):
                            c_in.append(torch.cat([unconditional_conditioning[i], c[i], uc[i]]))
                    else:
                        c_in = torch.cat([unconditional_conditioning, c, uc])

                    #このプログラムの核部分
                    model_negative, model_t, model_uncond = self.model.apply_model(x_in, t_in, c_in).chunk(3)
                    model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond) - negative_scale * (model_negative - model_uncond)

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
                return x_prev, pred_x0

        #私みたいなめんどくさがりのためにg=nにしたいときはnを設定しなくてもいいようにしておく        
        if use_guidance_scale:
            negative_scale = p.cfg_scale

        #unconditionのテキスト埋め込みベクトル
        with devices.autocast():
            uc = prompt_parser.get_learned_conditioning(p.sd_model, [""], p.steps)

        #DDIMSamplerを改造する
        ddim.DDIMSampler = WrapDDIMSampler
        #モジュールのリロードで無理やり上の変更を反映
        importlib.reload(modules.processing)
        result = modules.processing.process_images(p)

        #元に戻っていることをお祈りする
        ddim.DDIMSampler = OrgDDIMSampler
        importlib.reload(modules.processing)

        return result

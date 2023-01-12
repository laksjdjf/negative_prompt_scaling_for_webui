# Negative prompt scaler for webui

let $n$ = negative prompt emb, $c$ = prompt emb, $u$ = empty prompt emb, $\epsilon$ = output of UNet, $x$ = latent

The formula for using [negative prompt](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Negative-prompt) is as follows,

$$\epsilon(x|n) + g_{scale} * (\epsilon(x|c) - \epsilon(x|n)) $$

This replaces $u$ with $n$ in the CFG formula.

However it cant scaling negative prompt.

Furthermore the smaller the guidance scale, the less effective the negative prompt (try substituting 1 for $g_{scale}$).

This script replaces this formula as follows,

$$\epsilon(x|u) + g_{scale} * (\epsilon(x|c) - \epsilon(x|n)) - n_{scale} * (\epsilon(x|n) - \epsilon(x|u))$$

$n_{scale}$ scales negative prompt.

**As shown in the formula, the computational complexity is 1.5 times higher because three noises need to be inferenced.**

## Usage

put negative_scaler.py in scripts. see [it](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Scripts).

set "negative prompt scaling" in Scripts drop-down menu and check "enable".

**Only DDIM is applicable.**

**This script is replacing a module, so unintended effects may be added somewhere. Use at your own risk.**

I am assuming using txt2img, but it seems to work with img2img.

![ui](https://github.com/laksjdjf/negative_prompt_scaling_for_webui/blob/images/ui.png?raw=true)

## Examples
model: https://huggingface.co/hakurei/waifu-diffusion-v1-4

prompt: masterpiece,best quality,1girl,solo,standing,blush,red eyes,blonde hair,twintails,hair ribbon,school uniform,blue sailor collar,blue skirt,black thighhighs

negative: prompt:worst quality, low quality, medium quality, deleted, lowres, comic, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry

seed: 4545,...

guidance_scale: 7

$n_{scale} = 0$
![0](https://github.com/laksjdjf/negative_prompt_scaling_for_webui/blob/images/example0.png?raw=true)
$n_{scale} = 3.5$
![1](https://github.com/laksjdjf/negative_prompt_scaling_for_webui/blob/images/example1.png?raw=true)
$n_{scale} = 7$
![2](https://github.com/laksjdjf/negative_prompt_scaling_for_webui/blob/images/example2.png?raw=true)
$n_{scale} = 10.5$
![3](https://github.com/laksjdjf/negative_prompt_scaling_for_webui/blob/images/example3.png?raw=true)
$n_{scale} = 14$
![4](https://github.com/laksjdjf/negative_prompt_scaling_for_webui/blob/images/example4.png?raw=true)

# Negative prompt scaler for webui

let $n$ = negative prompt emb, $c$ = prompt emb, $u$ = empty prompt emb, $\epsilon$ = output of UNet, $x$ = latent

The formula for using [negative prompt](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Negative-prompt) is as follows,

$$\epsilon(x|n) + g_{scale} * (\epsilon(x|c) - \epsilon(x|n)) $$

This replaces $u$ with $n$ in the CFG formula.

However it cant scaling negative prompt.

Furthermore the smaller the guidance scale, the less effective the negative prompt (try substituting 1 for $g_{scale}$).

This script replaces this formula as follows,

$$\epsilon(x|u) + g_{scale} * (\epsilon(x|c) - \epsilon(x|u)) - n_{scale} * (\epsilon(x|n) - \epsilon(x|u))$$

$n_{scale}$ scales negative prompt.

**As shown in the formula, the computational complexity is 1.5 times higher because three noises need to be inferenced.**

## Usage

put negative_scaler.py in stable-diffusion-webui/scripts. see [it](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Scripts).

set "negative prompt scaling" in Scripts drop-down menu and check "enable".

**Only DDIM is applicable.**

**This script is replacing a module, so unintended effects may be added somewhere. Use at your own risk.**

I am assuming using txt2img, but it seems to work with img2img.

![ui](https://github.com/laksjdjf/negative_prompt_scaling_for_webui/blob/images/ui.png?raw=true)

## Examples
[Examle.md](https://github.com/laksjdjf/negative_prompt_scaling_for_webui/blob/main/Examle.md)

"""
Photorealistic Animal Dataset Generator - Production Configuration
==================================================================
Optimized with Background Caching, Alpha Matting, and Kaggle-Ready Augmentations.
"""

import random
import warnings
from pathlib import Path

import numpy as np
import torch
import albumentations as A
from PIL import Image
from diffusers import (
    AutoencoderKL,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from rembg import new_session, remove as rembg_remove

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────── CONFIG ──────────────────────────────────────

MODEL_ID  = "SG161222/Realistic_Vision_V5.1_noVAE"
VAE_ID    = "stabilityai/sd-vae-ft-mse"

CLASSES = {0: "fox", 1: "leopard", 2: "sheep"}

ANIMAL_PROMPTS = {
    0: [
        "RAW photo of a red fox sitting on grass, Canon EOS R5, 85mm f/1.4, sharp focus, natural fur texture, bokeh background, 8k uhd, photorealistic, white background",
        "RAW photo of a red fox standing alert, looking forward, Fujifilm XT3, natural lighting, highly detailed fur, white background, photorealistic",
        "RAW photo of a red fox lying down, relaxed, DSLR, natural lighting, sharp fur detail, photorealistic, white background",
        "RAW photo of a red fox walking, side profile, 85mm lens, natural lighting, photorealistic, detailed fur texture, white background",
        "RAW close-up portrait of a red fox, 100mm macro, shallow depth of field, individual fur strands visible, photorealistic, white background",
        "RAW photo of a red fox crouching low to the ground, stalking pose, dynamic angle, highly detailed, photorealistic, white background",
        "RAW photo of a red fox mid-stride, running action pose, sharp focus, side view, natural muscle definition, photorealistic, white background",
        "RAW photo of a red fox looking over its shoulder, back profile, high-end photography, sharp details, photorealistic, white background",
        "RAW photo of a red fox curling up tightly, resting posture, compact silhouette, soft lighting, photorealistic, white background",
        "RAW photo of a red fox with its head tilted curiously, tight portrait shot, expressive eyes, sharp whiskers, photorealistic, white background",
        "RAW low-angle shot of a red fox standing proudly, looking upward, dramatic natural lighting, highly detailed, photorealistic, white background",
        "RAW photo of a red fox yawning, open mouth detail, visible teeth and tongue, macro lens, sharp focus, photorealistic, white background",
        "RAW photo of a red fox stretching its front legs forward, elongated body posture, full body view, photorealistic, white background",
        "RAW three-quarter portrait of an adult red fox, sharp eye contact, studio lighting style, flawless fur texture, photorealistic, white background",
        "RAW photo of a slender red fox stepping forward cautiously, front-facing action pose, highly detailed coat, photorealistic, white background",
    ],
    1: [
        "RAW photo of a leopard sitting, Canon EOS R5, 400mm telephoto, sharp spotted coat, natural lighting, 8k uhd, photorealistic, white background",
        "RAW photo of a leopard standing alert, Nikon D850, natural savanna light, detailed rosette pattern, photorealistic, white background",
        "RAW photo of a leopard walking, side profile, telephoto lens, motion-ready pose, photorealistic, detailed coat, white background",
        "RAW photo of a leopard lying down, relaxed, DSLR, dappled natural light, photorealistic fur detail, white background",
        "RAW close-up portrait of a leopard, 100mm lens, shallow DOF, eye detail, whisker texture, photorealistic, white background",
        "RAW photo of a leopard crouching, low angle, dramatic natural lighting, photorealistic, detailed spotted coat, white background",
        "RAW photo of a leopard pacing forward directly toward the camera, powerful front profile, fierce gaze, highly detailed, photorealistic, white background",
        "RAW photo of a leopard lying on its side, fully extended lazy posture, clear belly and spot patterns, sharp focus, photorealistic, white background",
        "RAW high-angle shot of a leopard looking up, unique spatial perspective, sharp feline features, photorealistic, white background",
        "RAW photo of a leopard snarling, aggressive facial expression, detailed teeth and jaw structure, macro zoom, photorealistic, white background",
        "RAW photo of a leopard in a tense, muscular pre-leap posture, coiled energy, detailed muscle definitions, photorealistic, white background",
        "RAW tight close-up of a leopard's face and shoulders, striking green eyes, crisp whisker pores, ultra-high texture, photorealistic, white background",
        "RAW photo of a leopard turning its head sharply to the right, dramatic side neck silhouette, clear rosette details, photorealistic, white background",
        "RAW photo of a leopard stretching its back, arched spine posture, full body profile, highly detailed coat, photorealistic, white background",
        "RAW macro portrait of an apex leopard, direct intense eye contact, detailed fur grain, professional wildlife photography, photorealistic, white background",
    ],
    2: [
        "RAW photo of a sheep standing, Canon EOS R5, natural daylight, individual wool fibers visible, 8k uhd, photorealistic, white background",
        "RAW photo of a sheep grazing, DSLR, warm golden hour light, photorealistic, detailed fleece texture, white background",
        "RAW close-up portrait of a sheep, 85mm lens, shallow DOF, photorealistic wool and eye detail, white background",
        "RAW photo of a sheep walking, side profile, natural lighting, photorealistic, detailed fleece, white background",
        "RAW photo of a sheep with its head turned completely to the side, crisp profile view, thick fleece texture, photorealistic, white background",
        "RAW front-facing portrait of a sheep looking directly ahead, symmetrical composition, detailed face and ears, photorealistic, white background",
        "RAW low-angle photo of a sheep standing on an incline, majestic posture, sharp focus on thick wool, photorealistic, white background",
        "RAW photo of a sheep chewing grass, candid expression, detailed jaw and fleece structure, highly realistic, white background",
        "RAW photo of a sheep lying down, resting posture, compact rounded fleece silhouette, soft natural lighting, photorealistic, white background",
        "RAW macro shot of a sheep's face, extreme detail on the snout, eyes, and curly wool head-cap, photorealistic, white background",
        "RAW photo of a sheep running, dynamic movement pose, side view, natural motion framing, photorealistic, white background",
        "RAW photo of a sheep tilting its head downward, curious posture, sharp focus on the texture of the coat, photorealistic, white background",
        "RAW three-quarter body shot of a mature sheep, heavy detailed fleece, professional farm wildlife shot, photorealistic, white background",
        "RAW photo of a sheep shaking its head, dynamic action blur minimization, crisp facial features, thick woolly body, photorealistic, white background",
        "RAW photo of a fluffy sheep stepping forward, front action view, detailed hooves and lower leg wool, photorealistic, white background",
    ],
}

ANIMAL_NEG_PROMPT = (
    "blurry, watermark, text, multiple animals, cartoon, anime, illustration, "
    "painting, render, CGI, 3d, plastic, smooth, overexposed, oversaturated, "
    "bad anatomy, extra limbs, deformed, disfigured, ugly, low quality, "
    "low resolution, grainy noise, jpeg artifacts, logo, signature, bad eyes, deformed iris"
)

BACKGROUND_PROMPTS = [
    "RAW photo of a sunny forest clearing, photorealistic landscape, empty environment, no animals, 8k uhd, natural lighting, Canon EOS R5",
    "RAW photo of a grassy meadow at golden hour, photorealistic field, empty, no animals, warm volumetric light, 8k uhd, Nikon D850",
    "RAW photo of a rocky hillside with sparse dry trees, photorealistic mountain terrain, no animals, natural bright lighting, 8k uhd",
    "RAW photo of a snowy woodland path, silent winter forest landscape, photorealistic, no animals, soft diffuse overcast light",
    "RAW photo of an autumn forest floor covered with vibrant fallen leaves, macro ground texture focus, photorealistic, no animals",
    "RAW photo of a riverbank with smooth pebbles and patches of wild grass, clear water edge, photorealistic, no animals, natural bright daylight",
    "RAW photo of a wide countryside field at dusk, rolling hills background, photorealistic, no animals, cinematic golden hour light",
    "RAW photo of a sunny open savanna, dry yellow grass plain, photorealistic landscape, no animals, 8k uhd, sharp deep horizon",
    "RAW photo of a dense jungle interior, tropical foliage, photorealistic moss and ferns, no animals, natural dappled light filtering through trees",
    "RAW photo of a dry scrubland desert, arid environment with low bushes, photorealistic, no animals, harsh clear natural lighting",
]

BACKGROUND_NEG_PROMPT = "people, humans, animals, birds, insects, text, watermark, cartoon, painting, render, CGI, 3d"

CANVAS_W, CANVAS_H = 640, 640
FG_SCALE_MIN, FG_SCALE_MAX = 0.25, 0.55
NUM_AUGMENTED_IMAGES = 1000  # Outputs 1000 per class
SEED = 42
OUTPUT_DIR = Path("combined_dataset")

BASE_SIZE        = (512, 512)
HIRES_SIZE       = (640, 640)  # Native match for YOLO grid dimension
HIRES_DENOISE    = 0.45
BASE_STEPS       = 18          # Super fast convergence with DPM++ Karras
HIRES_STEPS      = 12
CFG_SCALE        = 5.5
BG_CFG_SCALE     = 5.5

# ─────────────────────────────── SCHEDULER ───────────────────────────────────

def make_dpm_karras_scheduler(config) -> DPMSolverMultistepScheduler:
    return DPMSolverMultistepScheduler.from_config(
        config, algorithm_type="dpmsolver++", solver_type="midpoint", use_karras_sigmas=True
    )

# ─────────────────────────────── SD PIPELINES ────────────────────────────────

def _apply_memory_optimisations(pipe, device: str) -> None:
    if device == "cpu": return
    try:
        pipe.enable_xformers_memory_efficient_attention()
        return
    except Exception: pass
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            pipe.unet.set_attn_processor(AttnProcessor2_0())
            return
        except Exception: pass
    pipe.enable_sequential_cpu_offload()

def load_pipelines() -> tuple[StableDiffusionPipeline, StableDiffusionImg2ImgPipeline]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"[SD] Loading {MODEL_ID} ...")

    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=dtype)
    txt2img = StableDiffusionPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=dtype, safety_checker=None).to(device)
    img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID, vae=vae, text_encoder=txt2img.text_encoder, tokenizer=txt2img.tokenizer,
        unet=txt2img.unet, safety_checker=None, torch_dtype=dtype
    ).to(device)

    txt2img.scheduler = make_dpm_karras_scheduler(txt2img.scheduler.config)
    img2img.scheduler = make_dpm_karras_scheduler(img2img.scheduler.config)

    _apply_memory_optimisations(txt2img, device)
    _apply_memory_optimisations(img2img, device)
    return txt2img, img2img

# ─────────────────────────────── GENERATION ──────────────────────────────────

def _make_generator(pipe, seed: int) -> torch.Generator:
    return torch.Generator(device=pipe.device).manual_seed(seed)

def generate_image_hires(txt2img, img2img, prompt, negative_prompt, base_size, hires_size, base_steps, hires_steps, cfg_scale, seed) -> Image.Image:
    gen1 = _make_generator(txt2img, seed)
    base_img = txt2img(
        prompt=prompt, negative_prompt=negative_prompt, width=base_size[0], height=base_size[1],
        num_inference_steps=base_steps, guidance_scale=cfg_scale, generator=gen1
    ).images[0]

    upscaled_img = base_img.resize(hires_size, Image.LANCZOS)
    gen2 = _make_generator(img2img, seed + 1)
    final_img = img2img(
        prompt=prompt, negative_prompt=negative_prompt, image=upscaled_img, strength=HIRES_DENOISE,
        num_inference_steps=hires_steps, guidance_scale=cfg_scale, generator=gen2
    ).images[0]
    return final_img

# ─────────────────────────────── REMBG MATTING FIX ───────────────────────────

REMBG_SESSION = new_session("isnet-general-use")

def remove_background(img: Image.Image) -> Image.Image:
    # Alpha Matting isolates hair/fur outlines completely
    return rembg_remove(
        img, session=REMBG_SESSION, alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10
    )

def crop_to_content(rgba: Image.Image) -> Image.Image:
    bbox = rgba.getbbox()
    return rgba.crop(bbox) if bbox else rgba

def resize_foreground(fg: Image.Image, canvas_w: int, canvas_h: int) -> Image.Image:
    scale    = random.uniform(FG_SCALE_MIN, FG_SCALE_MAX)
    target_w = int(canvas_w * scale)
    target_h = int(target_w * (fg.height / fg.width))
    return fg.resize((target_w, target_h), Image.LANCZOS)

# ─────────────────────────────── AUGMENTATION ────────────────────────────────

def build_fg_augmenter() -> A.Compose:
    # Hardened to mimic chaotic real-world Kaggle OpenImage distributions
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6),
        A.CoarseDropout(max_holes=12, max_height=32, max_width=32, fill_value=0, p=0.4),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.Rotate(limit=15, border_mode=0, p=0.5),
    ])

def augment_foreground(fg_rgba: Image.Image, augmenter: A.Compose, idx: int) -> Image.Image:
    rgb   = np.array(fg_rgba.convert("RGB"))
    alpha = np.array(fg_rgba.split()[-1])

    random.seed(SEED + idx)
    np.random.seed(SEED + idx)
    aug_rgb = augmenter(image=rgb)["image"]

    random.seed(SEED + idx)
    np.random.seed(SEED + idx)
    aug_alpha = augmenter(image=np.stack([alpha] * 3, axis=-1))["image"][:, :, 0]

    res = Image.fromarray(aug_rgb).convert("RGBA")
    r, g, b, _ = res.split()
    return Image.merge("RGBA", (r, g, b, Image.fromarray(aug_alpha)))

# ─────────────────────────────── COMPOSITE ───────────────────────────────────

def composite_and_label(bg: Image.Image, fg_rgba: Image.Image, canvas_w: int, canvas_h: int, idx: int, class_id: int) -> tuple:
    canvas = bg.copy().convert("RGB").resize((canvas_w, canvas_h), Image.LANCZOS)
    fg     = resize_foreground(fg_rgba, canvas_w, canvas_h)
    fg_w, fg_h = fg.size

    random.seed(SEED + idx * 13)
    x_offset = random.randint(0, max(canvas_w - fg_w, 0))
    y_offset  = random.randint(0, max(canvas_h - fg_h, 0))

    canvas.paste(fg, (x_offset, y_offset), mask=fg.split()[3])

    fg_alpha = np.array(fg.split()[3])
    rows = np.any(fg_alpha > 10, axis=1)
    cols = np.any(fg_alpha > 10, axis=0)

    if rows.any() and cols.any():
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]
        abs_x1, abs_y1 = x_offset + c_min, y_offset + r_min
        abs_x2, abs_y2 = x_offset + c_max, y_offset + r_max
    else:
        abs_x1, abs_y1 = x_offset, y_offset
        abs_x2, abs_y2 = x_offset + fg_w, y_offset + fg_h

    cx = ((abs_x1 + abs_x2) / 2) / canvas_w
    cy = ((abs_y1 + abs_y2) / 2) / canvas_h
    bw = (abs_x2 - abs_x1) / canvas_w
    bh = (abs_y2 - abs_y1) / canvas_h

    return canvas, f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

def write_data_yaml(output_dir: Path, classes: dict) -> None:
    names_str = "[" + ", ".join(f"'{n}'" for n in classes.values()) + "]"
    (output_dir / "data.yaml").write_text(
        f"path: {output_dir.resolve()}\ntrain: images/train\nval: images/val\nnc: {len(classes)}\nnames: {names_str}\n"
    )

# ─────────────────────────────── MAIN EXECUTION ──────────────────────────────

def main() -> None:
    img_dir = OUTPUT_DIR / "images" / "train"
    lbl_dir = OUTPUT_DIR / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    txt2img, img2img = load_pipelines()
    augmenter = build_fg_augmenter()

    print(f"\n[1/3] Caching {len(BACKGROUND_PROMPTS)} Background Landscapes Exactly Once...")
    bg_list = []
    for bi, prompt in enumerate(BACKGROUND_PROMPTS):
        print(f"      Processing background layout [{bi+1}/{len(BACKGROUND_PROMPTS)}] ...")
        bg = generate_image_hires(
            txt2img, img2img, prompt=prompt, negative_prompt=BACKGROUND_NEG_PROMPT,
            base_size=(512, 512), hires_size=HIRES_SIZE, base_steps=BASE_STEPS,
            hires_steps=HIRES_STEPS, cfg_scale=BG_CFG_SCALE, seed=SEED + 500 + bi
        )
        bg_list.append(bg)

    for class_id, class_name in CLASSES.items():
        prompts = ANIMAL_PROMPTS[class_id]
        print(f"\n[2/3] Class {class_id} '{class_name}' — Extracting shapes via {len(prompts)} distinct foreground templates...")

        fg_list = []
        for fi, prompt in enumerate(prompts):
            print(f"      [{fi+1}/{len(prompts)}] Isolating foreground silhouette ...")
            fg_raw = generate_image_hires(
                txt2img, img2img, prompt=prompt, negative_prompt=ANIMAL_NEG_PROMPT,
                base_size=BASE_SIZE, hires_size=HIRES_SIZE, base_steps=BASE_STEPS,
                hires_steps=HIRES_STEPS, cfg_scale=CFG_SCALE, seed=SEED + class_id * 100 + fi
            )
            fg_rgba = remove_background(fg_raw)
            fg_rgba = crop_to_content(fg_rgba)
            fg_list.append(fg_rgba)

        print(f"\n[3/3] Class {class_id} '{class_name}' — Compiling {NUM_AUGMENTED_IMAGES} domain-hardened training frames...")
        for i in range(NUM_AUGMENTED_IMAGES):
            # 80% (800) go to train, 20% (200) go to val
            split_name = "train" if i < 800 else "val"
            current_img_dir = OUTPUT_DIR / "images" / split_name
            current_lbl_dir = OUTPUT_DIR / "labels" / split_name
            
            # Ensure these directories exist
            current_img_dir.mkdir(parents=True, exist_ok=True)
            current_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            fg_rgba = fg_list[i % len(fg_list)]
            bg      = bg_list[random.randint(0, len(bg_list) - 1)]

            aug_fg  = augment_foreground(fg_rgba, augmenter, idx=class_id * 10000 + i)
            composite, label = composite_and_label(bg, aug_fg, CANVAS_W, CANVAS_H, idx=class_id * 10000 + i, class_id=class_id)

            img_name = f"sample_{class_id}_{i:04d}.jpg"
            composite.save(current_img_dir / img_name, quality=95)
            (current_lbl_dir / f"sample_{class_id}_{i:04d}.txt").write_text(label + "\n")

            if (i + 1) % 250 == 0 or i == 0:
                print(f"      Assembled [{i+1}/{NUM_AUGMENTED_IMAGES}] → {img_name}")

    write_data_yaml(OUTPUT_DIR, CLASSES)
    print(f"\n✅ Asset script execution successfully finished.")

if __name__ == "__main__":
    main()

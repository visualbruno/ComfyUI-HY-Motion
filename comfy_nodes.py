import os
import torch
import yaml
import numpy as np
from typing import Dict, List, Any, Optional, Union

import folder_paths

from .hymotion.utils.loaders import load_object
from .hymotion.pipeline.body_model import WoodenMesh
from .hymotion.pipeline.motion_diffusion import length_to_mask, randn_tensor

class HYMotionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["HY-Motion-1.0","HY-Motion-1.0-Lite"], {"default": "HY-Motion-1.0"}),
                #"precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("HYMOTION_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "HYMotion"

    def load_model(self, model_name, precision = "bf16"):
        model_path = os.path.join(folder_paths.models_dir,"HY-Motion","ckpts","tencent",model_name)
        cfg_path = os.path.join(model_path, "config.yml")
        ckpt_path = os.path.join(model_path, "latest.ckpt")
        
        if not os.path.exists(cfg_path):
            local_cfg = os.path.join(os.path.dirname(__file__), model_path, "config.yml")
            if os.path.exists(local_cfg):
                cfg_path = local_cfg
                ckpt_path = os.path.join(os.path.dirname(__file__), model_path, "latest.ckpt")
            else:
                raise FileNotFoundError(f"Config not found: {cfg_path}")
        
        with open(cfg_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        pipeline = load_object(
            config["train_pipeline"],
            config["train_pipeline_args"],
            network_module=config["network_module"],
            network_module_args=config["network_module_args"],
        )
        
        dtype = torch.float32
        if precision == "fp16":
            dtype = torch.float16
        elif precision == "bf16":
            dtype = torch.bfloat16
            
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            pipeline.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"Loaded HYMotion model from {ckpt_path}")
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}, using randomly initialized weights.")
            
        pipeline.to(dtype)
        pipeline.motion_transformer.eval()
        
        return (pipeline,)

class HYMotionTextEncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": ("STRING", {"default": "clip-vit-large-patch14"}),
                "qwen_name": ("STRING", {"default": "Qwen3-8B"}),
                "precision": (["fp16", "bf16", "fp32", "int8", "int4"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("HYMOTION_ENCODER",)
    FUNCTION = "load_encoder"
    CATEGORY = "HYMotion"

    def load_encoder(self, clip_name, qwen_name, precision):
        from .hymotion.network.text_encoders.text_encoder import HYTextModel
        from transformers import BitsAndBytesConfig
        import comfy.model_management as mm
        
        # In this project, HYTextModel expects layouts to be pre-configured or paths provided.
        # We can temporarily patch the layouts to use the user-provided paths.
        from .hymotion.network.text_encoders import text_encoder
        
        clip_path = os.path.join(folder_paths.models_dir,"HY-Motion","ckpts",clip_name)
        qwen_path = os.path.join(folder_paths.models_dir,"HY-Motion","ckpts",qwen_name)
        
        # Resolve absolute paths
        # if not os.path.isabs(clip_path):
            # clip_path = os.path.join(os.path.dirname(__file__), clip_path)
        # if not os.path.isabs(qwen_path):
            # qwen_path = os.path.join(os.path.dirname(__file__), qwen_path)

        # Update layouts in the module
        text_encoder.SENTENCE_EMB_LAYOUT["clipl"]["module_path"] = clip_path
        text_encoder.LLM_ENCODER_LAYOUT["qwen3"]["module_path"] = qwen_path
        
        llm_kwargs = {}              
        dtype = torch.float32
        if precision == "fp16":
            dtype = torch.float16
        elif precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "int8":
            llm_kwargs["load_in_8bit"] = True
            dtype = torch.float16 # Compute dtype
        elif precision == "int4":
            llm_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            dtype = torch.bfloat16 # Compute dtype
            
        llm_kwargs["torch_dtype"] = dtype                                 
        encoder = HYTextModel(llm_type="qwen3", sentence_emb_type="clipl", llm_kwargs=llm_kwargs)
        
        device = mm.get_torch_device()
        # Note: BitsAndBytes handles device placement during loading.
        # But we still need to move the other parts (CLIP) to the device.
        if precision not in ["int8", "int4"]:
            encoder.to(device)
            
        encoder.eval().requires_grad_(False)
        
        return (encoder,)

class HYMotionTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "encoder": ("HYMOTION_ENCODER",),
                "text": ("STRING", {"multiline": True, "default": "A person walks forward"}),
                "offload_after_encode": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("HYMOTION_CONDITIONING",)
    FUNCTION = "encode_text"
    CATEGORY = "HYMotion"

    def encode_text(self, encoder, text, offload_after_encode):
        import comfy.model_management as mm
        # encoder is an instance of HYTextModel
        # text can be a string or multiple lines
        prompt_list = [t.strip() for t in text.split("\n") if t.strip()]
        if not prompt_list:
            prompt_list = [""]
            
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        # Ensure it's on device
        encoder.to(device)
        
        vtxt_raw, ctxt_raw, ctxt_length = encoder.encode(prompt_list)
        
        if offload_after_encode:
            encoder.to(offload_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Use the forensic debug style for transparency
        print(f"HYMotion Custom Encoding Debug:")
        print(f"  CLIP Pooled Shape: {vtxt_raw.shape}, Min: {vtxt_raw.min():.4f}, Max: {vtxt_raw.max():.4f}")
        print(f"  Qwen Ctxt Shape: {ctxt_raw.shape}, Min: {ctxt_raw.min():.4f}, Max: {ctxt_raw.max():.4f}")
        print(f"  Qwen Ctxt Lengths: {ctxt_length.tolist()}")

        conditioning = {
            "text_vec_raw": vtxt_raw,
            "text_ctxt_raw": ctxt_raw,
            "text_ctxt_raw_length": ctxt_length,
            "text_ctxt_mask": None, # Mask is implicitly handled by length in the sampler
        }
        
        return (conditioning,)

class HYMotionSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HYMOTION_MODEL",),
                "conditioning": ("HYMOTION_CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 60.0, "step": 0.1}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 250}),
                "offload_after_sample": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("HYMOTION_MOTION",)
    FUNCTION = "sample"
    CATEGORY = "HYMotion"

    def sample(self, model, conditioning, seed, duration, cfg_scale, steps, offload_after_sample):
        from comfy.utils import ProgressBar
        import comfy.model_management as mm
        from .hymotion.utils.loaders import load_object
        from .hymotion.pipeline.motion_diffusion import length_to_mask, randn_tensor

        pipeline = model
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        pipeline.to(device)
        
        vtxt_cond = conditioning["text_vec_raw"].to(device)
        ctxt_cond = conditioning["text_ctxt_raw"].to(device)
        ctxt_length_cond = conditioning["text_ctxt_raw_length"].to(device)
        
        dtype = next(pipeline.motion_transformer.parameters()).dtype
        vtxt_cond = vtxt_cond.to(dtype)
        ctxt_cond = ctxt_cond.to(dtype)
        
        length = int(round(duration * 30))
        length = min(length, pipeline.train_frames)
        
        # Determine if we have multiple samples or just one
        batch_size = ctxt_cond.shape[0]
        
        do_classifier_free_guidance = cfg_scale > 1.0
        if do_classifier_free_guidance:
            # Fallback to internal null tokens for CFG
            # null_vtxt shape: [1, 1, D] -> expand to [batch_size, 1, D]
            null_vtxt = pipeline.null_vtxt_feat.to(device, dtype).expand(batch_size, -1, -1)
            # null_ctxt shape: [1, 1, D] -> expand to [batch_size, seq_len, D]
            null_ctxt = pipeline.null_ctxt_input.to(device, dtype).expand(batch_size, ctxt_cond.shape[1], -1)
            
            vtxt_input = torch.cat([null_vtxt, vtxt_cond], dim=0)
            ctxt_input = torch.cat([null_ctxt, ctxt_cond], dim=0)
            # Null token length is usually 1
            ctxt_length = torch.cat([torch.ones(batch_size, device=device, dtype=torch.long), ctxt_length_cond], dim=0)
        else:
            vtxt_input = vtxt_cond
            ctxt_input = ctxt_cond
            ctxt_length = ctxt_length_cond

        ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
        x_length = torch.LongTensor([length] * (batch_size * 2 if do_classifier_free_guidance else batch_size)).to(device)
        x_mask_temporal = length_to_mask(x_length, pipeline.train_frames)

        # Manual Euler Integration with Progress Bar
        dt = 1.0 / steps
        x = randn_tensor(
            (batch_size, pipeline.train_frames, pipeline._network_module_args["input_dim"]),
            generator=torch.Generator(device=device).manual_seed(seed),
            device=device,
            dtype=dtype
        )
        
        pbar = ProgressBar(steps)
        
        with torch.no_grad():
            for i in range(steps):
                t = torch.tensor([i * dt], device=device, dtype=dtype)
                
                x_in = torch.cat([x] * 2, dim=0) if do_classifier_free_guidance else x
                v = pipeline.motion_transformer(
                    x=x_in,
                    ctxt_input=ctxt_input,
                    vtxt_input=vtxt_input,
                    timesteps=t.expand(x_in.shape[0]),
                    x_mask_temporal=x_mask_temporal,
                    ctxt_mask_temporal=ctxt_mask_temporal,
                )
                
                if do_classifier_free_guidance:
                    v_uncond, v_cond = v.chunk(2, dim=0)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                
                x = x + v * dt
                pbar.update(1)
        
        # Extract the requested length
        sampled = x[:, :length, ...].clone()
        
        # Final decode
        motion_dict = pipeline.decode_motion_from_latent(sampled, should_apply_smooothing=True)
        
        if offload_after_sample:
            pipeline.to(offload_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return (motion_dict,)

class HYMotionVisualizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("HYMOTION_MOTION",),
                "output_name": ("STRING", {"default": "hymotion_output"}),
                "export_fbx": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("fbx_path","fbx_name",)
    FUNCTION = "visualize"
    CATEGORY = "HYMotion"
    OUTPUT_NODE = True

    def visualize(self, motion, output_name, export_fbx):
        from .hymotion.utils.visualize_mesh_web import save_visualization_data, generate_static_html_content
        import time
        
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        output_dir = folder_paths.output_directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Use a sanitized file name for stability
        safe_name = output_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        save_data, base_filename = save_visualization_data(
            output=motion,
            text=output_name,
            rewritten_text=output_name,
            timestamp=ts,
            output_dir=output_dir,
            output_filename=safe_name,
        )
        
        # In this utility, 'folder_name' in generate_static_html_content is actually 
        # relative to the PROJECT ROOT because it calls get_output_dir(folder_name).
        # To fix this, we can just pass the path relative to the root or the absolute path.
        # However, the utility is rigid, so we'll just use the base_filename and the 
        # directory we already have.
        
        # html_content = generate_static_html_content(
            # folder_name=output_dir, # This utility will handle absolute paths if passed
            # file_name=base_filename,
            # hide_captions=False,
        # )
        
        # html_path = os.path.join(output_dir, base_filename + ".html")
        # with open(html_path, "w", encoding="utf-8") as f:
            # f.write(html_content)
        
        fbx_name = ""
        fbx_path = ""
        if export_fbx:
            try:
                from .hymotion.utils.smplh2woodfbx import SMPLH2WoodFBX
                converter = SMPLH2WoodFBX()
                smpl_data_list = save_data["smpl_data"]
                smpl_data = smpl_data_list[0]
                fbx_name = f"{safe_name}_{ts}.fbx"
                fbx_path = os.path.abspath(os.path.join(output_dir, fbx_name))
                success = converter.convert_npz_to_fbx(smpl_data, fbx_path)
                if not success:
                    print(f"Warning: FBX conversion failed for {fbx_path}")
                    fbx_path = "FBX conversion error (check logs)"
            except ImportError:
                print("Warning: FBX SDK not found. Skipping FBX export. Please install 'fbx' (Autodesk FBX Python SDK).")
                fbx_path = "FBX SDK missing"
            except Exception as e:
                print(f"Warning: FBX export failed with error: {e}")
                fbx_path = f"FBX export error: {e}"
        
        return (fbx_path, fbx_name,)

NODE_CLASS_MAPPINGS = {
    "HYMotionModelLoader": HYMotionModelLoader,
    "HYMotionTextEncoderLoader": HYMotionTextEncoderLoader,
    "HYMotionTextEncode": HYMotionTextEncode,
    "HYMotionSampler": HYMotionSampler,
    "HYMotionVisualizer": HYMotionVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYMotionModelLoader": "HY-Motion Model Loader",
    "HYMotionTextEncoderLoader": "HY-Motion Text Encoder Loader",
    "HYMotionTextEncode": "HY-Motion Text Encode",
    "HYMotionSampler": "HY-Motion Sampler",
    "HYMotionVisualizer": "HY-Motion Visualizer",
}

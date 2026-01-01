# 🌀 ComfyUI Wrapper for [https://github.com/Tencent-Hunyuan/HY-Motion-1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0)

---

<img width="1151" height="374" alt="{D0EC5C97-9B7D-4035-98A9-E33326E7B305}" src="https://github.com/user-attachments/assets/d0757491-5106-413c-9dc6-842cfeee31bd" />

---
## REQUIREMENTS ##

You need to download first these models:
- clip-vit-large-patch14: [https://huggingface.co/openai/clip-vit-large-patch14/tree/main](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)
- Qwen3-8B: [https://huggingface.co/Qwen/Qwen3-8B/tree/main](https://huggingface.co/Qwen/Qwen3-8B/tree/main)
- HY-Motion-1.0: [https://huggingface.co/tencent/HY-Motion-1.0/tree/main](https://huggingface.co/tencent/HY-Motion-1.0/tree/main)

And save them in the folder "ComfyUI/models/HY-Motion/ckpts/"

So you must have a folder structure like this:

```
HY-Motion/
  |──ckpts/
      ├── tencent/
      │   ├── HY-Motion-1.0/          # Contains config.yml and latest.ckpt
      │   └── HY-Motion-1.0-Lite/     # Optional
      ├── clip-vit-large-patch14/     # CLIP weights
      ├── Qwen3-8B/                   # Qwen text encoder weights
```

---

## ⚙️ Installation Guide

> Tested on **Windows 11** with **Python 3.11** and **Torch = 2.7.0 + cu128**.

### 1. Install Requirements.txt file

#### For a standard python environment:

```bash
python -m pip install -r ComfyUI/custom_nodes/ComfyUI-HY-Motion/requirements.txt
```

---

#### For ComfyUI Portable:

```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-HY-Motion\requirements.txt
```

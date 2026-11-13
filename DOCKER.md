# Docker: POTracker / POSOLM live UI

These Docker files are for the Flask/Transformers project at
https://github.com/pdhung3012/POSOLM-KD-UI . The model weights are **not**
packaged in the image; Docker mounts local folders read-only.

## 1. Prerequisites

- Docker Engine and Docker Compose plugin (`docker compose version`).
- For NVIDIA GPUs: working driver (`nvidia-smi`) and NVIDIA Container Toolkit configured for Docker.
- Free disk space for PyTorch base image, model downloads and fine-tuned weights.

## 2. Set model paths

```bash
cp .env.example .env
nano .env
```

`HOST_MODELS_DIR` and `HOST_CHECKPOINTS_DIR` are absolute paths **on the host**.
They become `/models` and `/checkpoints` **inside the container**.

Example:
- host `/home/hungphd/models/Qwen2.5-3B` => container `/models/Qwen2.5-3B`
- host `/home/hungphd/checkpoints/POSOLM-KD` => container `/checkpoints/POSOLM-KD`

`KD_MODEL_PATH` is the base model directory; `KD_WEIGHTS_PATH` is the
fine-tuned weight/adapter directory. `auto` detects LoRA by `adapter_config.json`.
For a full HF checkpoint, the weight directory needs model checkpoint files
saved by `save_pretrained()`. For raw `.pt` state dicts, convert first.

If using Hugging Face hub models, set e.g. `KD_MODEL_PATH=Qwen/Qwen2.5-3B`.
The named `hf-cache` Docker volume stores downloads between restarts.

You can alternatively edit `config/models.json`, which is read from a bind mount.
Environment variables from `.env` take precedence over JSON fields when nonempty.

## 3. Start with an NVIDIA GPU

```bash
docker compose up --build -d
docker compose logs -f posolm-ui
```

Browse **http://localhost:5000** (or the configured `HOST_PORT`).
Test model configuration:

```bash
curl http://localhost:5000/api/models
```

Note that model weights load lazily, on the first generation request; a healthy
HTTP service does not guarantee that a checkpoint loads or fits in VRAM.

## 4. CPU-only fallback

```bash
docker compose -f compose.cpu.yaml up --build -d
docker compose -f compose.cpu.yaml logs -f posolm-ui
```

CPU inference on 3B/7B models can be very slow and memory intensive.

## 5. Stop

```bash
docker compose down
# For CPU: docker compose -f compose.cpu.yaml down
```

Keep the `hf-cache` volume unless you want to redownload Hugging Face files.

## Troubleshooting

- **CUDA unavailable**: check `nvidia-smi`, `docker info` and NVIDIA Container Toolkit;
  `docker compose exec posolm-ui python -c 'import torch; print(torch.cuda.is_available())'`.
- **File not found**: model paths inside Docker start with `/models` or `/checkpoints`.
  Check `.env` and host folder permissions.
- **Out of memory**: free GPU memory or use a smaller/quantized checkpoint. The current
  model loader uses standard HF precision and does not implement 4-bit quantization.
- **Remote access**: port 5000 is published on host interfaces. Do not expose publicly
  without authentication, HTTPS, and request limits. For local-only access, change the
  ports mapping to `127.0.0.1:${HOST_PORT:-5000}:5000`.

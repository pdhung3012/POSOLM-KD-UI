# POTracker / POSOLM — live Python inference UI

This updates the supplied HTML interface to generate actual model completions using a local Python Flask server and Hugging Face Transformers. Prewritten `.txt` outputs are no longer used for generation.

## Setup

```bash
cd POSOLM-KD-UI-main
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python server.py
```

Open **http://127.0.0.1:5000** in a browser. The first generation downloads `Qwen/Qwen2.5-3B` from Hugging Face, then runs inference. Internet access is needed for the initial model download. A GPU is strongly recommended; CPU generation can be very slow. For CUDA, install an appropriate CUDA-enabled PyTorch wheel per PyTorch's instructions.

## Model selection

- **Qwen2.5-3B without KD** defaults to `Qwen/Qwen2.5-3B` (the exact model linked in the request). This is a *base* completion model; for better instruction-following, set `BASE_MODEL_ID=Qwen/Qwen2.5-3B-Instruct` before launching.
- **Qwen2.5-3B with KD** requires your actual checkpoint: `export KD_MODEL_ID=/path/to/your/distilled-checkpoint`.
- **POTracker Qwen2.5-7B** requires your actual checkpoint: `export POTRACKER_MODEL_ID=/path/to/your/potracker-checkpoint`.

The two fine-tuned selections intentionally display a configuration error until their trained checkpoints are set. Setting them to the public pretrained models does not provide POTracker/KD behavior. HF repository IDs also work.

Example:

```bash
export BASE_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
export KD_MODEL_ID=/models/posolm-kd-3b
export POTRACKER_MODEL_ID=/models/potracker-7b
python server.py
```

## HTTP API

`POST /api/generate` accepts JSON:

```json
{
  "model": "POSOLM-Qwen2.5-3B-no-KD",
  "input": "Outage in Ames, Iowa; 200 customers; start 08:30; cause unknown.",
  "format": "XML",
  "temperature": 0.2,
  "top_p": 0.9,
  "max_new_tokens": 512,
  "verification_passes": 1,
  "modules": ["Schema-aware Correction", "Field Reconstruction"]
}
```

Example command:

```bash
curl -X POST http://127.0.0.1:5000/api/generate \
 -H 'Content-Type: application/json' \
 -d '{"model":"POSOLM-Qwen2.5-3B-no-KD","input":"Outage in Ames at 08:30, 200 affected","format":"XML","verification_passes":1}'
```

`GET /api/models` shows which model checkpoints are configured. One model is cached at a time; the server serializes inference requests. Verification passes >1 run repeated refinement generations, so latency increases. The denoising module selections are *prompt instructions*, not independently implemented classifiers or schema validators. The confidence threshold slider is currently illustrative and does not filter generation. Generated XML is not guaranteed schema-valid and should be validated before production use.

**Metric:** The UI compares output against `data/label.txt` only when the input exactly matches the bundled `data/input.txt`, reporting token-frequency cosine similarity (not actual accuracy or XML schema correctness). Other inputs show N/A. Gemini ground-truth view is still a static local reference, not a Gemini API call.

**Security:** The development server is bound to localhost by default. Do not expose it to the Internet without authentication, HTTPS, request controls and proper production hosting.

## Fine-tuned models stored in separate model and weight folders

Edit **`config/models.json`** to configure the three model choices in the existing HTML UI. Paths can be absolute paths on the Python server or Hugging Face repository IDs. The model directories are **not** browsed or read by the browser; Python accesses them locally.

Example (replace these sample directories with your real ones):

```json
{
  "models": {
    "POSOLM-KD-Qwen2.5-3B": {
      "model_path": "/models/Qwen2.5-3B", "weights_path": "/checkpoints/POSOLM-KD", "checkpoint_type": "auto"
    }
  }
}
```

Each model additionally accepts optional `"tokenizer_path": "/path/to/tokenizer"` if its tokenizer differs from the base directory.

**Full fine-tuning:** use `"checkpoint_type": "full"` (or `"auto"`) when `weights_path` contains HF `save_pretrained()` output, e.g. `model.safetensors` or multiple `model-00001-of-....safetensors` shards and their index. The architecture/configuration is loaded from `model_path`; weights are loaded from `weights_path`. The tokenizer defaults to `model_path`. If `weights_path` is blank, all files are loaded from `model_path`. If a full checkpoint contains its own `config.json` and tokenizer files, you can set both paths to that checkpoint folder.

**LoRA / PEFT:** use `"checkpoint_type": "adapter"` (or `"auto"` if `adapter_config.json` exists in `weights_path`). The base model loads from `model_path`, then PEFT loads `adapter_model.safetensors` / `adapter_model.bin` from `weights_path`. Ensure that the base model is the **same checkpoint family used for training**. `adapter_config.json` and adapter weights must be in the adapter folder itself.

**Important:** Raw training `.pt` / `.pth` `state_dict` files without Hugging Face checkpoint metadata, or optimizer-only training checkpoints, are not automatically supported. Export the full fine-tuned model with `model.save_pretrained(folder)` or an adapter with `peft_model.save_pretrained(folder)` first. The Python process needs permission to read these directories. Model switching replaces the cached model and may temporarily require additional RAM/VRAM.

### Optional environment-variable overrides

Variables named `BASE_MODEL_PATH`, `BASE_WEIGHTS_PATH`, `KD_MODEL_PATH`, `KD_WEIGHTS_PATH`, `POTRACKER_MODEL_PATH`, `POTRACKER_WEIGHTS_PATH` override the corresponding JSON fields. `*_TOKENIZER_PATH` and `*_CHECKPOINT_TYPE` are also available. Legacy `*_MODEL_ID` variables continue to override `model_path`. `MODEL_CONFIG_FILE` can point to another configuration JSON document.

Example:

```bash
export KD_MODEL_PATH=/data/models/Qwen2.5-3B
export KD_WEIGHTS_PATH=/data/experiments/posolm-kd/final
export KD_CHECKPOINT_TYPE=adapter
python server.py
```

### Verify configuration

```bash
curl http://127.0.0.1:5000/api/models
```

This reports the active model locations and detected checkpoint type. The first generation loads weights into memory; subsequent generations for the same model reuse them. For LoRA, the `peft` dependency in `requirements.txt` must be installed.

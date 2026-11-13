"""Load local Hugging Face fine-tuned checkpoints and PEFT adapters."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = Path(os.getenv('MODEL_CONFIG_FILE', str(ROOT / 'config' / 'models.json')))

UI_MODELS = (
    'POSOLM-Qwen2.5-3B-no-KD',
    'POSOLM-KD-Qwen2.5-3B',
    'POTracker-Qwen2.5-7B-Instruct',
)

ENV_PREFIXES = {
    'POSOLM-Qwen2.5-3B-no-KD': 'BASE',
    'POSOLM-KD-Qwen2.5-3B': 'KD',
    'POTracker-Qwen2.5-7B-Instruct': 'POTRACKER',
}


def read_config():
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(f'Model configuration missing: {CONFIG_FILE}')
    with CONFIG_FILE.open(encoding='utf-8') as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not isinstance(raw.get('models'), dict):
        raise ValueError('Model configuration must contain a "models" object.')
    return raw['models']


def resolve_spec(key):
    if key not in UI_MODELS:
        raise ValueError('Unknown UI model key.')
    config = read_config().get(key, {})
    if not isinstance(config, dict):
        raise ValueError(f'Invalid configuration for {key}.')
    config = dict(config)
    prefix = ENV_PREFIXES[key]
    # Environment variables override the JSON configuration when provided.
    env_names = {
        'model_path': (f'{prefix}_MODEL_PATH', f'{prefix}_MODEL_ID'),
        'weights_path': (f'{prefix}_WEIGHTS_PATH',),
        'tokenizer_path': (f'{prefix}_TOKENIZER_PATH',),
        'checkpoint_type': (f'{prefix}_CHECKPOINT_TYPE',),
    }
    for field, names in env_names.items():
        for name in names:
            if os.getenv(name):
                config[field] = os.environ[name]
                break
    spec = {name: str(config.get(name) or '').strip() for name in
            ('model_path', 'weights_path', 'tokenizer_path', 'checkpoint_type')}
    spec['checkpoint_type'] = spec['checkpoint_type'].lower() or 'auto'
    if spec['checkpoint_type'] not in ('auto', 'full', 'adapter'):
        raise ValueError('checkpoint_type must be auto, full, or adapter')
    if not (spec['model_path'] or spec['weights_path']):
        return spec
    if not spec['model_path']:
        # A full fine-tuned checkpoint can be complete in weights_path.
        spec['model_path'] = spec['weights_path']
    return spec


def is_local_reference(value):
    return value.startswith(('/', './', '../', '~')) or Path(value).is_dir()


def assert_exists(reference, label):
    if reference and is_local_reference(reference) and not Path(reference).expanduser().is_dir():
        raise FileNotFoundError(f'{label} directory does not exist: {reference}')


def detect_type(spec):
    mode = spec['checkpoint_type']
    if mode != 'auto':
        return mode
    target = Path(spec['weights_path'] or spec['model_path']).expanduser()
    return 'adapter' if (target / 'adapter_config.json').is_file() else 'full'


def load_checkpoint(spec):
    """Return (tokenizer, causal_lm) for full HF save_pretrained or PEFT adapter."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    base = spec['model_path']
    weights = spec['weights_path'] or base
    mode = detect_type(spec)
    for label, reference in [('model_path', base), ('weights_path', weights),
                             ('tokenizer_path', spec['tokenizer_path'])]:
        assert_exists(reference, label)

    tokenizer_ref = spec['tokenizer_path'] or base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, trust_remote_code=False)
    # Use model config and architecture from base, but load learned weights from weights_path.
    kwargs = {'trust_remote_code': False, 'torch_dtype': 'auto',
              'device_map': 'auto' if torch.cuda.is_available() else 'cpu'}
    if mode == 'adapter':
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
        model = PeftModel.from_pretrained(model, weights, is_trainable=False)
    else:
        # weights_path might contain only model.safetensors and no config.json;
        # supply architecture config explicitly from the separate model_path.
        model_config = AutoConfig.from_pretrained(base, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(weights, config=model_config, **kwargs)
    model.eval()
    return tokenizer, model, mode

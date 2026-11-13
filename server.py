"""Local Hugging Face inference backend for the supplied outage-report UI."""
import os
import re
import threading
from collections import Counter
from math import sqrt
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from model_loader import UI_MODELS, resolve_spec, load_checkpoint, detect_type

ROOT = Path(__file__).resolve().parent
UI = ROOT / 'ui'
app = Flask(__name__)
app.json.ensure_ascii = False

# Keep one loaded fine-tuned model in memory to limit GPU memory pressure.
lock = threading.Lock()
loaded = {'spec': None, 'tokenizer': None, 'model': None, 'type': None}


def get_model(spec):
    if loaded['spec'] == spec and loaded['model'] is not None:
        return loaded['tokenizer'], loaded['model'], loaded['type']
    import gc
    import torch
    loaded.update(spec=None, tokenizer=None, model=None, type=None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer, model, mode = load_checkpoint(spec)
    loaded.update(spec=spec.copy(), tokenizer=tokenizer, model=model, type=mode)
    return tokenizer, model, mode


def prompt_for(tokenizer, system_prompt, user_text):
    if getattr(tokenizer, 'chat_template', None):
        return tokenizer.apply_chat_template(
            [{'role': 'system', 'content': system_prompt},
             {'role': 'user', 'content': user_text}],
            tokenize=False, add_generation_prompt=True)
    # The user-linked Qwen/Qwen2.5-3B is a base completion model, not instruction tuned.
    return f'{system_prompt}\n\nUser request:\n{user_text}\n\nAnswer:\n'


def generate(tokenizer, model, prompt, temperature, top_p, max_new_tokens):
    import torch
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=4096)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    sampling = temperature > 0
    opts = {'max_new_tokens': max_new_tokens, 'do_sample': sampling,
            'pad_token_id': tokenizer.eos_token_id}
    if sampling:
        opts.update(temperature=temperature, top_p=top_p)
    with torch.inference_mode():
        result = model.generate(**inputs, **opts)
    new_tokens = result[0][inputs['input_ids'].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def cosine_similarity(a, b):
    x = Counter(re.findall(r'[a-z0-9]+', a.lower()))
    y = Counter(re.findall(r'[a-z0-9]+', b.lower()))
    if not x or not y:
        return 0.0
    return sum(n * y.get(k, 0) for k, n in x.items()) / sqrt(
        sum(n*n for n in x.values()) * sum(n*n for n in y.values()))


@app.get('/')
def home():
    return send_from_directory(UI, 'extended-ui.html')


@app.get('/ui/<path:name>')
def ui_file(name):
    return send_from_directory(UI, name)


@app.get('/data/<path:name>')
def data_file(name):
    return send_from_directory(UI / 'data', name)


@app.get('/api/models')
def models():
    info = {}
    for key in UI_MODELS:
        try:
            spec = resolve_spec(key)
            info[key] = {
                'configured': bool(spec['model_path']),
                'model_path': spec['model_path'] or None,
                'weights_path': spec['weights_path'] or None,
                'checkpoint_type': detect_type(spec) if spec['model_path'] else None,
            }
        except Exception as exc:
            info[key] = {'configured': False, 'error': str(exc)}
    return jsonify(models=info)


@app.post('/api/generate')
def generate_endpoint():
    data = request.get_json(silent=True) or {}
    model_key = data.get('model')
    if model_key not in UI_MODELS:
        return jsonify(error='Unknown model selection.'), 400
    try:
        spec = resolve_spec(model_key)
    except (OSError, ValueError) as exc:
        return jsonify(error=f'Invalid model configuration: {exc}'), 400
    if not spec['model_path']:
        return jsonify(error=f'{model_key} is not configured. Edit config/models.json with model_path and weights_path.'), 400
    source = str(data.get('input', '')).strip()
    if not source or len(source) > 20000:
        return jsonify(error='Input must contain between 1 and 20,000 characters.'), 400
    try:
        temperature = float(data.get('temperature', 0.2))
        top_p = float(data.get('top_p', 0.9))
        max_tokens = int(data.get('max_new_tokens', 1024))
        passes = int(data.get('verification_passes', 1))
        fmt = str(data.get('format', 'XML')).upper()
    except (TypeError, ValueError):
        return jsonify(error='Invalid generation parameters.'), 400
    if not (0 <= temperature <= 1 and 0 < top_p <= 1 and 1 <= max_tokens <= 2048
            and 1 <= passes <= 5 and fmt in ('JSON', 'XML')):
        return jsonify(error='Generation parameters out of range.'), 400
    modules = data.get('modules', [])
    if not isinstance(modules, list):
        return jsonify(error='modules must be a list.'), 400
    modules = [str(m)[:100] for m in modules[:12]]
    instructions = (
        f'Convert the input power outage report into valid {fmt}. Preserve all verifiable facts. '
        'Never invent values; mark ambiguous or missing information as unknown/unverified. '
        'When producing XML, aim for IEC 61968-3-style PubOutages, with namespace '
        'http://iec.ch/TC57/2014/PubOutages# where appropriate. '
        'Return only the output document, without Markdown fences. '
        'Apply these priorities when relevant: ' + ', '.join(modules) + '.'
    )
    try:
        with lock:
            tokenizer, model, checkpoint_type = get_model(spec)
            result = generate(tokenizer, model, prompt_for(tokenizer, instructions, source),
                              temperature, top_p, max_tokens)
            for _ in range(passes - 1):
                refinement = (f'Revise this {fmt} document for internal consistency and syntax. '
                              'Preserve grounded facts, do not invent details, output only revised content.\n\n'
                              f'Original input:\n{source}\n\nDraft:\n{result}')
                result = generate(tokenizer, model,
                                  prompt_for(tokenizer, instructions, refinement),
                                  temperature, top_p, max_tokens)
        # Only report a similarity score for the exact sample with a matching label.
        example = (UI / 'data' / 'input.txt').read_text(encoding='utf-8').strip()
        score = None
        if source == example:
            label = (UI / 'data' / 'label.txt').read_text(encoding='utf-8')
            score = round(cosine_similarity(result, label) * 100, 1)
        return jsonify(output=result, similarity=score, checkpoint=spec['weights_path'] or spec['model_path'], checkpoint_type=checkpoint_type,
                       note='Similarity is a token-cosine proxy for the bundled example only, not accuracy.')
    except Exception as exc:
        app.logger.exception('Model inference failed')
        return jsonify(error=f'Inference failed: {type(exc).__name__}: {exc}'), 500


if __name__ == '__main__':
    app.run(host=os.getenv('HOST', '127.0.0.1'), port=int(os.getenv('PORT', '5000')),
            debug=False, threaded=True)

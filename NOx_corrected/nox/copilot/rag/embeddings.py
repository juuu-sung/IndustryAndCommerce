"""Pinned multilingual E5 inference, using safe local weights and both prefixes."""
from pathlib import Path
import threading
import numpy as np

from .common import MODEL_PATH, MODEL_REVISION, RUNTIME, RagError, asset_bytes, dependency, read_json
from ..registry import _digest, _sha


class E5Encoder:
    def __init__(self, model_path=MODEL_PATH, *, threads=2):
        path = Path(model_path).resolve()
        receipt = read_json(RUNTIME / 'asset_receipt.json')
        if receipt.get('model') != 'intfloat/multilingual-e5-small' or receipt.get('revision') != MODEL_REVISION:
            raise RagError('embedding_version_conflict', 'Pinned embedding revision differs')
        files = receipt.get('model_files', {})
        if not {'config.json', 'model.safetensors', 'tokenizer.json'}.issubset(files):
            raise RagError('embedding_version_conflict', 'Required safe embedding files are absent')
        for name, expected in files.items():
            if Path(name).name != name or _sha(asset_bytes(path / name)) != expected:
                raise RagError('embedding_version_conflict', 'Embedding file checksum differs')
        self.torch = dependency('torch')
        self.torch.set_num_threads(threads)
        transformers = dependency('transformers')
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                                        trust_remote_code=False, use_fast=True)
            self.model = transformers.AutoModel.from_pretrained(path, local_files_only=True,
                                                                trust_remote_code=False, use_safetensors=True).eval()
        except (ImportError, OSError, RuntimeError) as error:
            raise RagError('embedding_unavailable', 'Pinned embedding model could not be loaded', 503) from error
        self.lock = threading.Lock()
        self.descriptor = {'kind': 'pretrained_multilingual_e5', 'model_id': receipt['model'],
                           'revision': MODEL_REVISION, 'dimension': 384, 'maximum_tokens': 512,
                           'query_prefix': 'query: ', 'passage_prefix': 'passage: ',
                           'pooling': 'attention_mask_mean_then_l2', 'weights_sha256': _digest(files),
                           'device': 'cpu', 'training_performed': False}

    def offsets(self, text):
        return self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                              truncation=False, verbose=False)['offset_mapping']

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def encode(self, texts, *, kind, batch_size=8):
        if kind not in ('query', 'passage'):
            raise RagError('invalid_embedding_kind', 'Embedding kind must be query or passage', 400)
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        prepared = [kind + ': ' + text for text in texts]
        if any(self.count(text) > 512 for text in prepared):
            raise RagError('embedding_input_too_long', 'Embedding input exceeds the pinned token limit', 400)
        outputs = []
        with self.lock, self.torch.inference_mode():
            for start in range(0, len(prepared), batch_size):
                tokens = self.tokenizer(prepared[start:start + batch_size], padding=True,
                                        truncation=False, return_tensors='pt')
                result = self.model(**tokens).last_hidden_state
                masked = result.masked_fill(~tokens['attention_mask'][..., None].bool(), 0.)
                mean = masked.sum(dim=1) / tokens['attention_mask'].sum(dim=1)[..., None]
                normalized = self.torch.nn.functional.normalize(mean, p=2, dim=1)
                outputs.append(normalized.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

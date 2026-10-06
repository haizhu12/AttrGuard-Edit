"""Explicit, tensor-preserving migration of the reviewed Full research format."""
import copy
import hashlib
import json
from pathlib import Path

import torch

from .adapters import _same_metadata
from .runtime import _validate_adapter


def convert_full_research_state(state, config):
    """Validate an exact Full architecture; never promote an ablation by relabeling."""
    if (type(state) is not dict or set(state) != {'schema', 'arm', 'adapter'}
            or state['schema'] != 'controlled_source_ablation_checkpoint_v1'
            or state['arm'] != 'full'):
        raise ValueError('Only the explicit Full research checkpoint is convertible')
    inner = state['adapter']
    if (type(inner) is not dict or set(inner) != {'metadata', 'adapters'}
            or type(inner['metadata']) is not dict):
        raise ValueError('Invalid research adapter envelope')
    metadata = copy.deepcopy(inner['metadata'])
    if (metadata.get('schema') != 'controlled_source_ablation_adapter_v1'
            or metadata.get('arm') != 'full'
            or metadata.get('use_source') is not True
            or metadata.get('use_roles') is not True
            or not _same_metadata(metadata.get('injection_sites'), metadata.get('block_indices'))):
        raise ValueError('Research metadata does not describe the Full architecture')
    del metadata['arm'], metadata['injection_sites']
    metadata['schema'] = 'boogu_source_preservation_v1'
    portable = {'metadata': metadata, 'adapters': inner['adapters']}
    _validate_adapter(portable, config)
    portable['adapters'] = {
        name: value.detach().cpu().clone() for name, value in inner['adapters'].items()
    }
    return portable


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def export_full_research_adapter(source, destination, config):
    """Write new weights and provenance, with no overwrite or implicit selection."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    sidecar = destination.with_suffix(destination.suffix + '.json')
    if destination.suffix.lower() != '.pt':
        raise ValueError('Destination must end in .pt')
    if destination.exists() or sidecar.exists() or source in (destination, sidecar):
        raise FileExistsError('Choose a fresh adapter and provenance destination')
    source_digest = _sha256(source)
    state = torch.load(source, map_location='cpu', weights_only=True)
    portable = convert_full_research_state(state, config)
    if _sha256(source) != source_digest:
        raise ValueError('Research checkpoint changed during export')
    destination.parent.mkdir(parents=True, exist_ok=True)
    owned = []
    try:
        with destination.open('xb') as stream:
            owned.append(destination)
            torch.save(portable, stream)
        saved = torch.load(destination, map_location='cpu', weights_only=True)
        _validate_adapter(saved, config)
        if any(not torch.equal(value, saved['adapters'][name])
               for name, value in state['adapter']['adapters'].items()):
            raise ValueError('Exported tensor values differ from research weights')
        report = {
            'schema': 'boogu_full_adapter_export_v1', 'code_version': '0.2.1',
            'source_file': source.name, 'source_sha256': source_digest,
            'adapter_file': destination.name, 'adapter_sha256': _sha256(destination),
            'tensor_values_unchanged': True,
            'parameter_count': sum(t.numel() for t in saved['adapters'].values()),
            'metadata': saved['metadata'],
            'warning': 'Format migration only; no retraining or new quality claim.',
        }
        with sidecar.open('x', encoding='utf-8') as stream:
            owned.append(sidecar)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        return report
    except BaseException:
        # Only files exclusively created by this call can be removed.
        for path in reversed(owned):
            path.unlink(missing_ok=True)
        raise

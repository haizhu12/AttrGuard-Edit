"""Portable export must preserve the exact Full tensors and reject ablations."""
import copy
import importlib
import importlib.util
import json

import pytest
import torch


def checkpoint_module():
    assert importlib.util.find_spec('boogu.preservation.checkpoints') is not None
    return importlib.import_module('boogu.preservation.checkpoints')


def research_state():
    config = dict(num_layers=2, num_double_stream_layers=1, hidden_size=8,
                  in_channels=2, patch_size=2)
    metadata = dict(schema='controlled_source_ablation_adapter_v1', arm='full',
                    block_indices=[0], injection_sites=[0], rank=2, hidden_size=8,
                    in_channels=2, patch_size=2, use_source=True, use_roles=True,
                    attributes=['color','structure','texture','glyph','pixel'],
                    role_order=['keep','modify','unknown'])
    tensors = {}
    for name, shape in dict(W_h=(2,8),W_s=(2,8),W_r=(2,15),W_t=(2,1),W_out=(8,2)).items():
        tensors[f'banks.0.{name}.weight'] = torch.arange(shape[0]*shape[1]).reshape(shape).float()
        tensors[f'banks.0.{name}.bias'] = torch.ones(shape[0])
    return dict(schema='controlled_source_ablation_checkpoint_v1',arm='full',
                adapter=dict(metadata=metadata,adapters=tensors)), config


def test_export_full_is_tensor_exact_and_does_not_mutate_source():
    checkpoints = checkpoint_module()
    from boogu.preservation.runtime import _validate_adapter
    state, config = research_state()
    before = copy.deepcopy(state['adapter']['metadata'])
    result = checkpoints.convert_full_research_state(state, config)
    _validate_adapter(result, config)
    assert state['adapter']['metadata'] == before
    for name, value in state['adapter']['adapters'].items():
        assert torch.equal(value, result['adapters'][name])
        assert value.data_ptr() != result['adapters'][name].data_ptr()


@pytest.mark.parametrize('bad',['arm','schema','sites','roles','nan','dtype','extra'])
def test_export_rejects_wrong_or_corrupt_research_state(bad):
    checkpoints = checkpoint_module()
    state, config = research_state()
    if bad == 'arm': state['arm'] = 'no_role_input'
    if bad == 'schema': state['adapter']['metadata']['schema'] = 'unknown'
    if bad == 'sites': state['adapter']['metadata']['injection_sites'] = [1]
    if bad == 'roles': state['adapter']['metadata']['use_roles'] = False
    if bad == 'nan': state['adapter']['adapters']['banks.0.W_h.weight'][0,0] = float('nan')
    if bad == 'dtype': state['adapter']['adapters']['banks.0.W_h.bias'] = torch.ones(2).half()
    if bad == 'extra': state['adapter']['metadata']['unreviewed'] = True
    with pytest.raises((ValueError,TypeError)):
        checkpoints.convert_full_research_state(state, config)


def test_disk_export_provenance_and_no_overwrite(tmp_path):
    checkpoints = checkpoint_module()
    state, config = research_state()
    source, dest = tmp_path/'research.pt', tmp_path/'full.pt'
    torch.save(state, source)
    before = source.read_bytes()
    report = checkpoints.export_full_research_adapter(source, dest, config)
    assert source.read_bytes() == before
    assert report['tensor_values_unchanged'] is True
    assert json.loads((tmp_path/'full.pt.json').read_text()) == report
    saved = dest.read_bytes()
    with pytest.raises(FileExistsError):
        checkpoints.export_full_research_adapter(source,dest,config)
    assert dest.read_bytes() == saved


def test_existing_sidecar_prevents_partial_export(tmp_path):
    checkpoints = checkpoint_module()
    state, config = research_state()
    source, dest = tmp_path/'research.pt', tmp_path/'full.pt'
    torch.save(state, source)
    (tmp_path/'full.pt.json').write_text('existing')
    with pytest.raises(FileExistsError):
        checkpoints.export_full_research_adapter(source,dest,config)
    assert not dest.exists()

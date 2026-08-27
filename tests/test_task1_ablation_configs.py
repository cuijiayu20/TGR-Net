from pathlib import Path

import pytest

mmengine = pytest.importorskip('mmengine')
from mmengine.config import Config


CONFIG_DIR = (
    Path(__file__).resolve().parents[1] / 'configs' / 'open_world' /
    'weather_owod_mamba_kat' / 'ablations')

EXPECTED = {
    'task1_full.py': {
        'branches': (True, True, True),
        'cwd': 10.0,
        'det': (0.0, 0.1, 0.1),
    },
    'task1_wo_mamba.py': {
        'branches': (False, True, True),
        'cwd': 10.0,
        'det': (0.0, 0.1, 0.1),
    },
    'task1_wo_kat.py': {
        'branches': (True, False, True),
        'cwd': 10.0,
        'det': (0.0, 0.1, 0.1),
    },
    'task1_wo_conv.py': {
        'branches': (True, True, False),
        'cwd': 10.0,
        'det': (0.0, 0.1, 0.1),
    },
    'task1_det_only.py': {
        'branches': (True, True, True),
        'cwd': None,
        'det': (0.0, 0.1, 0.1),
    },
    'task1_cwd_only.py': {
        'branches': (True, True, True),
        'cwd': 10.0,
        'det': (0.0, 0.0, 0.0),
    },
}

@pytest.mark.parametrize('config_name', EXPECTED)
def test_task1_ablation_config(config_name):
    cfg = Config.fromfile(CONFIG_DIR / config_name)
    expected = EXPECTED[config_name]
    model = cfg.model
    adapter = model.degradation_adapter
    det_weights = model.det_loss_weights

    assert cfg.max_epochs == 6
    assert cfg.train_cfg.max_epochs == 6
    assert cfg.train_cfg.val_interval == 6
    assert cfg.default_hooks.param_scheduler.max_epochs == 500
    assert cfg.randomness.seed == 3407
    assert cfg.custom_hooks == []
    assert model.pipline == []

    assert model.num_train_classes == 5
    assert model.num_test_classes == 5
    assert model.train_cfg.assigner.num_classes == 5
    assert model.bbox_head.head_module.num_classes == 5
    assert model.test_cfg.score_thr == 0.30
    assert model.freeze_base_detector is True

    assert cfg.train_dataloader.dataset.dataset.ann_file == (
        'annotations/coco/task1/train_fog.json')
    assert cfg.val_dataloader.dataset.ann_file == (
        'annotations/coco/task1/test_fog_rain_snow_ow.json')
    assert cfg.val_evaluator.cfg.file_name == 'test_fog_rain_snow.txt'
    assert cfg.default_hooks.checkpoint.save_best == 'Known AP50'

    assert adapter.use_mamba_ssm is False
    assert (
        adapter.use_mamba_branch,
        adapter.use_kat_branch,
        adapter.use_conv_branch,
    ) == expected['branches']

    if expected['cwd'] is None:
        assert model.alignment_loss is None
    else:
        assert model.alignment_loss.loss_weight == expected['cwd']
        assert model.alignment_loss.temperature == 1.0

    assert (
        det_weights.loss_cls,
        det_weights.loss_bbox,
        det_weights.loss_dfl,
    ) == expected['det']

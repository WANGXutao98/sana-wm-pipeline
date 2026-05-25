"""Minimal smoke test that the package imports and pipeline config loads."""
from pathlib import Path
import yaml

import sana_wm_pipeline


def test_version_defined():
    assert sana_wm_pipeline.__version__ == "0.1.0"


def test_pipeline_config_loads():
    cfg_path = Path(__file__).parent.parent / "configs" / "pipeline.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    # Paper-fixed values (arXiv:2605.15178v1)
    assert cfg["target"]["resolution"] == [1280, 720]
    assert cfg["target"]["fps"] == 16
    assert cfg["target"]["camera_frames"] == 961
    assert cfg["target"]["raw_per_latent"] == 8
    assert cfg["target"]["vae_channels"] == 128
    assert cfg["depth_fusion"]["ema_momentum"] == 0.99
    assert cfg["umeyama"]["inlier_percentile"] == 80
    assert cfg["camera_quality"]["fov_deg_min"] == 25
    assert cfg["camera_quality"]["fov_deg_max"] == 120
    assert cfg["camera_quality"]["focal_divergence_max"] == 0.20
    assert cfg["camera_quality"]["scale_cv_max"] == 2.0
    assert cfg["difix3d"]["timestep"] == 199
    assert cfg["difix3d"]["prompt"] == "remove degradation"
    assert cfg["difix3d"]["guidance"] == 0.0

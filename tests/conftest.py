import pytest
from tests.fixtures.generate import generate_experiment


@pytest.fixture
def tiny_experiment(tmp_path):
    return generate_experiment(tmp_path/"exp", wells=["A1", "B2"], fovs_per_well=1, nt=2,
        recording_frames=5, recording_planes=3,
        zstack_channels=[("DAPI", "#4a9eff"), ("GFP", "#34d058")], zstack_nz=4)


@pytest.fixture
def tiny_experiment_poor(tmp_path):
    return generate_experiment(tmp_path/"exp", wells=["A1"], fovs_per_well=6, nt=1,
        recording_frames=5, recording_planes=1,
        zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=3, rich=False)

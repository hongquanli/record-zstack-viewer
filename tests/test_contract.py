# tests/test_contract.py
def test_contract_version_is_semver():
    from record_zstack_viewer.contract.version import CONTRACT_VERSION
    parts = CONTRACT_VERSION.split(".")
    assert len(parts) == 2 and all(p.isdigit() for p in parts)


def test_validator_passes_rich_fixture(tiny_experiment):
    from record_zstack_viewer.contract.validate import validate_experiment
    r = validate_experiment(tiny_experiment)
    assert r.ok and not r.errors


def test_validator_warns_on_missing_sidecars(tiny_experiment_poor):
    from record_zstack_viewer.contract.validate import validate_experiment
    r = validate_experiment(tiny_experiment_poor)
    assert r.ok  # stable contract still satisfied
    assert any("coordinates.csv" in w for w in r.warnings)
    assert any("events.jsonl" in w for w in r.warnings)

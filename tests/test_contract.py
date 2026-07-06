# tests/test_contract.py
def test_contract_version_is_semver():
    from record_zstack_viewer.contract.version import CONTRACT_VERSION
    parts = CONTRACT_VERSION.split(".")
    assert len(parts) == 2 and all(p.isdigit() for p in parts)

from assignment.student_assignment.configerator import Configerator


def test_from_config_creates_an_isolated_policy_loader(monkeypatch):
    config = {
        "grade": 6,
        "subconfigs": ["status_quo"],
        "nested": {"value": "base"},
    }
    configurator = Configerator.from_config(config)
    monkeypatch.setattr(configurator, "_validate_schema", lambda *args: None)
    monkeypatch.setattr(
        configurator,
        "_load_yaml",
        lambda path: {"nested": {"value": "policy"}},
    )

    config["nested"]["value"] = "caller"
    assert configurator.config["nested"]["value"] == "base"
    assert configurator.original_config["nested"]["value"] == "base"
    assert configurator.config["grade"] == "06"
    assert configurator.original_config["grade"] == "06"

    assert configurator.load_next_subconfig() is True
    assert configurator.config["nested"]["value"] == "policy"
    assert configurator.config["subconfig-name"] == "status_quo"
    assert configurator.load_next_subconfig() is False

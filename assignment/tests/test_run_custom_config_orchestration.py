import copy
import json

import pytest
import yaml

from assignment import run_custom_config


def _config(output_dir, subconfigs):
    return {
        "data": {"scenario": "legacy", "overrides": {}},
        "save-assignment": True,
        "paths": {"assignment-folder": str(output_dir)},
        "subconfigs": subconfigs,
        "utility-model": {"enable": True, "save-path": str(output_dir / "u.csv")},
    }


def test_worker_does_not_write_root_config_or_non_owned_utility(monkeypatch, tmp_path):
    seen = []

    class FakeMarketGenerator:
        def __init__(self, *, config, write_config):
            seen.append((copy.deepcopy(config), write_config))

        def simulate(self):
            return None

    monkeypatch.setattr(run_custom_config, "MarketGenerator", FakeMarketGenerator)
    config = _config(tmp_path, ["first", "second"])

    run_custom_config._run_subconfig_worker(config, "first", False)
    run_custom_config._run_subconfig_worker(config, "second", True)

    first_config, first_writes_config = seen[0]
    second_config, second_writes_config = seen[1]
    assert first_config["subconfigs"] == ["first"]
    assert "save-path" not in first_config["utility-model"]
    assert second_config["subconfigs"] == ["second"]
    assert second_config["utility-model"]["save-path"].endswith("u.csv")
    assert not first_writes_config
    assert not second_writes_config
    assert "save-path" in config["utility-model"]


def test_parallel_attempts_every_subconfig_and_attributes_failures(
    monkeypatch, tmp_path
):
    attempted = []
    submissions = []

    class ImmediateFuture:
        def __init__(self, function, args):
            try:
                self.value = function(*args)
                self.error = None
            except Exception as exc:
                self.value = None
                self.error = exc

        def result(self):
            if self.error is not None:
                raise self.error
            return self.value

    class ImmediateExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, *args):
            submissions.append(args)
            return ImmediateFuture(function, args)

    def fake_worker(_config, subconfig_name, _write_shared_output):
        attempted.append(subconfig_name)
        if subconfig_name.startswith("bad"):
            raise ValueError(f"failure in {subconfig_name}")

    monkeypatch.setattr(run_custom_config, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(
        run_custom_config, "as_completed", lambda futures: reversed(list(futures))
    )
    monkeypatch.setattr(run_custom_config, "_run_subconfig_worker", fake_worker)

    config = _config(tmp_path / "runs", ["bad-first", "good", "bad-last"])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(RuntimeError) as exc_info:
        run_custom_config.generate.callback(
            config_path=str(config_path), sample=None, frac=None, workers=3
        )

    assert attempted == ["bad-first", "good", "bad-last"]
    message = str(exc_info.value)
    assert message.index("bad-first") < message.index("bad-last")
    assert "good" not in message
    assert [args[-1] for args in submissions] == [False, False, True]

    provenance = json.loads((tmp_path / "runs" / "config.json").read_text())
    assert provenance["subconfigs"] == ["bad-first", "good", "bad-last"]


def test_parallel_direct_config_runs_once(monkeypatch, tmp_path):
    seen = []

    class FakeMarketGenerator:
        def __init__(self, *, config, write_config):
            seen.append((copy.deepcopy(config), write_config, 0))

        def simulate(self):
            config, write_config, calls = seen[-1]
            seen[-1] = (config, write_config, calls + 1)

    monkeypatch.setattr(run_custom_config, "MarketGenerator", FakeMarketGenerator)
    config = _config(tmp_path / "runs", [])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_custom_config.generate.callback(
        config_path=str(config_path), sample=None, frac=None, workers=4
    )

    assert len(seen) == 1
    assert seen[0][1:] == (False, 1)

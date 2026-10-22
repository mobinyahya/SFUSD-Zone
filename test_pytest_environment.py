import os


def test_pytest_uses_noninteractive_matplotlib_backend():
    import matplotlib

    assert os.environ["MPLBACKEND"] == "Agg"
    assert matplotlib.get_backend().lower() == "agg"

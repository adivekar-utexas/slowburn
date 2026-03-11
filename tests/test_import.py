"""Smoke tests verifying the package can be imported."""


def test_import_slowburn() -> None:
    import slowburn  # noqa: F401


def test_version_is_string() -> None:
    from importlib.metadata import version

    v = version("slowburn")
    assert isinstance(v, str)
    assert len(v) > 0

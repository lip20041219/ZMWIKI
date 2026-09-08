def test_import():
    import zmrag

    assert zmrag.__version__


def test_version():
    import zmrag

    assert zmrag.__version__.count(".") == 2

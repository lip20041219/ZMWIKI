def test_import():
    import zmwiki

    assert zmwiki.__version__


def test_version():
    import zmwiki

    assert zmwiki.__version__.count(".") == 2

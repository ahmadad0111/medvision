import importlib


def test_labels_cover_seven_classes():
    import src.core.config as c
    importlib.reload(c)
    assert c.Config.NUM_CLASSES == 7
    assert len(c.Config.labels()) == 7
    assert "melanoma" in c.Config.labels().values()


def test_backbone_override(monkeypatch):
    monkeypatch.setenv("BACKBONE", "vit_small_patch16_224")
    import src.core.config as c
    importlib.reload(c)
    assert c.Config.BACKBONE == "vit_small_patch16_224"
    assert "backbone" in c.Config.summary()

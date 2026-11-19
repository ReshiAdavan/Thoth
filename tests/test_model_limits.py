import json
from pathlib import Path
import pytest
from thoth import load

CASES = json.loads(Path("tests/fixtures/models.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_python_import(case, tmp_path):
    p = tmp_path / "model.json"
    p.write_text(json.dumps(case["model"]))
    if case["valid"]:
        assert load(p).encode("abab") == [257]
    else:
        with pytest.raises(ValueError):
            load(p)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_native_import(case):
    native = pytest.importorskip("thoth_native")
    if case["valid"]:
        assert native.NativeTokenizer.from_json(json.dumps(case["model"])).encode(
            "abab"
        ) == [257]
    else:
        with pytest.raises(ValueError):
            native.NativeTokenizer.from_json(json.dumps(case["model"]))

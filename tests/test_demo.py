import pytest
from thoth_demo import execute


@pytest.mark.parametrize("name", ["local-basic", "local-regex", "local-rust"])
def test_local_training(name):
    if name == "local-rust":
        pytest.importorskip("thoth_native")
    state = execute(
        {"name": name, "operation": "train", "text": "abababab", "size": 258}
    )
    result = execute({"name": name, "text": "abababab", "modelId": state["modelId"]})
    assert result["ids"] == [257, 257]
    assert bytes(b for chunk in result["bytes"] for b in chunk).decode() == "abababab"


def test_sentencepiece_demo():
    state = execute(
        {
            "name": "local-llama",
            "operation": "train",
            "text": "hello hello world",
            "size": 256,
        }
    )
    result = execute(
        {"name": "local-llama", "text": "hello world", "modelId": state["modelId"]}
    )
    assert result["decoded"] == "hello world"
    assert len(result["ids"]) == len(result["pieces"])


def test_demo_errors():
    with pytest.raises(ValueError):
        execute({"name": "other"})
    with pytest.raises(ValueError):
        execute({"name": "local-llama", "text": "hello"})
    with pytest.raises(ValueError):
        execute({"name": "local-basic", "text": "a" * 20001})

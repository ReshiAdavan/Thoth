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
            "size": 300,
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
        execute({"name": "local-llama", "text": "hello", "modelId": "expired"})
    with pytest.raises(ValueError):
        execute({"name": "local-basic", "text": "a" * 20001})


@pytest.mark.parametrize(
    "name", ["local-basic", "local-regex", "local-rust", "local-llama"]
)
def test_saved_family(name):
    if name == "local-rust":
        pytest.importorskip("thoth_native")
    result = execute(
        {
            "name": name,
            "text": "Hello world! Python programming is fun.",
            "compare": True,
        }
    )
    assert len(result["ids"]) < 25
    assert set(result["counts"]) == {name, "cl100k_base"}
    if name == "local-llama":
        assert result["decoded"] == "Hello world! Python programming is fun."
    else:
        assert (
            bytes(b for chunk in result["bytes"] for b in chunk).decode()
            == "Hello world! Python programming is fun."
        )

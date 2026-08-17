from ultron_reasoning.tokenizer import ByteTokenizer


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    text = "Ultron reasoning: 2 + 2 = 4"
    ids = tok.encode(text)
    assert tok.decode(ids) == text

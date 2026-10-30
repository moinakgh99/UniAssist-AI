"""Optional FLAN-T5 helper (experimental). The model is loaded lazily on first use, so importing this
module is cheap and does not require torch/transformers until get_answer() is actually called."""

MODEL_PATH = "google/flan-t5-small"

_tokenizer = None
_model = None


def _load():
    """Load (once) and return (tokenizer, model)."""
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        _model.to("cpu")
    return _tokenizer, _model


def get_answer(question: str) -> str:
    """Generate an answer to *question* using FLAN-T5-small.
    The prompt frames the model as a university student-support assistant.
    """
    import torch

    tokenizer, model = _load()
    prompt = (
        "You are a university student support assistant. "
        "Answer the student's question clearly and concisely. "
        f"Question: {question}"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to("cpu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=2,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    q = input("Ask your university chatbot: ")
    print("Answer:", get_answer(q))

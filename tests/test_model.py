import pytest
from src.model.predict import GujaratiQAModel
from src.model.evaluate import f1_score, exact_match_score

@pytest.fixture(scope="module")
def model():
    return GujaratiQAModel("models/gujarati-qa-best")

def test_model_loads(model):
    assert model is not None

def test_basic_prediction(model):
    question = "ગુજરાતની રાજધાની કઈ છે?"
    context = "ગાંધીનગર ગુજરાતની રાજધાની છે."
    result = model.answer(question, context)

    assert "answer" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1

def test_confidence_on_irrelevant_context(model):
    """Model should have low confidence when context is unrelated"""
    question = "ગુજરાતની રાજધાની કઈ છે?"
    context = "ક્રિકેટ એ ભારતની સૌથી લોકપ્રિય રમત છે."
    result = model.answer(question, context)
    assert result["confidence"] < 0.7

def test_f1_score():
    assert f1_score("ગાંધીનગર", "ગાંધીનગર") == 1.0
    assert f1_score("", "ગાંધીનગર") == 0

def test_exact_match():
    assert exact_match_score("ગાંધીનગર", "ગાંધીનગર") is True
    assert exact_match_score("અમદાવાદ", "ગાંધીનગર") is False
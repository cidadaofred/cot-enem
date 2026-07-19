import json

from cot_enem.dataset.schema import NormalizedQuestion
from cot_enem.generation.prompting import markup_to_plain_text, render_question_plain


def test_mathml_is_converted_to_plain_fraction_and_power():
    value = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        "<mfrac><mn>1</mn><msup><mi>x</mi><mn>2</mn></msup></mfrac>"
        "</math>"
    )

    assert markup_to_plain_text(value) == "(1)/(x^(2))"


def test_plain_question_render_does_not_expose_markup_to_evolution_agent():
    question = NormalizedQuestion(
        id="q1",
        statement="<p>Calcule <strong>x</strong>.</p>",
        alternatives={
            "A": '<math xmlns="m"><mn>1</mn></math>',
            "B": "2",
            "C": "3",
            "D": "4",
            "E": "5",
        },
        gold_answer="A",
    )

    rendered = render_question_plain(question)
    payload = json.loads(rendered)

    assert "<" not in rendered
    assert "xmlns" not in rendered
    assert payload["statement"] == "Calcule x."
    assert payload["alternatives"]["A"] == "1"

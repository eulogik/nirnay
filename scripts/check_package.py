"""Package smoke test: import, label handling, CLI parser, no model load."""
from __future__ import annotations

import json


def main() -> int:
    import nirnay_450m
    from nirnay_450m import Classifier, Decision, NirnayAgent, normalise_labels
    from nirnay_450m.cli import build_parser

    assert nirnay_450m.__version__ == "0.2.1", nirnay_450m.__version__

    labels = normalise_labels({"a": "first", "b": ""})
    assert labels == {"a": "first", "b": "b"}, labels
    labels = normalise_labels(["one", "two_three"])
    assert labels == {"one": "one", "two_three": "two three"}, labels
    banking = normalise_labels("banking77")
    assert len(banking) == 77 and "card_arrival" in banking

    parser = build_parser()
    args = parser.parse_args(["classify", "hello", "--labels", "banking77"])
    assert args.command == "classify" and args.text == "hello"
    args = parser.parse_args(["serve", "--port", "9001"])
    assert args.command == "serve" and args.port == 9001
    args = parser.parse_args(["fit", "--data", "x.jsonl"])
    assert args.command == "fit" and args.data == "x.jsonl"

    d = Decision(choice="a", confidence=0.9, probabilities={"a": 0.9, "b": 0.1})
    assert json.loads(json.dumps(d.as_dict()))["choice"] == "a"

    # No model download in this check.
    c = Classifier(checkpoint="/definitely/not/here.pt")
    try:
        c.resolve_checkpoint()
        raise AssertionError("missing checkpoint should raise")
    except FileNotFoundError:
        pass

    assert Classifier and NirnayAgent
    print("PACKAGE_OK version=0.2.1 labels77=true cli=classify,serve,doctor,fit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

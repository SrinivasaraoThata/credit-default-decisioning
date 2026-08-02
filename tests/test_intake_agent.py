from agents.intake_agent import run_intake


def test_valid_application_passes_intake(valid_application):
    state = run_intake({"raw_application": valid_application, "trace": []})
    assert state["intake_valid"] is True
    assert state["intake_errors"] == []
    assert state["application"] is not None
    assert state["trace"][-1]["agent"] == "intake"


def test_missing_field_is_rejected(valid_application):
    bad = dict(valid_application)
    del bad["limit_bal"]
    state = run_intake({"raw_application": bad, "trace": []})
    assert state["intake_valid"] is False
    assert any("limit_bal" in e for e in state["intake_errors"])


def test_out_of_range_field_is_rejected(valid_application):
    bad = dict(valid_application)
    bad["sex"] = 1
    bad["age"] = 5  # below allowed minimum
    state = run_intake({"raw_application": bad, "trace": []})
    assert state["intake_valid"] is False
    assert any("age" in e for e in state["intake_errors"])


def test_negative_limit_is_rejected(valid_application):
    bad = dict(valid_application)
    bad["limit_bal"] = -100
    state = run_intake({"raw_application": bad, "trace": []})
    assert state["intake_valid"] is False

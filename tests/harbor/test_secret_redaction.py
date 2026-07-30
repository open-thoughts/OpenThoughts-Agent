from scripts.harbor.secret_redaction import redact, redact_record


def test_redact_replaces_known_credentials_and_reports_shapes() -> None:
    token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJydW4ifQ.signature"

    cleaned, findings = redact(f"proxy={token} key=sk-abcdefghijklmnopqrstuvwxyz")

    assert token not in cleaned
    assert "<redacted:jwt>" in cleaned
    assert {finding.shape for finding in findings} == {"jwt", "openai_key"}


def test_redact_record_is_recursive_and_idempotent() -> None:
    record = {"messages": [{"content": "hf_abcdefghijklmnopqrstuvwxyz"}]}

    cleaned, findings = redact_record(record)
    cleaned_again, later_findings = redact_record(cleaned)

    assert record["messages"][0]["content"].startswith("hf_")
    assert cleaned["messages"][0]["content"] == "<redacted:huggingface_token>"
    assert findings[0].location == "messages[0].content"
    assert cleaned_again == cleaned
    assert later_findings == []

"""Secret redaction for artifacts and logs (spec sections 43.1, 69).

Best-effort pattern-based redaction. This is not a promise of perfect secret
detection; raw artifact storage is opt-in via `--save-raw` and still passes
through this module.
"""

from __future__ import annotations

import re

# Ordered list of (compiled pattern, replacement). More specific patterns first.
PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"Authorization:\s*Bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE),
        "Authorization: Bearer [REDACTED]",
    ),
    (
        re.compile(r"Authorization:\s*[A-Za-z]+\s+[A-Za-z0-9._\-]+", re.IGNORECASE),
        "Authorization: [REDACTED]",
    ),
    (
        re.compile(r"api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9._\-]{8,}['\"]?", re.IGNORECASE),
        "api_key=[REDACTED]",
    ),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    # Fine-grained GitHub PATs ('github_pat_...') do not match the classic
    # gh[pousr]_ prefix and would otherwise pass through untouched.
    (re.compile(r"\bgithub_pat_[A-Za-z0-9]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    # AWS access key IDs anywhere (bare or assigned): 20 chars starting
    # AKIA/ASIA, uppercase alnum. The assignment-only pattern above misses
    # a bare key quoted in prose.
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY_ID]"),
    # Alternation must not match the bare vendor word 'atlassian' — that
    # would rewrite benign text that merely mentions Atlassian/Jira.
    (
        re.compile(r"(?:atlassian|bitbucket)[ _-]?token\s*[:=]\s*\S+", re.IGNORECASE),
        "bitbucket_token=[REDACTED]",
    ),
    (re.compile(r"xox[baprs]-[A-Za-z0-9\-]+"), "[REDACTED_SLACK_TOKEN]"),
    (
        re.compile(r"(?:AWS_ACCESS_KEY_ID|aws_access_key_id)\s*[:=]\s*\S+"),
        "AWS_ACCESS_KEY_ID=[REDACTED]",
    ),
    (
        re.compile(r"(?:AWS_SECRET_ACCESS_KEY|aws_secret_access_key)\s*[:=]\s*\S+"),
        "AWS_SECRET_ACCESS_KEY=[REDACTED]",
    ),
    # JWTs: three base64url segments, middle one non-trivial.
    (
        re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]*\b"),
        "[REDACTED_JWT]",
    ),
    # PEM private key blocks.
    (
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----.*?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    # Passwords in key=value form. Only matches an explicit assignment with
    # a non-trivial value; prose ("the password was discussed") is untouched.
    # The lookaheads exclude values that are source code rather than
    # secrets (states legitimately embed real upstream code, e.g. Django):
    #   - a bare identifier or dotted chain followed by '(' — a call, e.g.
    #     'password = ReadOnlyPasswordHashField(' or
    #     'password = self.fields.get("password")'
    #   - a bare dotted identifier chain — an attribute reference, e.g.
    #     'if password: password.help_text = ...' where the ':' is Python
    #     syntax, not an assignment separator
    # Single-token unquoted values ('password=hunter2secret') and quoted
    # values are still matched.
    (
        re.compile(
            r"\b\w*pass(?:word|wd)?(?:[_-]?key)?\s*[:=]\s*['\"]?"
            r"(?![A-Za-z_][A-Za-z0-9_.]*\()"
            r"(?![A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)"
            r"[^\s'\"]{8,}['\"]?",
            re.IGNORECASE,
        ),
        "password=[REDACTED]",
    ),
    # Database/connection URLs with embedded credentials.
    (
        re.compile(
            r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:@/]+:[^\s@/]+@",
            re.IGNORECASE,
        ),
        "[REDACTED_DB_URL]",
    ),
]


def redact_text(text: str) -> str:
    """Redact common secret formats from a string."""
    for pattern, replacement in PATTERNS:
        text = pattern.sub(replacement, text)
    return text

"""Typed error hierarchy.

The workflow decides failure policy by catching these types rather than
inspecting strings, so that "review incomplete" can never be reported as a
passing review (design doc section 5.3).
"""

from __future__ import annotations


class CodeReviewError(Exception):
    """Base class for all agent errors."""


class InputError(CodeReviewError):
    """Invalid user input: missing directory, unreadable target, bad config."""


class SecurityError(CodeReviewError):
    """A path or command escaped the authorised project sandbox."""


class ToolError(CodeReviewError):
    """A tool (scanner, reader, static analyzer) failed."""


class ToolNotAvailableError(ToolError):
    """An external checker is not installed on this machine."""


class LLMError(CodeReviewError):
    """Base class for model-facing failures."""


class LLMRequestError(LLMError):
    """Transport / provider error. Retryable."""


class LLMResponseFormatError(LLMError):
    """Model output was not parseable into the required schema. Retryable."""


class BudgetExceededError(CodeReviewError):
    """Token or cost budget exhausted; stop issuing new model requests."""


class ReportError(CodeReviewError):
    """Report could not be written. Must never be swallowed as success."""

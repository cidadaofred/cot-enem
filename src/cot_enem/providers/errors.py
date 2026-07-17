"""Errors safe to log without exposing credentials."""


class ProviderError(RuntimeError):
    """Base provider failure."""


class ProviderConfigurationError(ProviderError):
    """Missing or inconsistent provider configuration."""


class ProviderRequestError(ProviderError):
    """Remote or local inference request failed."""


class StructuredResponseError(ProviderError):
    """The model did not return a JSON object matching basic expectations."""

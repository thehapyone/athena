"""Upload failures whose message is safe to show a user.

Every message here ends up in a job ``detail`` and, through the Node proxy, in
the browser. They must therefore never carry a URL, a token, or a stack detail.
"""


class DocumentError(Exception):
    """Base class for upload failures with a user-facing message."""


class UnsupportedDocumentError(DocumentError):
    """The extension or declared media type is not accepted."""


class DocumentTooLargeError(DocumentError):
    """The upload, or the text it converted to, exceeds a configured limit."""


class DocumentDecodeError(DocumentError):
    """A text upload was not decodable UTF-8, or held no usable content."""


class ConversionUnavailableError(DocumentError):
    """No document converter is configured or reachable for this format."""


class ConversionFailedError(DocumentError):
    """The converter answered, but not with usable document text."""

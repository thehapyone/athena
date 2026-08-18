"""Upload failures whose message is safe to show a user.

Every message here ends up in a job ``detail`` and, through the Node proxy, in
the browser. They must therefore never carry a URL, a token, or a stack detail.

The conversion failures are deliberately distinct types rather than one error
with different text: an asynchronous conversion can fail at submission, in the
remote worker, by losing its task, or by outrunning its deadline, and only some
of those are worth retrying with the same file.
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


class ConversionSubmissionError(DocumentError):
    """The converter refused to accept the document for conversion."""


class ConversionFailedError(DocumentError):
    """The converter answered, but not with usable document text."""


class ConversionTaskLostError(DocumentError):
    """The converter no longer knows the task this document was submitted as.

    Raised when a task has expired or the converter was restarted. The document
    itself is still fine, so uploading it again starts a fresh conversion.
    """


class ConversionDeadlineExceededError(DocumentError):
    """The whole conversion, across submission and polling, ran out of time."""


class ConversionResultUnavailableError(DocumentError):
    """The conversion finished, but its result could not be retrieved."""

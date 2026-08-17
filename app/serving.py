"""Rules for handing stored source bytes back over HTTP.

Two things are decided here rather than at the route, because both are security
properties rather than presentation choices: which media types may ever render
inline in a browser, and how a user-supplied file name becomes a header value.
"""

from dataclasses import dataclass
from urllib.parse import quote

# Types a browser may render in place. Everything else downloads, so an uploaded
# HTML or Office file can never execute in this origin. Note that ``text/html``
# is deliberately absent: user HTML is only ever served as an attachment, and its
# normalized text preview is what the viewer shows instead.
INLINE_MEDIA_TYPES = frozenset({"application/pdf", "text/plain", "text/markdown"})
DOWNLOAD_MEDIA_TYPE = "application/octet-stream"
PREVIEW_MEDIA_TYPE = "text/plain; charset=utf-8"

_ASCII_FALLBACK_CHARACTER = "_"
_MAXIMUM_HEADER_FILENAME_CHARACTERS = 200


@dataclass(frozen=True, slots=True)
class ByteRange:
    """One satisfiable range of a known-length object."""

    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length - 1


class UnsatisfiableRangeError(ValueError):
    """Raised when a Range header names bytes the object does not have."""


def resolve_media_type(media_type: str, *, variant: str) -> tuple[str, bool]:
    """Return the response media type and whether it may be shown inline.

    A stored type that is not on the inline allowlist is downgraded to
    ``application/octet-stream``, so a browser cannot be talked into rendering it
    by the stored value alone.
    """
    if variant == "preview":
        return PREVIEW_MEDIA_TYPE, True
    base = (media_type or "").split(";", 1)[0].strip().lower()
    if base in INLINE_MEDIA_TYPES:
        charset = "; charset=utf-8" if base.startswith("text/") else ""
        return f"{base}{charset}", True
    return DOWNLOAD_MEDIA_TYPE, False


def content_disposition(filename: str, *, inline: bool) -> str:
    """Build a Content-Disposition value that cannot inject a header.

    The name reaching this function has already been reduced to a bare label by
    the upload path. It is still re-sanitized: the quoted form keeps only printable
    ASCII, and the exact original is carried in the RFC 5987 ``filename*`` form,
    which is percent-encoded and therefore has no delimiter to escape.
    """
    trimmed = filename.strip()[:_MAXIMUM_HEADER_FILENAME_CHARACTERS]
    ascii_name = "".join(
        character if 0x20 <= ord(character) < 0x7F and character not in '"\\' else _ASCII_FALLBACK_CHARACTER
        for character in trimmed
    ).strip()
    if not ascii_name or set(ascii_name) <= {_ASCII_FALLBACK_CHARACTER, " ", "."}:
        ascii_name = "document"
    encoded = quote(trimmed or "document", safe="")
    kind = "inline" if inline else "attachment"
    return f'{kind}; filename="{ascii_name}"; filename*=UTF-8\'\'{encoded}'


def parse_range(header: str | None, byte_size: int) -> ByteRange | None:
    """Resolve a Range header against an object of *byte_size* bytes.

    Returns ``None`` when the client asked for the whole object or used a form
    this service does not implement — a multi-range request is answered in full,
    which is a valid response. Raises :class:`UnsatisfiableRangeError` when the
    range is well-formed but outside the object.
    """
    if not header or byte_size <= 0:
        return None
    prefix, separator, raw = header.partition("=")
    if not separator or prefix.strip().lower() != "bytes":
        return None
    specifications = [part.strip() for part in raw.split(",")]
    if len(specifications) != 1 or not specifications[0]:
        return None

    first, dash, last = specifications[0].partition("-")
    if not dash:
        return None
    # Parsing is kept separate from validation so an UnsatisfiableRangeError —
    # itself a ValueError — is never swallowed as an unparsable header.
    try:
        first_byte = int(first) if first else None
        last_byte = int(last) if last else None
    except ValueError:
        return None

    if first_byte is None:
        if last_byte is None or last_byte <= 0:
            raise UnsatisfiableRangeError("A suffix range must ask for at least one byte.")
        start = max(0, byte_size - last_byte)
        return ByteRange(start=start, length=byte_size - start)

    start = first_byte
    if start < 0 or (last_byte is not None and last_byte < start):
        return None
    if start >= byte_size:
        raise UnsatisfiableRangeError("The requested range starts past the end of the file.")
    end = byte_size - 1 if last_byte is None else min(last_byte, byte_size - 1)
    return ByteRange(start=start, length=end - start + 1)

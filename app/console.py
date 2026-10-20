"""Console diagnostics must not abort accepted business operations."""
import sys

def console_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="backslashreplace").decode(encoding))

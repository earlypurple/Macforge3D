"""
Stub file for blosc2 module.
"""

from typing import Any, Union, Optional, ByteString

def compress(
    data: ByteString, clevel: int = 5, typesize: int = 8, *args: Any, **kwargs: Any
) -> bytes: ...
def decompress(data: ByteString, *args: Any, **kwargs: Any) -> bytes: ...

"""
Stub file for lz4 module.
"""

from typing import Any, Union, Optional, ByteString

class frame:
    @staticmethod
    def compress(
        data: ByteString, compression_level: int = 0, *args: Any, **kwargs: Any
    ) -> bytes: ...
    @staticmethod
    def decompress(data: ByteString, *args: Any, **kwargs: Any) -> bytes: ...

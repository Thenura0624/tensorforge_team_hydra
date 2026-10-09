"""Startup-configured authentication; no expiry, rotation, logging or disk writes."""
import hashlib
import hmac

class APIKeyAuth:
    def __init__(self, key):
        # Preserve the exact configured string, including any meaningful spaces.
        self._digest = hashlib.sha256(key.encode('utf-8')).digest() if key else None

    def accepts(self, headers):
        candidates = [headers.get('x-api-key', '')]
        scheme, separator, value = headers.get('authorization', '').partition(' ')
        if separator and scheme.lower() == 'bearer':
            candidates.append(value)
        match = False
        for candidate in candidates:
            actual = hashlib.sha256(candidate.encode('utf-8')).digest()
            match |= hmac.compare_digest(actual, self._digest or bytes(32))
        return self._digest is not None and match

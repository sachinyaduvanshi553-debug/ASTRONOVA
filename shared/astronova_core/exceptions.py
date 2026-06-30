from fastapi import status


class AstroNovaError(Exception):
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class DataIngestionError(AstroNovaError):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_400_BAD_REQUEST)

class DataValidationError(AstroNovaError):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_422_UNPROCESSABLE_ENTITY)

class DataProcessingError(AstroNovaError):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)

class ModelNotFoundError(AstroNovaError):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_444_NOT_FOUND if hasattr(status, "HTTP_444_NOT_FOUND") else 404)

class InferenceError(AstroNovaError):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)

class AuthenticationError(AstroNovaError):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_401_UNAUTHORIZED)

class AuthorizationError(AstroNovaError):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_403_FORBIDDEN)

class ServiceUnavailableError(AstroNovaError):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_503_SERVICE_UNAVAILABLE)

class RateLimitError(AstroNovaError):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_429_TOO_MANY_REQUESTS)

# Backward compatibility alias
AstroNovaException = AstroNovaError

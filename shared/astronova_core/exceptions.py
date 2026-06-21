from fastapi import status

class AstroNovaException(Exception):
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class DataIngestionError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_400_BAD_REQUEST)

class DataValidationError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_422_UNPROCESSABLE_ENTITY)

class DataProcessingError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)

class ModelNotFoundError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_444_NOT_FOUND if hasattr(status, "HTTP_444_NOT_FOUND") else 404)

class InferenceError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)

class AuthenticationError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_401_UNAUTHORIZED)

class AuthorizationError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_403_FORBIDDEN)

class ServiceUnavailableError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_503_SERVICE_UNAVAILABLE)

class RateLimitError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_429_TOO_MANY_REQUESTS)

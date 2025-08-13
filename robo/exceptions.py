
class UnknownConversationException(BaseException):
    """Raised when a conversation cannot be found for continuation"""

class FieldValuesMissingException(BaseException):
    """Raised when expected interpolable fields were not provided"""

__all__ = ['UnknownConversationException', 'FieldValuesMissingException']
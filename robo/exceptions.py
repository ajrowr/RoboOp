
class UnknownConversationException(BaseException):
    """Raised when a conversation cannot be found for continuation"""

class FieldValuesMissingException(BaseException):
    """Raised when expected interpolable fields were not provided"""

class SyncAsyncMismatchError(BaseException):
    """Raised when async operations are attempted in a sync-mode context, and vice versa"""

__all__ = ['UnknownConversationException', 'FieldValuesMissingException', 'SyncAsyncMismatchError']
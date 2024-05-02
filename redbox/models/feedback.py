from typing import Optional

from redbox.models.base import PersistableModel
from redbox.models.chat import ChatMessage


class Feedback(PersistableModel):
    input: ChatMessage | list[ChatMessage]
    output: ChatMessage
    feedback_type: str
    feedback_score: str
    feedback_text: Optional[str] = None

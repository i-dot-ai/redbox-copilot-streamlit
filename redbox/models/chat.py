from typing import Literal
from typing_extensions import TypedDict


# class ChatMessage(PersistableModel):
#     # langchain.chains.base.Chain needs pydantic v1, breaks
#     # https://python.langchain.com/docs/guides/pydantic_compatibility
#     chain: Optional[object] = None
#     message: object

#     @field_serializer("chain")
#     def serialise_chain(self, chain: Chain, _info):
#         if isinstance(chain, Chain):
#             return chain.dict()
#         else:
#             return chain

#     @field_serializer("message")
#     def serialise_message(self, message: AIMessage | HumanMessage | SystemMessage, _info):
#         if isinstance(message, (AIMessage, HumanMessage, SystemMessage)):
#             return message.dict()
#         else:
#             return message


class ChatMessage(TypedDict):
    text: str
    role: Literal["user", "ai", "system"]

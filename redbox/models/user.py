from typing import Optional

import jwt

from redbox.models.base import PersistableModel


class User(PersistableModel):
    name: Optional[str]
    email: Optional[str]
    department: str
    role: str
    preferred_language: str

    def str_llm(self) -> str:
        return "\n".join(
            [
                f"{k}: {v}"
                for k, v in self.model_dump().items()
                if k not in ("uuid", "email", "name", "created_datetime", "creator_user_uuid")
            ]
        )

    def dict_llm(self) -> dict[str, str]:
        return {
            k: v
            for k, v in self.model_dump().items()
            if k not in ("uuid", "email", "name", "created_datetime", "creator_user_uuid")
        }

    def get_bearer_token(self, key: str) -> str:
        """the bearer token expected by the core-api"""
        bearer_token = jwt.encode({"user_uuid": str(self.uuid)}, key=key, algorithm="HS512")
        return f"Bearer {bearer_token}"

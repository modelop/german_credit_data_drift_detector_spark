import json
from typing import Dict

import requests


class Token:
    def __init__(self, token_dict: Dict) -> None:
        self.value = token_dict["access_token"]
        self.refresh_token = token_dict["refresh_token"]
        self.expires_in = token_dict["expires_in"]
        self.type = token_dict["token_type"]


class OAuth2(requests.auth.AuthBase):
    def __init__(self, token_url: str, username: str, password: str) -> None:
        self.token_url = token_url
        self.username = username
        self.password = password

        self.token = None
        self._set_token()

    def __call__(self, r: Dict) -> Dict:
        r.headers["Authorization"] = f"{self.token.type} {self.token.value}"
        return r

    def _set_token(self) -> None:
        payload = {"username": self.username, "password": self.password}
        response = requests.post(self.token_url, data=payload)
        if not response.ok:
            error_message = json.loads(response.text)["message"]
            raise ValueError(
                f"Could not authenticate with ModelOp Center; received {error_message}"
            )

        token_dict = json.loads(response.text)
        self.token = Token(token_dict)

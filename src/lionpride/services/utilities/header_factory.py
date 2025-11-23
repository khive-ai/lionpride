# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import SecretStr

AUTH_TYPES = Literal["bearer", "x-api-key", "none"]


class HeaderFactory:
    @staticmethod
    def get_content_type_header(
        content_type: str = "application/json",
    ) -> dict[str, str]:
        return {"Content-Type": content_type}

    @staticmethod
    def get_bearer_auth_header(api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"}

    @staticmethod
    def get_x_api_key_header(api_key: str) -> dict[str, str]:
        return {"x-api-key": api_key}

    @staticmethod
    def get_header(
        auth_type: AUTH_TYPES,
        content_type: str | None = "application/json",
        api_key: str | SecretStr | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        dict_ = {}
        if content_type is not None:
            dict_ = HeaderFactory.get_content_type_header(content_type)

        if auth_type == "none":
            # No authentication needed
            pass
        else:
            # Dereference SecretStr first, then validate
            if isinstance(api_key, SecretStr):
                api_key = api_key.get_secret_value()

            # Validate after dereferencing - check for None, empty, or whitespace
            if not api_key or not str(api_key).strip():
                raise ValueError("API key is required for authentication")

            # Strip whitespace for safety
            api_key = api_key.strip()

            if auth_type == "bearer":
                dict_.update(HeaderFactory.get_bearer_auth_header(api_key))
            elif auth_type == "x-api-key":
                dict_.update(HeaderFactory.get_x_api_key_header(api_key))
            else:
                raise ValueError(f"Unsupported auth type: {auth_type}")

        if default_headers:
            dict_.update(default_headers)
        return dict_

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    access_token: str = ""
    session_token: str = ""

    host: str = "0.0.0.0"
    port: int = 8100

    api_key: str = ""

    chatgpt_base_url: str = "https://chatgpt.com/backend-api"
    chatgpt_auth_url: str = "https://chatgpt.com/api/auth/session"

    # curl_cffi 用的代理地址 (Clash 默认 HTTP 代理 7890)
    # curl_cffi 通过 HTTP 代理 + Chrome impersonate 可以绕过 Cloudflare
    proxy: str = "http://127.0.0.1:7890"

    model_config = {"env_file": ".env", "env_prefix": "CHATGPT_"}


settings = Settings()

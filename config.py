"""
設定ファイルの読み込み (.env から環境変数として取得)
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from mcp import StdioServerParameters

# スクリプトのディレクトリにある .env を安全に読み込む
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    # .env が存在しない場合は警告出力（開発時に気づけるように）
    print("⚠️ 警告: .env ファイルが見つかりません。デフォルト設定を使用します。")


class Config:
    """環境設定を管理するクラス"""

    @staticmethod
    def get_lm_studio_url() -> str:
        host = os.getenv("LM_STUDIO_HOST", "127.0.0.1")
        port = os.getenv("LM_STUDIO_PORT", "1234")
        return f"http://{host}:{port}"

    @staticmethod
    def get_api_key() -> str:
        return os.getenv("LM_STUDIO_API_KEY", "")

    @staticmethod
    def get_api_endpoint() -> str:
        return f"{Config.get_lm_studio_url()}/v1/chat/completions"

    @staticmethod
    def get_models_endpoint() -> str:
        return f"{Config.get_lm_studio_url()}/v1/models"

    @staticmethod
    def get_auth_username() -> str:
        return os.getenv("APP_USERNAME", "admin")

    @staticmethod
    def get_auth_password() -> str:
        return os.getenv("APP_PASSWORD", "changeme")

    @staticmethod
    def get_app_host() -> str:
        return os.getenv("APP_HOST", "0.0.0.0")

    @staticmethod
    def get_app_port() -> int:
        return int(os.getenv("APP_PORT", "8021"))

    @staticmethod
    def is_authenticated(username: str, password: str) -> bool:
        return (
            username == Config.get_auth_username()
            and password == Config.get_auth_password()
        )

    # ─── MCP サーバー設定 ────────────────────────────────────────────────────
  
    MCP_COMMAND = r"uv"
    MCP_ARGS = [
        "run",
        "--directory",
        r"E:\MY-MCPSV-POSTGRES\MCP-SERVER",
        "python",
        "server.py",
    ]

    @staticmethod
    def get_mcp_server_params() -> StdioServerParameters:
        return StdioServerParameters(
            command=Config.MCP_COMMAND,
            args=Config.MCP_ARGS,
        )

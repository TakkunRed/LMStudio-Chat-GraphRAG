"""
LM Studio Chat - FastAPI アプリケーション
"""

import csv
import httpx
import io
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import config
from rag import GraphRAGManager


# ─── GraphRAG コンテキスト整形 ───────────────────────────────────────────

def _format_rag_context(raw_context: str) -> str:
    """
    LightRAG の only_need_context=True が返すコンテキストを、
    チャット LLM が読みやすい自然言語テキストに変換する。

    LightRAG (新形式) は以下のような JSON Lines ブロックを返す:
      Knowledge Graph Data (Entity):
      ```json
      {"entity": "...", "type": "...", "description": "..."}
      ...
      ```
      Knowledge Graph Data (Relation):
      ```json
      {"src_id": "...", "tgt_id": "...", "description": "...", "keywords": "...", "weight": ...}
      ...
      ```
      Knowledge Graph Data (Source):
      ```json
      {"id": "...", "content": "..."}
      ...
      ```

    旧形式 (-----Entities-----, CSV) にも対応する。
    """

    # ================================================================
    # ① JSON Lines 形式（新形式）を検出してパース
    # ================================================================
    def _extract_block(label_pattern: str) -> list[dict]:
        """'Knowledge Graph Data (Label):' に続く ```json ブロックをパース"""
        m = re.search(
            label_pattern + r"[^\n]*\n+```(?:json)?\s*\n(.*?)```",
            raw_context, re.DOTALL | re.IGNORECASE,
        )
        if not m:
            return []
        block = m.group(1).strip()
        rows: list[dict] = []
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return rows

    entity_rows   = _extract_block(r"Knowledge Graph Data\s*\(Entity\)")
    relation_rows = _extract_block(r"Knowledge Graph Data\s*\(Relation\)")
    source_rows   = _extract_block(r"Knowledge Graph Data\s*\(Source\)")
    report_rows   = _extract_block(r"Knowledge Graph Data\s*\(Report\)")

    result_parts: list[str] = []

    if entity_rows:
        result_parts.append("【エンティティ（登場人物・概念）】")
        for row in entity_rows:
            entity = str(row.get("entity") or "").strip()
            etype  = str(row.get("type")   or "").strip()
            desc   = str(row.get("description") or "").strip()
            s = f"・{entity}"
            if etype:
                s += f"（{etype}）"
            if desc:
                s += f": {desc}"
            result_parts.append(s)

    if relation_rows:
        result_parts.append("\n【関係性（ source → target: 説明 ）】")
        for row in relation_rows:
            # フィールド名が src_id/tgt_id の場合と source/target の場合がある
            src  = str(row.get("src_id")  or row.get("source") or "").strip()
            tgt  = str(row.get("tgt_id")  or row.get("target") or "").strip()
            desc = str(row.get("description") or "").strip()
            kw   = str(row.get("keywords")    or "").strip()
            s = f"・{src} → {tgt}: {desc}"
            if kw:
                s += f"（キーワード: {kw}）"
            result_parts.append(s)

    if report_rows:
        result_parts.append("\n【コミュニティ要約】")
        for row in report_rows:
            report = str(row.get("report") or row.get("content") or "").strip()
            if report:
                result_parts.append(f"・{report[:400]}")

    if source_rows:
        result_parts.append("\n【参照テキスト（原文抜粋）】")
        for row in source_rows[:5]:
            content = str(row.get("content") or "").strip()
            if content:
                result_parts.append(f"・{content[:300]}")

    if result_parts:
        formatted = "\n".join(result_parts)
        print(f"[GraphRAG] コンテキスト整形(JSON形式): {len(raw_context)}文字 → {len(formatted)}文字")
        return formatted

    # ================================================================
    # ② 旧形式 (-----Entities-----, CSV) へのフォールバック
    # ================================================================
    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in raw_context.split("\n"):
        m = re.match(r"^-{2,}\s*(\w+)\s*-{2,}$", line.strip())
        if m:
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = m.group(1)
            current_lines = []
        else:
            if current_key is not None:
                current_lines.append(line)
    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()

    def _extract_csv_block(text: str) -> str:
        m = re.search(r"```(?:csv)?\s*\n(.*?)```", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    csv_parts: list[str] = []

    for sec_key, label, src_col, tgt_col in [
        ("Entities",      "【エンティティ】",   None,     None),
        ("Relationships", "【関係性】",          "source", "target"),
        ("Reports",       "【コミュニティ要約】", None,    None),
        ("Sources",       "【参照テキスト】",    None,     None),
    ]:
        if sec_key not in sections:
            continue
        csv_text = _extract_csv_block(sections[sec_key])
        try:
            reader = csv.DictReader(io.StringIO(csv_text))
            rows = [r for r in reader if any((v or "").strip() for v in r.values())]
            if not rows:
                continue
            csv_parts.append(f"\n{label}")
            for row in rows[:50]:
                if sec_key == "Entities":
                    entity = (row.get("entity") or "").strip().strip('"')
                    etype  = (row.get("type")   or "").strip().strip('"')
                    desc   = (row.get("description") or "").strip().strip('"')
                    s = f"・{entity}"
                    if etype: s += f"（{etype}）"
                    if desc:  s += f": {desc}"
                    csv_parts.append(s)
                elif sec_key == "Relationships":
                    src  = (row.get(src_col or "source") or "").strip().strip('"')
                    tgt  = (row.get(tgt_col or "target") or "").strip().strip('"')
                    desc = (row.get("description") or "").strip().strip('"')
                    kw   = (row.get("keywords") or "").strip().strip('"')
                    s = f"・{src} → {tgt}: {desc}"
                    if kw: s += f"（キーワード: {kw}）"
                    csv_parts.append(s)
                else:
                    content = (row.get("report") or row.get("content") or "").strip().strip('"')
                    if content:
                        csv_parts.append(f"・{content[:300]}")
        except Exception:
            csv_parts.append(f"{label}\n" + csv_text[:300])

    if csv_parts:
        formatted = "\n".join(csv_parts)
        print(f"[GraphRAG] コンテキスト整形(CSV形式): {len(raw_context)}文字 → {len(formatted)}文字")
        return formatted

    # どちらにも該当しない場合はそのまま返す
    print("[GraphRAG] コンテキスト整形: 既知フォーマットなし、生データをそのまま使用")
    return raw_context


# ─── アプリケーション設定 ───────────────────────────────────────────────

app = FastAPI(title="LM Studio Chat")

# テンプレート・静的ファイルの設定
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# チャット履歴の保存先
HISTORY_FILE = BASE_DIR / "chat_history.json"

# ─── GraphRAG 初期化 ─────────────────────────────────────────────────────

RAG_DOCS_DIR    = BASE_DIR / "rag_docs"
RAG_CONFIG_FILE = BASE_DIR / "rag_config.json"
RAG_DOCS_DIR.mkdir(exist_ok=True)

_SEARCH_MODE_DEFAULT = "hybrid"

def load_rag_config() -> dict:
    """rag_config.json を読み込む。存在しない場合はデフォルト値を返す"""
    if RAG_CONFIG_FILE.exists():
        try:
            cfg = json.loads(RAG_CONFIG_FILE.read_text(encoding="utf-8"))
            return {
                "search_mode":       cfg.get("search_mode", _SEARCH_MODE_DEFAULT),
                "llm_model":         cfg.get("llm_model", ""),
                "normalize_on_insert": cfg.get("normalize_on_insert", False),
            }
        except Exception:
            pass
    return {"search_mode": _SEARCH_MODE_DEFAULT, "llm_model": "", "normalize_on_insert": False}

rag_manager: GraphRAGManager | None = None

@app.on_event("startup")
async def startup_event():
    global rag_manager
    try:
        cfg = load_rag_config()
        base_url = f"http://{os.getenv('LM_STUDIO_HOST', '127.0.0.1')}:{os.getenv('LM_STUDIO_PORT', '1234')}/v1"
        # LLM モデルの優先順位: rag_config.json（UI保存値）> 環境変数 > 自動検出
        # rag_config.json を優先することで、設定画面での変更がサーバー再起動後も維持される
        llm_model = cfg.get("llm_model", "") or os.getenv("LM_STUDIO_LLM_MODEL", "")
        rag_manager = GraphRAGManager(
            working_dir=str(BASE_DIR / "lightrag_db"),
            lm_studio_base_url=base_url,
            llm_model=llm_model,
            embedding_model=os.getenv("LIGHTRAG_EMBED_MODEL", "nomic-embed-text"),
            embedding_dim=int(os.getenv("LIGHTRAG_EMBED_DIM", "768")),
            api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
            search_mode=cfg["search_mode"],
            normalize_on_insert=cfg.get("normalize_on_insert", False),
        )
        await rag_manager.initialize()
        print(f"[GraphRAG] 初期化完了 - {rag_manager.get_status()}")
    except Exception as e:
        print(f"[GraphRAG] 初期化失敗（RAG機能は無効）: {e}")
        rag_manager = None

# ─── モデル定義 ─────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 8196


class LoginRequest(BaseModel):
    username: str
    password: str


class SettingsUpdate(BaseModel):
    lm_studio_host: str
    lm_studio_port: int
    app_username: str
    app_password: str
    lm_studio_api_key: str


# ─── チャット履歴管理 ───────────────────────────────────────────────────


def load_history() -> dict:
    """チャット履歴を読み込む"""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_history(history: dict) -> None:
    """チャット履歴を保存する"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def get_session_history(session_id: str) -> list[dict]:
    """セッションの履歴を取得"""
    history = load_history()
    return history.get(session_id, [])


def append_to_history(session_id: str, role: str, content: str) -> None:
    """履歴に追加"""
    history = load_history()
    if session_id not in history:
        history[session_id] = []

    history[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

    # 履歴は最新100件に制限
    if len(history[session_id]) > 100:
        history[session_id] = history[session_id][-100:]

    save_history(history)


def clear_session_history(session_id: str) -> None:
    """セッション履歴をクリア"""
    history = load_history()
    if session_id in history:
        del history[session_id]
    save_history(history)


# ─── LM Studio API 通信 ─────────────────────────────────────────────────


async def fetch_models() -> list[dict]:
    """LM Studio から利用可能なモデル一覧を取得"""
    try:
        api_key = config.Config.get_api_key()
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                config.Config.get_models_endpoint(),
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
    except Exception as e:
        print(f"モデル取得エラー: {e}")
        return []


async def send_to_lm_studio(
    messages: list[dict],
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 8196
) -> str:
    """LM Studio にチャットリクエストを送信"""
    api_key = config.Config.get_api_key()
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model:
        payload["model"] = model

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                config.Config.get_api_endpoint(),
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except httpx.TimeoutException:
        return "⚠️ タイムアウトしました。LM Studio サーバーが実行中か確認してください。"
    except httpx.HTTPStatusError as e:
        return f"⚠️ HTTPエラー: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"⚠️ エラーが発生しました: {str(e)}"


# ─── 認証ミドルウェア ───────────────────────────────────────────────────


def check_auth(request: Request) -> Optional[str]:
    """セッション認証をチェック"""
    session = request.cookies.get("session_token")
    if not session:
        return None
    return session  # 簡易的にトークン自体をユーザーIDとして使用


# ─── ルート ─────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """ログインページ / チャットページ"""
    session = request.cookies.get("session_token")
    
    if session:
        # ✅ request を context 外に明示的に指定
        return templates.TemplateResponse(
            name="chat.html",
            context={
                "authenticated": True,
                "session_id": session,
                "history": get_session_history(session),
                "models": await fetch_models(),
                "lm_studio_url": config.Config.get_lm_studio_url(),
            },
            request=request
        )
    
    return templates.TemplateResponse(
        name="login.html",
        context={
            "authenticated": False,
            "error": "ユーザー名またはパスワードが異なります。",
        },
        request=request
    )

@app.post("/login")
async def login(request: Request):
    """ログイン処理"""
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")

    if config.Config.is_authenticated(username, password):
        response = HTMLResponse(
            """<script>window.location.href='/';</script>"""
        )
        response.set_cookie(
            key="session_token",
            value=f"user_{username}_{datetime.now().timestamp()}",
            httponly=True,
            max_age=86400,
        )
        return response

    return templates.TemplateResponse("login.html", {
        "request": request,
        "authenticated": False,
        "error": "ユーザー名またはパスワードが異なります。",
    })


@app.get("/logout")
async def logout(request: Request):
    """ログアウト"""
    response = HTMLResponse("""<script>window.location.href='/';</script>""")
    response.delete_cookie(key="session_token")
    return response


@app.get("/api/models")
async def api_models(request: Request):
    """モデル一覧取得 API"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    models = await fetch_models()
    return JSONResponse({"models": models})


@app.post("/api/chat")
async def api_chat(request: Request):
    """チャットリクエスト API（RAG + ツール対応版）"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "")
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 8192)
    tools = body.get("tools", [])
    use_rag = body.get("use_rag", False)

    if not messages:
        return JSONResponse({"error": "メッセージが空です"}, status_code=400)

    # GraphRAG: ユーザーの最後の発言でグラフ検索してシステムメッセージに注入
    rag_sources = []
    if use_rag and rag_manager:
        user_query = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        if user_query:
            hits = await rag_manager.search(user_query)
            if hits:
                # only_need_context=True が返す CSV を読みやすい箇条書きに変換
                raw_context  = "\n\n".join(h["content"] for h in hits)
                context_text = _format_rag_context(raw_context)
                rag_system = {
                    "role": "system",
                    "content": (
                        "[STRICT INSTRUCTION]\n"
                        "Answer the user's question using ONLY the knowledge graph data provided below.\n"
                        "The data lists:\n"
                        "  - 【エンティティ】: people, places, concepts with their descriptions\n"
                        "  - 【関係性】: directed relationships in the form 'A → B: description'\n"
                        "    (meaning A is related to B as described; e.g. '孫悟空 → 孫悟飯: 父子関係' means "
                        "孫悟空 is the parent of 孫悟飯)\n"
                        "  - 【参照テキスト】: original text excerpts\n\n"
                        "Rules:\n"
                        "1. Use ONLY facts explicitly listed in the data below.\n"
                        "2. Relationship entries '孫悟空 → 孫悟飯: 父子関係' mean 孫悟空 is the PARENT "
                        "and 孫悟飯 is the CHILD. Read directionality carefully.\n"
                        "3. Do NOT use your training knowledge or prior information about the topic.\n"
                        "4. If the answer is not in the data, say '提供されたデータに記載がありません'.\n"
                        "5. Do NOT infer or guess relationships not explicitly stated.\n\n"
                        "【知識グラフデータ】\n"
                        f"{context_text}"
                    ),
                }
                # 既存のシステムメッセージがあれば RAG コンテキストを先頭に追加
                messages = [rag_system] + messages
                rag_sources = [{"source": h["source"], "score": h["score"]} for h in hits]

    assistant_reply = await chat_with_tools(
        messages=messages,
        tools=tools,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    append_to_history(session, "user", messages[-1]["content"])
    append_to_history(session, "assistant", assistant_reply)

    return JSONResponse({
        "reply": assistant_reply,
        "history": get_session_history(session),
        "rag_sources": rag_sources,
    })


# ─── MCP ツール実行 ──────────────────────────────────────────────────────────

async def call_mcp_tool(tool_name: str, tool_args: dict) -> str:
    """
    stdio MCP サーバーを起動してツールを呼び出し、結果を文字列で返す。
    呼び出しのたびに新しいプロセスを起動する（ステートレス）。
    """
    try:
        async with stdio_client(config.Config.get_mcp_server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, tool_args)
                # result.content はリスト。TextContent を結合して返す
                parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    else:
                        parts.append(str(block))
                return "\n".join(parts)
    except Exception as e:
        return json.dumps({"error": f"MCPツール呼び出しエラー: {str(e)}"}, ensure_ascii=False)


async def chat_with_tools(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    max_tool_iterations: int = 5
) -> str:
    """
    ツール呼び出しに対応したチャット処理。
    tool_calls が返ってきたら MCP サーバーを実際に呼び出して
    tool_result を組み立て、LM Studio に再送して最終応答を得る。
    """
    current_messages = list(messages)
    tool_definitions = list(tools) if tools else []

    api_key = config.Config.get_api_key()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for iteration in range(max_tool_iterations):
        payload = {
            "messages": current_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model:
            payload["model"] = model
        if tool_definitions:
            payload["tools"] = tool_definitions
            payload["tool_choice"] = "auto"

        print(f"[chat_with_tools] iteration={iteration}, messages={len(current_messages)}, tools={len(tool_definitions)}")
        print(f"[chat_with_tools] payload={json.dumps(payload, ensure_ascii=False)[:600]}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    config.Config.get_api_endpoint(),
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            return f"⚠️ HTTPエラー: {e.response.status_code} - {e.response.text}"
        except Exception as e:
            return f"⚠️ API通信エラー: {str(e)}"

        print(f"[chat_with_tools] response={json.dumps(data, ensure_ascii=False)[:600]}")

        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason", "")
        assistant_msg = choice["message"]
        tool_calls = assistant_msg.get("tool_calls") or []

        # ── ツール呼び出しなし → 最終応答 ─────────────────────────────
        if not tool_calls:
            content = assistant_msg.get("content")
            return content if content is not None else ""

        # ── ツール呼び出しあり → MCP サーバーを実際に呼ぶ ─────────────
        # assistant メッセージを履歴に追加
        current_messages.append({
            "role": "assistant",
            "content": assistant_msg.get("content"),
            "tool_calls": tool_calls,
        })

        # 各 tool_call を MCP で実行し tool_result を追加
        for call in tool_calls:
            func_name = call["function"]["name"]
            raw_args = call["function"].get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    func_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    func_args = {}
            else:
                func_args = raw_args

            # None 文字列を実際の None に変換（モデルが "None" を渡してくる場合の対策）
            sanitized_args = {
                k: (None if v in (None, "None", "null", "") else v)
                for k, v in func_args.items()
            }

            print(f"[chat_with_tools] calling MCP tool: {func_name}({sanitized_args})")
            result_content = await call_mcp_tool(func_name, sanitized_args)
            print(f"[chat_with_tools] MCP result: {result_content[:300]}")

            current_messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result_content,
            })

    return "⚠️ ツール呼び出しの最大反復回数を超えました。"

@app.get("/api/history/{session_id}")
async def api_get_history(session_id: str, request: Request):
    """履歴取得 API"""
    _ = check_auth(request)  # 認証チェック（トークンベース）
    return JSONResponse({"history": get_session_history(session_id)})


@app.post("/api/clear-history")
async def api_clear_history(request: Request):
    """履歴クリア API"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    clear_session_history(session)
    return JSONResponse({"status": "cleared"})


@app.get("/api/settings")
async def api_get_settings(request: Request):
    """設定取得 API"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    return JSONResponse({
        "lm_studio_host": os.getenv("LM_STUDIO_HOST", "127.0.0.1"),
        "lm_studio_port": os.getenv("LM_STUDIO_PORT", "1234"),
        "has_api_key": bool(os.getenv("LM_STUDIO_API_KEY", "")),
        "app_username": os.getenv("APP_USERNAME", "admin"),
    })


@app.post("/api/update-settings")
async def api_update_settings(request: Request):
    """設定更新 API（変更キーのみ上書き、未知のキーは保持）"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    env_path = BASE_DIR / ".env"

    # リクエストから更新するキーと値を組み立てる（空文字は除外しない）
    updates: dict[str, str] = {
        "LM_STUDIO_HOST": str(body.get("lm_studio_host", "")),
        "LM_STUDIO_PORT": str(body.get("lm_studio_port", "")),
        "LM_STUDIO_API_KEY": str(body.get("lm_studio_api_key", "")),
        "APP_USERNAME": str(body.get("app_username", "")),
    }
    # パスワードは空送信の場合は更新しない
    if body.get("app_password"):
        updates["APP_PASSWORD"] = str(body["app_password"])

    # 既存ファイルを行単位で読み込み、該当キーだけ置換する
    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    updated_keys: set[str] = set()
    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        # コメント・空行はそのまま保持
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        # KEY=VALUE 形式のみ処理
        if "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)

    # ファイルに存在しなかったキーは末尾に追記
    for key, value in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    load_dotenv(env_path, override=True)

    return JSONResponse({"status": "updated", "message": "設定を更新しました。サーバーを再起動してください。"})


# ─── Function Calling ツール設定の保存・読み込み ─────────────────────────

TOOLS_CONFIG_FILE = BASE_DIR / "tools_config.json"

@app.get("/api/tools")
async def api_get_tools(request: Request):
    """保存済みツール設定を取得"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    if TOOLS_CONFIG_FILE.exists():
        tools = json.loads(TOOLS_CONFIG_FILE.read_text(encoding="utf-8"))
    else:
        tools = []
    return JSONResponse({"tools": tools})


@app.post("/api/tools")
async def api_save_tools(request: Request):
    """ツール設定を保存"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    tools = body.get("tools", [])
    TOOLS_CONFIG_FILE.write_text(
        json.dumps(tools, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return JSONResponse({"status": "saved", "count": len(tools)})


# ─── RAG エンドポイント ──────────────────────────────────────────────────

@app.get("/api/rag/config")
async def rag_get_config(request: Request):
    """GraphRAG 設定を取得"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    cfg = load_rag_config()
    if rag_manager:
        status = rag_manager.get_status()
        cfg["search_mode"]       = rag_manager.search_mode
        cfg["llm_model"]         = status.get("llm_model", "")
        cfg["normalize_on_insert"] = status.get("normalize_on_insert", False)
    return JSONResponse(cfg)


@app.post("/api/rag/config")
async def rag_save_config(request: Request):
    """GraphRAG 設定（検索モード・LLM モデル）を保存"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    mode = body.get("search_mode", _SEARCH_MODE_DEFAULT)
    if mode not in {"naive", "local", "global", "hybrid"}:
        return JSONResponse({"error": "search_mode は naive/local/global/hybrid のいずれかです"}, status_code=400)

    llm_model        = str(body.get("llm_model", "")).strip()
    normalize_on_insert = bool(body.get("normalize_on_insert", False))

    # rag_config.json に保存
    RAG_CONFIG_FILE.write_text(
        json.dumps(
            {"search_mode": mode, "llm_model": llm_model, "normalize_on_insert": normalize_on_insert},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 実行中の rag_manager に反映
    if rag_manager:
        rag_manager.search_mode = mode
        rag_manager._config["normalize_on_insert"] = normalize_on_insert
        current_model = rag_manager.get_status().get("llm_model", "")
        if llm_model != current_model:
            await rag_manager.set_llm_model(llm_model)

    return JSONResponse({
        "status": "saved",
        "search_mode": mode,
        "llm_model": llm_model,
        "normalize_on_insert": normalize_on_insert,
    })


@app.get("/api/rag/status")
async def rag_status(request: Request):
    """インデックスの状態確認"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)
    return JSONResponse(rag_manager.get_status())


@app.post("/api/rag/upload")
async def rag_upload(request: Request, file: UploadFile = File(...)):
    """ファイルをアップロードしてインデックスに追加"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".pdf"}:
        return JSONResponse({"error": f"未対応の形式: {suffix}"}, status_code=400)

    dest = RAG_DOCS_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = await rag_manager.add_document(dest)
    return JSONResponse(result)


@app.post("/api/rag/index-dir")
async def rag_index_dir(request: Request):
    """rag_docs/ ディレクトリ内の全ファイルを再インデックス"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    results = await rag_manager.add_directory(RAG_DOCS_DIR)
    success = sum(1 for r in results if r.get("success"))
    return JSONResponse({
        "total": len(results),
        "success": success,
        "failed": len(results) - success,
        "details": results,
    })


@app.delete("/api/rag/document/{file_name}")
async def rag_delete_document(file_name: str, request: Request):
    """指定ファイルをインデックスから削除"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    result = await rag_manager.delete_document(file_name)
    return JSONResponse(result)


@app.delete("/api/rag/clear")
async def rag_clear(request: Request):
    """インデックスを全クリア"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    result = await rag_manager.clear()
    return JSONResponse(result)


@app.post("/api/rag/search")
async def rag_search(request: Request):
    """RAG検索のテスト用エンドポイント"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    body = await request.json()
    query = body.get("query", "")
    n_results = body.get("n_results", 5)
    if not query:
        return JSONResponse({"error": "queryが空です"}, status_code=400)

    hits = await rag_manager.search(query)
    return JSONResponse({"results": hits})


# ─── GraphRAG 可視化エンドポイント ──────────────────────────────────────

GRAPHML_FILE = BASE_DIR / "lightrag_db" / "graph_chunk_entity_relation.graphml"
_GRAPHML_NS  = "http://graphml.graphdrawing.org/xmlns"


def _parse_graphml(limit: int = 300) -> dict:
    """
    graphml ファイルを解析してノード・エッジ一覧を返す。
    limit > 0 のとき、次数（接続エッジ数）上位 limit 件のノードに絞って返す。
    limit = 0 のとき全件返す（グラフが大きい場合はブラウザが重くなる）。
    """
    if not GRAPHML_FILE.exists():
        return {"nodes": [], "edges": [], "stats": {"node_count": 0, "edge_count": 0, "total_nodes": 0, "total_edges": 0}}

    tree = ET.parse(str(GRAPHML_FILE))
    root = tree.getroot()
    ns = {"g": _GRAPHML_NS}

    # <key> 要素からキー名 → id のマッピングを構築
    node_key_map: dict[str, str] = {}
    edge_key_map: dict[str, str] = {}
    for key_el in root.findall("g:key", ns):
        kid   = key_el.get("id", "")
        kfor  = key_el.get("for", "")
        kname = key_el.get("attr.name", "")
        if kfor == "node":
            node_key_map[kname] = kid
        elif kfor == "edge":
            edge_key_map[kname] = kid

    graph_el = root.find("g:graph", ns)
    if graph_el is None:
        return {"nodes": [], "edges": [], "stats": {"node_count": 0, "edge_count": 0, "total_nodes": 0, "total_edges": 0}}

    # ── 全ノードを解析 ──────────────────────────────────────────────
    all_nodes: list[dict] = []
    for node_el in graph_el.findall("g:node", ns):
        node_id = node_el.get("id", "")
        data: dict[str, str] = {
            d.get("key", ""): (d.text or "").strip()
            for d in node_el.findall("g:data", ns)
        }
        entity_type = data.get(node_key_map.get("entity_type", ""), "") or "その他"
        description = data.get(node_key_map.get("description", ""), "")
        all_nodes.append({
            "id":          node_id,
            "label":       node_id,
            "entity_type": entity_type,
            "description": description,
        })

    # ── 全エッジを解析 ──────────────────────────────────────────────
    all_edges: list[dict] = []
    for i, edge_el in enumerate(graph_el.findall("g:edge", ns)):
        src = edge_el.get("source", "")
        tgt = edge_el.get("target", "")
        data: dict[str, str] = {
            d.get("key", ""): (d.text or "").strip()
            for d in edge_el.findall("g:data", ns)
        }
        weight_raw  = data.get(edge_key_map.get("weight", ""), "1.0")
        keywords    = data.get(edge_key_map.get("keywords", ""), "")
        description = data.get(edge_key_map.get("description", ""), "")
        try:
            weight = float(weight_raw)
        except ValueError:
            weight = 1.0
        all_edges.append({
            "id":          i,
            "from":        src,
            "to":          tgt,
            "label":       keywords[:30] if keywords else "",
            "keywords":    keywords,
            "description": description,
            "weight":      weight,
        })

    total_nodes = len(all_nodes)
    total_edges = len(all_edges)

    # ── 次数フィルタリング ───────────────────────────────────────────
    if limit > 0 and total_nodes > limit:
        # 各ノードの接続エッジ数を集計
        degree: dict[str, int] = {n["id"]: 0 for n in all_nodes}
        for e in all_edges:
            degree[e["from"]] = degree.get(e["from"], 0) + 1
            degree[e["to"]]   = degree.get(e["to"],   0) + 1

        # 次数上位 limit 件のノード ID セット
        top_ids: set[str] = {
            nid for nid, _ in sorted(degree.items(), key=lambda x: -x[1])[:limit]
        }

        nodes = [n for n in all_nodes if n["id"] in top_ids]
        edges = [e for e in all_edges if e["from"] in top_ids and e["to"] in top_ids]
    else:
        nodes = all_nodes
        edges = all_edges

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count":  len(nodes),
            "edge_count":  len(edges),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
        },
    }


@app.get("/graph", response_class=HTMLResponse)
async def graph_page(request: Request):
    """GraphRAG ナレッジグラフ可視化ページ"""
    session = request.cookies.get("session_token")
    if not session:
        return HTMLResponse("""<script>window.location.href='/';</script>""")
    return templates.TemplateResponse(
        name="graph.html",
        context={},
        request=request,
    )


@app.get("/api/graph/data")
async def api_graph_data(request: Request, limit: int = 300):
    """
    グラフデータを返す API（vis.js 形式）
    limit: 表示するノード数上限（次数降順。0=全件）
    """
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    limit = max(0, min(limit, 5000))   # 上限 5000
    try:
        data = _parse_graphml(limit=limit)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─── GraphRAG デバッグ画面 ──────────────────────────────────────────────


@app.get("/debug", response_class=HTMLResponse)
async def debug_page(request: Request):
    """GraphRAG チャンク抽出デバッグ画面"""
    session = request.cookies.get("session_token")
    if not session:
        return HTMLResponse("""<script>window.location.href='/';</script>""")
    return templates.TemplateResponse(
        name="debug_rag.html",
        context={},
        request=request,
    )


class DebugExtractRequest(BaseModel):
    file_content: str
    max_chars: int = 200
    overlap_sentences: int = 1
    max_chunks: int = 5
    model: str = ""        # 空 = RAG マネージャーのデフォルトモデル
    normalize: bool = False  # True = 抽出前に複合文 → 単純文に変換


@app.post("/api/rag/debug/read-pdf")
async def api_rag_debug_read_pdf(request: Request, file: UploadFile = File(...)):
    """デバッグ画面用: PDF ファイルをテキストとして返す"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    try:
        import pypdf, io
        data = await file.read()
        reader = pypdf.PdfReader(io.BytesIO(data))
        content = "\n".join(page.extract_text() or "" for page in reader.pages)
        return JSONResponse({"content": content, "chars": len(content)})
    except ImportError:
        return JSONResponse({"error": "pypdf がインストールされていません"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/rag/debug/run")
async def api_rag_debug_run(request: Request, body: DebugExtractRequest):
    """
    チャンクごとのLLM入出力をSSEでストリーミング返却するデバッグAPI。

    レスポンス形式: text/event-stream
    各イベントは改行区切りJSON ("data: {...}\\n\\n")
    """
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        raise HTTPException(status_code=503, detail="RAG マネージャーが初期化されていません")

    max_chars = max(50, min(body.max_chars, 2000))
    overlap   = max(0, min(body.overlap_sentences, 5))
    max_chk   = max(1, min(body.max_chunks, 20))

    async def event_stream():
        try:
            async for result in rag_manager.debug_extract(
                file_content=body.file_content,
                max_chars=max_chars,
                overlap_sentences=overlap,
                max_chunks=max_chk,
                model=body.model.strip(),
                normalize=body.normalize,
            ):
                yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── メイン ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.Config.get_app_host(),
        port=config.Config.get_app_port(),
        reload=True,
    )

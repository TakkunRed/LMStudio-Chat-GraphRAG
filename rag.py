"""
GraphRAG モジュール (LightRAG)
- LightRAG: グラフベースの RAG（エンティティ・関係グラフを構築）
- LM Studio: LLM（エンティティ抽出・クエリ処理）とエンベディング
- 対応形式: .txt, .pdf
"""

import json
import shutil
from pathlib import Path

import httpx
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc


def _patch_lightrag_handlers() -> None:
    """
    LightRAG 内部の entity/relation 抽出ハンドラーにフィールド数正規化を注入する。

    【背景】
    LLM 出力を _fix_llm_output で修正しても、LightRAG 内部の
    fix_tuple_delimiter_corruption() が後から追加の <|#|> を生成する場合がある。
    その結果、WARNING "found N/4 fields on ENTITY" / "found N/5 fields on RELATION"
    が発生し、エンティティ・関係がグラフに登録されない。

    【対処】
    WARNING が emitted される直前の _handle_single_entity_extraction /
    _handle_single_relationship_extraction をラップし、フィールド数を強制的に
    正規化してから元のハンドラーを呼ぶ。これにより原因に依らず WARNING を排除できる。

    【フィールド構造】
    ENTITY  (期待4): entity | name | type | description
    RELATION(期待5): relation | src | tgt | keywords | description
    """
    import lightrag.operate as _op

    _orig_entity   = _op._handle_single_entity_extraction        # type: ignore[attr-defined]
    _orig_relation = _op._handle_single_relationship_extraction  # type: ignore[attr-defined]

    def _fixed_entity(record_attributes, chunk_key, timestamp, file_path="unknown_source"):
        n = len(record_attributes)
        if n > 1 and "entity" in record_attributes[0]:
            if n > 4:
                # フィールド超過: 余分なフィールドを description に結合
                record_attributes = record_attributes[:3] + [" ".join(record_attributes[3:])]
            elif n < 4:
                # フィールド不足: "-" で補完
                record_attributes = record_attributes + ["-"] * (4 - n)
        return _orig_entity(record_attributes, chunk_key, timestamp, file_path)

    def _fixed_relation(record_attributes, chunk_key, timestamp, file_path="unknown_source"):
        n = len(record_attributes)
        if n > 1 and "relation" in record_attributes[0]:
            if n > 5:
                # フィールド超過: 余分なフィールドを description に結合
                record_attributes = record_attributes[:4] + [" ".join(record_attributes[4:])]
            elif n == 4:
                # keywords が欠落: description は正しい位置に保ち keywords を "-" で補完
                record_attributes = record_attributes[:3] + ["-"] + [record_attributes[3]]
            elif n < 4:
                # フィールド大幅不足: "-" で末尾補完
                record_attributes = record_attributes + ["-"] * (5 - n)
        return _orig_relation(record_attributes, chunk_key, timestamp, file_path)

    _op._handle_single_entity_extraction       = _fixed_entity    # type: ignore[attr-defined]
    _op._handle_single_relationship_extraction = _fixed_relation   # type: ignore[attr-defined]
    print("[GraphRAG] LightRAG ハンドラーにフィールド数正規化パッチを適用しました")


# モジュールロード時に一度だけ適用
_patch_lightrag_handlers()


def _make_sentence_chunker(
    max_chars: int = 200,
    overlap_sentences: int = 1,
):
    """
    段落（空行区切り）を優先し、文末（。！？）で補助的に分割するカスタムチャンキング関数を生成する。

    【分割の優先順位】
    1. 空行（\\n\\n）で段落に分割 → 1段落が max_chars 以内なら 1 チャンクとして出力
    2. 段落が長い場合は 。！？ で文に分割して積み重ね、超えたらチャンク確定
    3. 段落内で分割する場合、段落の先頭文（主語・話題を含む）を次チャンクに引き継ぐ

    【なぜ段落優先か】
    百科事典・人物紹介・仕様書などは「1エンティティ = 1段落」の構造が多い。
    段落をまたいでオーバーラップすると直前エンティティの記述が別エンティティの
    チャンクに混入し、LLM が主語を誤認識して抽出精度が下がる。

    【段落内分割のオーバーラップ戦略】
    - 段落の先頭文（主語文）を常に引き継ぐ
    - さらに直前チャンク末尾の overlap_sentences 文も引き継ぐ（重複は除く）
    - 1文が max_chars を超える場合はその文単独で 1 チャンクとする

    Args:
        max_chars: チャンクの最大文字数（デフォルト 200 文字）
                   LM Studio の Context Length が 4096 なら 150〜200、8192 以上なら 300〜400 が目安
        overlap_sentences: 段落内分割時に直前チャンクから引き継ぐ文の数（デフォルト 1 文）
                           ※先頭文は常に引き継がれるため、これは追加オーバーラップ数

    Returns:
        LightRAG の chunking_func インターフェースと互換な async 関数
    """
    import re as _re

    _PARA_SEP   = _re.compile(r"\n{2,}")          # 段落区切り（空行）
    _SENT_SPLIT = _re.compile(r"(?<=[。！？])\s*") # 文末区切り（改行は別処理済み）

    async def _chunk_by_paragraph(
        tokenizer,                              # LightRAG が渡す Tokenizer オブジェクト
        content: str,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        chunk_overlap_token_size: int = 100,
        chunk_token_size: int = 1200,
    ) -> list[dict]:
        """段落優先・文末補助のチャンキング関数（LightRAG chunking_func 互換）"""
        if not content or not content.strip():
            return []

        def _tok(text: str) -> int:
            """Tokenizer が使えればトークン数を実測、なければ文字数 // 2 で推定"""
            try:
                return max(1, len(tokenizer.encode(text)))
            except Exception:
                return max(1, len(text) // 2)

        def _flush(sentences: list[str], idx: int) -> dict:
            text = "".join(sentences)
            return {"tokens": _tok(text), "content": text, "chunk_order_index": idx}

        # ── まず空行で段落に分割 ─────────────────────────────────────────
        paragraphs: list[str] = [
            p.strip() for p in _PARA_SEP.split(content) if p.strip()
        ]
        if not paragraphs:
            return []

        chunks: list[dict] = []
        chunk_idx = 0

        for para in paragraphs:
            # ── 段落全体が max_chars に収まる → そのまま 1 チャンク ──────
            if len(para) <= max_chars:
                chunks.append(_flush([para], chunk_idx))
                chunk_idx += 1
                continue

            # ── 長い段落 → 文末で分割 ──────────────────────────────────
            sentences: list[str] = [
                s.strip() for s in _SENT_SPLIT.split(para) if s.strip()
            ]
            # 先頭文（主語・話題を含む文）を記憶しておく
            first_sent: str = sentences[0] if sentences else ""

            current: list[str] = []
            current_len = 0
            is_continuation = False  # 2チャンク目以降かどうか

            for sent in sentences:
                sent_len = len(sent)

                # 単一文が max_chars を超える → 単独チャンクとして出力
                if not current and sent_len > max_chars:
                    chunks.append(_flush([sent], chunk_idx))
                    chunk_idx += 1
                    # 次チャンクのオーバーラップ: 先頭文（主語）を引き継ぐ
                    if first_sent and first_sent != sent:
                        current = [first_sent]
                        current_len = len(first_sent)
                    is_continuation = True
                    continue

                # 追加するとオーバー → 現在チャンクを確定
                if current and current_len + sent_len > max_chars:
                    chunks.append(_flush(current, chunk_idx))
                    chunk_idx += 1

                    # オーバーラップ構築:
                    #   ① 先頭文（主語）が current に含まれていなければ先頭に追加
                    #   ② 直前チャンクの末尾 overlap_sentences 文を引き継ぐ
                    tail = current[-overlap_sentences:] if overlap_sentences > 0 else []
                    if first_sent and first_sent not in tail:
                        new_current = [first_sent] + [s for s in tail if s != first_sent]
                    else:
                        new_current = tail
                    current = new_current
                    current_len = sum(len(s) for s in current)
                    is_continuation = True

                current.append(sent)
                current_len += sent_len

            # 段落の残り
            if current:
                chunks.append(_flush(current, chunk_idx))
                chunk_idx += 1

        print(
            f"[GraphRAG] 段落優先チャンキング完了: {len(chunks)} チャンク"
            f"（段落数: {len(paragraphs)}, 最大 {max_chars} 文字）"
        )
        return chunks

    return _chunk_by_paragraph


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}
SEARCH_MODES = {"naive", "local", "global", "hybrid"}


class GraphRAGManager:
    def __init__(
        self,
        working_dir: str = "./lightrag_db",
        lm_studio_base_url: str = "http://127.0.0.1:1234/v1",
        llm_model: str = "",
        embedding_model: str = "nomic-embed-text",
        embedding_dim: int = 768,
        api_key: str = "lm-studio",
        search_mode: str = "hybrid",
        chunk_max_chars: int = 200,
        chunk_overlap_sentences: int = 1,
    ):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self._docs_index_file = self.working_dir / "_indexed_docs.json"
        self.search_mode = search_mode if search_mode in SEARCH_MODES else "hybrid"

        self._config = {
            "lm_studio_base_url": lm_studio_base_url,
            "llm_model": llm_model or "",   # 空のまま維持し initialize() で自動検出
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
            "api_key": api_key or "lm-studio",
            # 文末チャンキング設定
            "chunk_max_chars": max(50, chunk_max_chars),  # チャンク最大文字数
            "chunk_overlap_sentences": max(0, chunk_overlap_sentences),  # オーバーラップ文数
        }

        # インデックス済みドキュメントの追跡: {ファイル名: フルパス}
        self._docs: dict[str, str] = self._load_docs_index()

        self._rag = self._create_rag_instance()

    # ── エンベディング次元数の自動検出 ────────────────────────────────────

    async def _detect_embedding_dim(self) -> int:
        """LM Studio に1件テストリクエストを送り、実際の次元数を返す"""
        cfg = self._config
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{cfg['lm_studio_base_url']}/embeddings",
                    headers={
                        "Authorization": f"Bearer {cfg['api_key']}",
                        "Content-Type": "application/json",
                    },
                    json={"model": cfg["embedding_model"], "input": "test"},
                )
                resp.raise_for_status()
                dim = len(resp.json()["data"][0]["embedding"])
                print(f"[GraphRAG] エンベディング次元数を自動検出: {dim}")
                return dim
        except Exception as e:
            print(f"[GraphRAG] 次元数の自動検出に失敗（設定値 {cfg['embedding_dim']} を使用）: {e}")
            return cfg["embedding_dim"]

    async def _detect_llm_model(self) -> str:
        """LM Studio の /v1/models から現在ロードされているモデル ID を取得"""
        cfg = self._config
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{cfg['lm_studio_base_url']}/models",
                    headers={"Authorization": f"Bearer {cfg['api_key']}"},
                )
                resp.raise_for_status()
                models = resp.json().get("data", [])
                if models:
                    model_id = models[0]["id"]
                    print(f"[GraphRAG] LLM モデルを自動検出: {model_id}")
                    return model_id
        except Exception as e:
            print(f"[GraphRAG] LLM モデルの自動検出失敗: {e}")
        return ""

    # ── LightRAG インスタンス生成 ─────────────────────────────────────────

    def _create_rag_instance(self) -> LightRAG:
        cfg = self._config
        base_url = cfg["lm_studio_base_url"]
        llm_model = cfg["llm_model"]
        embed_model = cfg["embedding_model"]
        api_key = cfg["api_key"]
        embed_dim = cfg["embedding_dim"]

        async def llm_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            keyword_extraction=False,
            **kwargs,  # response_format 等は無視して LM Studio に送らない
        ) -> str:
            """LM Studio の /chat/completions に直接リクエスト（構造化出力なし）"""

            def _sanitize(text: str) -> str:
                """
                LightRAG のエンティティ抽出結果に含まれる <|#|> などの区切り子を
                LM Studio（llama.cpp）が特殊トークンとして誤認識しないよう置換する。

                gleaning ステップで前回の抽出結果（entity<|#|>名前<|#|>...）が
                  ① history_messages のアシスタントメッセージ
                  ② prompt パラメーター自体
                に含まれる場合に 400 エラーが発生するため、両方を対象にサニタイズする。
                ※ system_prompt はフォーマット指示（<|#|>の使い方）を含むためサニタイズしない。
                """
                if "<|" in text:
                    text = text.replace("<|#|>", " | ")   # エンティティ区切り子
                    import re
                    text = re.sub(r"<\|[^|>]*\|>", "", text)  # 残存する <|...|> パターンを除去
                return text

            def _fix_llm_output(text: str) -> str:
                """
                LLM 出力の entity / relation 行のフィールド数を正規化する。

                【根本原因】
                LightRAG は llm_func の返値を受け取った後、内部で
                fix_tuple_delimiter_corruption() を実行する。この関数は
                <|>、<#>、<||> などの「変形区切り子」を正式な <|#|> に変換する。
                そのため、LLM が <|> を誤出力した場合:
                  - 本関数実行時点: entity|名前<|>X|タイプ|説明 → split(<|#|>) で 4 フィールド(正常に見える)
                  - LightRAG 正規化後: entity|名前|X|タイプ|説明 → 5 フィールド → WARNING

                【対処】
                LightRAG と同じ正規化（fix_tuple_delimiter_corruption）を先に適用して
                フィールド数を数え、超過分を末尾フィールドに結合する。
                これにより「正規化後に超過する」パターンも確実に捕捉できる。

                【レコード種別の検出】
                LLM が "(entity<|#|>..." や '"entity"<|#|>...' のように
                括弧・引用符を付けて出力する場合を考慮し、先頭の非英字を無視する。
                """
                import re as _re
                SEP = "<|#|>"

                # 早期リターン: SEP も変形区切り子パターンも含まない
                if SEP not in text and not _re.search(r"<[|#]", text):
                    return text

                # LightRAG の fix_tuple_delimiter_corruption を使って正規化
                try:
                    from lightrag.utils import fix_tuple_delimiter_corruption as _ftdc
                    def _normalize(rec: str) -> str:
                        return _ftdc(rec, "#", SEP)
                except ImportError:
                    def _normalize(rec: str) -> str:  # type: ignore[misc]
                        return rec

                # レコード種別ごとの期待フィールド数
                EXPECTED: dict[str, int] = {
                    "entity":       4,
                    "relation":     5,
                    "relationship": 5,
                }

                fixed: list[str] = []
                for line in text.splitlines(keepends=True):
                    stripped = line.rstrip("\r\n")
                    # LightRAG が後で行う正規化を先に適用してフィールド数を正確に把握
                    normalized = _normalize(stripped)
                    eol = line[len(stripped):]  # 元の行末文字を保持

                    if SEP not in normalized:
                        fixed.append(line)
                        continue

                    parts = normalized.split(SEP)
                    # 先頭フィールドから記録種別を取得（括弧・引用符プレフィックスを除去）
                    clean_type = _re.sub(r"^[^a-z]+", "", parts[0].strip().lower())
                    expected = EXPECTED.get(clean_type)

                    if expected and len(parts) > expected:
                        # ① フィールド超過 → 末尾の余分なフィールドを説明フィールドに結合
                        merged = parts[: expected - 1] + [" ".join(parts[expected - 1 :])]
                        fixed.append(SEP.join(merged) + eol)
                    elif expected and 2 <= len(parts) < expected:
                        # ② フィールド不足 → "-" で補完（空文字はLightRAGのsplitでフィルタされるため使用不可）
                        # relation の4フィールドは keywords が欠落ケースが多い → position3 に挿入
                        if clean_type in ("relation", "relationship") and len(parts) == 4:
                            padded = parts[:3] + ["-"] + [parts[3]]
                        else:
                            padded = parts + ["-"] * (expected - len(parts))
                        fixed.append(SEP.join(padded) + eol)
                    else:
                        # ③ 正常 → 正規化済みの文字列を返す
                        fixed.append(normalized + eol)

                return "".join(fixed)

            messages = []
            if system_prompt:
                # system_prompt は <|#|> の使い方を LLM に教えるため変更しない
                messages.append({"role": "system", "content": system_prompt})

            for msg in (history_messages or []):
                # history（主にアシスタントメッセージ）に <|#|> が含まれる場合はサニタイズ
                content = msg.get("content") or ""
                if isinstance(content, str) and "<|" in content:
                    msg = dict(msg)
                    msg["content"] = _sanitize(content)
                messages.append(msg)

            # prompt 自体に <|#|> が含まれる場合もサニタイズ
            # （gleaning で前回の抽出結果が prompt に直接埋め込まれるケースへの対処）
            safe_prompt = _sanitize(prompt) if (prompt and "<|" in prompt) else prompt
            messages.append({"role": "user", "content": safe_prompt})

            # model が未指定 or デフォルト値の場合は送らない（LM Studio は省略で自動選択）
            payload: dict = {"messages": messages, "temperature": 0.1, "max_tokens": 4096}
            if llm_model and llm_model not in ("", "local-model"):
                payload["model"] = llm_model

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            url = f"{base_url}/chat/completions"

            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(url, headers=headers, json=payload)

                if resp.status_code == 400:
                    err_text = resp.text
                    print(f"[GraphRAG LLM Error] {resp.status_code}: {err_text[:300]}")

                    # ── コンテキスト超過は sanitize では直らないので即スキップ ──────
                    # LM Studio は "Context size has been exceeded" を返す。
                    # sanitize リトライをしても状況は変わらないため直ちにスキップする。
                    # 対処: LM Studio でモデルの「Context Length」を 8192 以上に設定する。
                    if "context" in err_text.lower() and "exceed" in err_text.lower():
                        print(
                            "[GraphRAG] ⚠️ コンテキスト超過 - チャンクをスキップします\n"
                            "           → LM Studio でモデルの Context Length を 8192 以上に設定してください"
                        )
                        return ""

                    # ── <|#|> トークン誤認識: 全メッセージを sanitize して再試行 ────
                    print("[GraphRAG] ⚠️ 400エラー - 全メッセージをサニタイズして再試行します")

                    retry_messages = []
                    for msg in payload["messages"]:
                        content = msg.get("content") or ""
                        if isinstance(content, str) and "<|" in content:
                            msg = dict(msg)
                            msg["content"] = _sanitize(content)
                        retry_messages.append(msg)

                    retry_payload = dict(payload)
                    retry_payload["messages"] = retry_messages
                    resp = await client.post(url, headers=headers, json=retry_payload)

                    # 再試行後も 400 → このチャンクのエンティティ抽出をスキップして継続
                    if resp.status_code == 400:
                        print(f"[GraphRAG LLM Error 再試行] {resp.status_code}: {resp.text[:200]}")
                        print("[GraphRAG] ⚠️ 再試行後も400エラー - チャンクをスキップして次へ継続")
                        return ""  # 空文字 → LightRAG がエンティティなしとして次チャンクへ進む

                if not resp.is_success:
                    print(f"[GraphRAG LLM Error] {resp.status_code}: {resp.text[:600]}")
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return _fix_llm_output(content)  # フィールド数超過による WARNING を抑制

        async def embed_func(texts: list[str]) -> np.ndarray:
            """LM Studio の /embeddings エンドポイントに1件ずつ直接リクエスト"""
            results = []
            async with httpx.AsyncClient(timeout=60.0) as client:
                for text in texts:
                    resp = await client.post(
                        f"{base_url}/embeddings",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"model": embed_model, "input": text},
                    )
                    resp.raise_for_status()
                    results.append(resp.json()["data"][0]["embedding"])
            return np.array(results, dtype=np.float32)

        return LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embed_dim,
                max_token_size=512,
                func=embed_func,
            ),
            # 文末チャンキング（トークン数固定ではなく 。！？ で区切る）
            chunking_func=_make_sentence_chunker(
                max_chars=cfg["chunk_max_chars"],
                overlap_sentences=cfg["chunk_overlap_sentences"],
            ),
            entity_extract_max_gleaning=0,  # 2回目抽出を無効化（コンテキスト節約）
            addon_params={
                "language": "Japanese",
                "entity_types": ["組織", "人物", "概念", "場所", "イベント", "製品"],
            },
        )

    async def initialize(self):
        """LLM モデル名・エンベディング次元数を自動検出し、LightRAG ストレージを初期化"""
        rebuild_needed = False

        # ① LLM モデル名の自動検出（未指定時）
        if not self._config["llm_model"]:
            detected = await self._detect_llm_model()
            if detected:
                self._config["llm_model"] = detected
                rebuild_needed = True
            else:
                raise RuntimeError(
                    "LM Studio に接続できないか、モデルがロードされていません。"
                    "LM Studio を起動してモデルをロードしてから再試行してください。"
                )

        # ② エンベディング次元数の自動検出
        #    ※ グラフデータは絶対に消去しない。設定値のみ更新し RAG インスタンスを再生成する
        actual_dim = await self._detect_embedding_dim()
        if actual_dim != self._config["embedding_dim"]:
            print(f"[GraphRAG] 次元数を修正: {self._config['embedding_dim']} → {actual_dim}")
            self._config["embedding_dim"] = actual_dim
            rebuild_needed = True

        if rebuild_needed:
            self._rag = self._create_rag_instance()

        if hasattr(self._rag, "initialize_storages"):
            await self._rag.initialize_storages()

        print(f"[GraphRAG] 初期化完了 - ドキュメント数: {len(self._docs)}, モデル: {self._config['llm_model']}")

    # ── ドキュメントインデックス管理 ─────────────────────────────────────

    def _load_docs_index(self) -> dict[str, str]:
        if self._docs_index_file.exists():
            try:
                return json.loads(self._docs_index_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_docs_index(self):
        self._docs_index_file.write_text(
            json.dumps(self._docs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── ファイル読み込み ──────────────────────────────────────────────────

    def _read_txt(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf(self, file_path: Path) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise ValueError(f"PDF読み込みエラー: {e}") from e

    def _read_file(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext == ".txt":
            return self._read_txt(file_path)
        if ext == ".pdf":
            return self._read_pdf(file_path)
        raise ValueError(f"未対応の形式: {ext}")

    # ── インデックス操作 ──────────────────────────────────────────────────

    async def add_document(self, file_path: str | Path) -> dict:
        """ドキュメントをグラフに追加"""
        file_path = Path(file_path)
        if not file_path.exists():
            return {"success": False, "file": file_path.name, "message": "ファイルが見つかりません"}
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return {"success": False, "file": file_path.name, "message": f"未対応の形式: {file_path.suffix}"}

        try:
            text = self._read_file(file_path)
        except ValueError as e:
            return {"success": False, "file": file_path.name, "message": str(e)}

        if not text.strip():
            return {"success": False, "file": file_path.name, "message": "テキストを抽出できませんでした"}

        try:
            await self._rag.ainsert(text)
            self._docs[file_path.name] = str(file_path)
            self._save_docs_index()
            return {"success": True, "file": file_path.name, "message": "グラフへの追加が完了しました"}
        except Exception as e:
            return {"success": False, "file": file_path.name, "message": str(e)}

    async def add_directory(self, dir_path: str | Path) -> list[dict]:
        """ディレクトリ内の全対応ファイルをグラフに追加"""
        dir_path = Path(dir_path)
        results = []
        for ext in SUPPORTED_EXTENSIONS:
            for fp in dir_path.rglob(f"*{ext}"):
                results.append(await self.add_document(fp))
        return results

    async def delete_document(self, file_name: str) -> dict:
        """ドキュメントを削除してグラフを再構築"""
        if file_name not in self._docs:
            return {"success": False, "message": "該当ファイルが見つかりません"}

        del self._docs[file_name]
        remaining = dict(self._docs)

        await self._clear_graph(rebuild=False)
        for fname, fpath in remaining.items():
            fp = Path(fpath)
            if fp.exists():
                try:
                    text = self._read_file(fp)
                    if text.strip():
                        await self._rag.ainsert(text)
                        self._docs[fname] = fpath
                except Exception:
                    pass
        self._save_docs_index()

        return {"success": True, "message": f"{file_name} を削除してグラフを再構築しました"}

    async def _clear_graph(self, rebuild: bool = True):
        """グラフデータファイルをすべて削除して再初期化"""
        for item in self.working_dir.iterdir():
            if item.name == "_indexed_docs.json":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        self._docs = {}
        self._save_docs_index()
        if rebuild:
            self._rag = self._create_rag_instance()
            if hasattr(self._rag, "initialize_storages"):
                await self._rag.initialize_storages()

    async def clear(self) -> dict:
        """グラフと全ドキュメント登録をクリア"""
        count = len(self._docs)
        await self._clear_graph()
        return {"cleared_documents": count}

    # ── 検索 ──────────────────────────────────────────────────────────────

    async def search(self, query: str, mode: str | None = None) -> list[dict]:
        """グラフ RAG でクエリを実行"""
        if not self._docs:
            print("[GraphRAG] 検索スキップ: ドキュメントが登録されていません")
            return []
        search_mode = mode or self.search_mode
        print(f"[GraphRAG] 検索開始: mode={search_mode}, docs={len(self._docs)}, query={query[:40]}")
        try:
            result = await self._rag.aquery(
                query,
                param=QueryParam(
                    mode=search_mode,
                    top_k=10,               # 取得エンティティ数（デフォルト40）
                    chunk_top_k=5,          # 取得チャンク数（デフォルト20）
                    max_entity_tokens=600,  # エンティティ最大トークン（デフォルト6000）
                    max_relation_tokens=800, # リレーション最大トークン（デフォルト8000）
                    max_total_tokens=2000,  # 合計最大トークン（デフォルト30000）
                    enable_rerank=False,    # rerankモデル未設定のため無効化
                ),
            )
            if result and result.strip():
                print(f"[GraphRAG] 検索成功: {len(result)}文字の回答を取得")
                print(f"[GraphRAG] 検索結果:\n{'='*60}\n{result}\n{'='*60}")
                return [{"content": result, "source": f"GraphRAG ({search_mode})", "score": 1.0}]
            print("[GraphRAG] 検索結果が空でした")
            return []
        except Exception as e:
            print(f"[GraphRAG] 検索エラー: {e}")
            return []

    # ── 状態確認 ──────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "total_documents": len(self._docs),
            "documents": sorted(self._docs.keys()),
            "search_mode": self.search_mode,
            "mode": "GraphRAG (LightRAG)",
        }

    # ── デバッグ用: チャンクごとのLLM入出力を可視化 ───────────────────────

    async def debug_extract(
        self,
        file_content: str,
        max_chars: int = 200,
        overlap_sentences: int = 1,
        max_chunks: int = 5,
    ):
        """
        チャンクごとのLLM入力・出力を非同期ジェネレータで返す。

        チャンクの切り方・プロンプト内容・LLM応答・パース結果を可視化するための
        デバッグ用メソッド。本番のインデックス処理とは独立して動作する。

        Yields:
            dict (type="progress"): チャンキング完了通知
            dict (type="result"):   各チャンクのLLM入出力・パース結果
            dict (type="done"):     全処理完了通知
            dict (type="error"):    エラー情報
        """
        from lightrag.prompt import PROMPTS

        # ── チャンキング ────────────────────────────────────────────────
        chunker = _make_sentence_chunker(
            max_chars=max_chars,
            overlap_sentences=overlap_sentences,
        )
        try:
            tokenizer = self._rag.tokenizer
        except AttributeError:
            class _FallbackTokenizer:
                def encode(self, text):
                    return list(text.encode("utf-8"))
            tokenizer = _FallbackTokenizer()

        chunks = await chunker(tokenizer, file_content)
        total_chunks = len(chunks)
        process_count = min(max_chunks, total_chunks)

        yield {
            "type": "progress",
            "total_chunks": total_chunks,
            "process_count": process_count,
            "message": f"チャンキング完了: {total_chunks} チャンク（最初の {process_count} 件を処理します）",
        }

        # ── LightRAG と同じプロンプトコンテキストを構築 ───────────────────
        # LightRAG インスタンスの addon_params から entity_types / language を取得
        addon = getattr(self._rag, "addon_params", {}) or {}
        entity_types: list[str] = addon.get(
            "entity_types", ["組織", "人物", "概念", "場所", "イベント", "製品"]
        )
        language: str = addon.get("language", "Japanese")

        try:
            examples = "\n".join(PROMPTS["entity_extraction_examples"])
            example_ctx = dict(
                tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
                completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
                entity_types=", ".join(entity_types),
                language=language,
            )
            examples = examples.format(**example_ctx)

            context_base = dict(
                tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
                completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
                entity_types=",".join(entity_types),
                examples=examples,
                language=language,
            )
            system_prompt = PROMPTS["entity_extraction_system_prompt"].format(**context_base)
        except Exception as e:
            yield {"type": "error", "message": f"プロンプトテンプレートの取得に失敗: {e}"}
            return

        # ── 各チャンクを処理 ───────────────────────────────────────────
        for i, chunk in enumerate(chunks[:max_chunks]):
            chunk_text: str = chunk["content"]
            chunk_tokens: int = chunk["tokens"]

            user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
                **{**context_base, "input_text": chunk_text}
            )

            import time as _time
            t0 = _time.perf_counter()
            try:
                raw_response: str = await self._rag.llm_model_func(
                    user_prompt,
                    system_prompt=system_prompt,
                )
            except Exception as e:
                raw_response = f"[ERROR] {e}"
            elapsed = round(_time.perf_counter() - t0, 2)

            entities, relations = _parse_extraction_response(raw_response)

            yield {
                "type": "result",
                "chunk_idx": i,
                "total_chunks": total_chunks,
                "process_count": process_count,
                "chunk_text": chunk_text,
                "tokens": chunk_tokens,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": raw_response,
                "entities": entities,
                "relations": relations,
                "elapsed_sec": elapsed,
            }

        yield {
            "type": "done",
            "total_chunks": total_chunks,
            "processed": process_count,
        }


def _parse_extraction_response(response: str) -> tuple[list[dict], list[dict]]:
    """
    LLM のエンティティ抽出レスポンスからエンティティと関係を解析する。

    LightRAG の区切り子 <|#|> でフィールドを分割し、
    先頭フィールドの種別（entity / relation）でリストに振り分ける。
    """
    import re as _re
    SEP = "<|#|>"
    entities: list[dict] = []
    relations: list[dict] = []

    for line in response.split("\n"):
        line = line.strip()
        # 終端マーカー・空行をスキップ
        if not line or "<|COMPLETE|>" in line:
            continue
        parts = [p.strip() for p in line.split(SEP)]
        if len(parts) < 2:
            continue

        # 先頭フィールドから種別を取得（括弧・引用符を除去）
        rec_type = _re.sub(r"^[^a-z]+", "", parts[0].strip().lower())

        if "entity" in rec_type:
            entities.append({
                "name":        parts[1] if len(parts) > 1 else "",
                "type":        parts[2] if len(parts) > 2 else "",
                "description": parts[3] if len(parts) > 3 else "",
                "raw":         line,
            })
        elif "relation" in rec_type or "relationship" in rec_type:
            relations.append({
                "source":      parts[1] if len(parts) > 1 else "",
                "target":      parts[2] if len(parts) > 2 else "",
                "keywords":    parts[3] if len(parts) > 3 else "",
                "description": parts[4] if len(parts) > 4 else "",
                "raw":         line,
            })

    return entities, relations

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


def _customize_extraction_prompts() -> None:
    """
    LightRAG の entity_extraction_system_prompt 先頭に幻覚抑制ルールを注入する。

    【設計方針】
    ① ドメイン非依存 ── 技術文書・ニュース・小説・仕様書など任意の入力に適用できる
                        汎用的なルールのみを記述し、特定のジャンルや固有名詞は使わない
    ② 英語で記述 ── llama-3.1-8b クラスの小規模モデルは英語の命令への従順性が
                     日本語より高い（出力言語とは無関係）
    ③ プロンプト先頭に配置 ── アテンションはプロンプト末尾ほど低下するため、
                               最重要ルールを先頭に置く
    ④ 短く番号付き ── 長文ルールは小規模モデルが途中を読み飛ばすため、
                       1ルール1文の箇条書きにする

    【注入するルールの根拠】
    RULE 1 (NO HALLUCINATION)
        有名な固有名詞（人名・地名・作品名など）を含む文書では、
        LLM が事前学習知識をテキストに投影する（幻覚）。
        「テキストに書かれていないことは追加するな」と明示することで抑制する。

    RULE 2 (MINIMAL DESCRIPTION)
        description を長く書こうとするほど幻覚が増える。
        テキストの言葉を使った1〜2文に制限することで幻覚の機会を減らす。

    RULE 3 (NO INFERRED RELATIONS)
        「AとBの息子C」からLLMは「AとBは夫婦」を推論して追加しがち。
        テキストに明記されていない関係は生成禁止と明示する。

    RULE 4 (EXPLICIT RELATIONS ONLY)
        共起（同じ文に登場する）だけで relation を生成するモデルへの対策。
        テキストで明示的に関係づけられている場合のみ relation を生成する。

    RULE 5 (DIRECTIONAL KEYWORDS)
        「兄弟」「家族」「関係」など方向性のない語は使わず、
        「兄」「弟子」「雇用主」など方向性のある語を要求する。
    """
    from lightrag.prompt import PROMPTS

    prefix = """\
=== CRITICAL EXTRACTION RULES (highest priority — follow before all other instructions) ===
RULE 1 [NO HALLUCINATION]:
  Extract ONLY information that is EXPLICITLY written in the input text.
  Do NOT add any fact from memory, training data, or inference — even if you know the topic well.
RULE 2 [ENTITY DESCRIPTION]:
  Entity description = direct paraphrase from the input text only (1-2 sentences max).
  Do NOT elaborate, speculate, or add background knowledge about the entity.
RULE 3 [RELATION DESCRIPTION — REQUIRED, NEVER OMIT]:
  Every relation MUST have a description (the 5th field). Never leave it empty.
  Write 1 sentence from the input text explaining how/why the relation exists.
  Correct format: relation<|#|>A<|#|>B<|#|>keywords<|#|>one-sentence description from text
  Wrong format:   relation<|#|>A<|#|>B<|#|>keywords          ← missing description field
RULE 4 [NO INFERRED RELATIONS]:
  "C is the child of A and B" → extract C→A and C→B ONLY.
  Do NOT infer A→B (e.g. spouse/partner) — that relationship is NOT stated in the text.
RULE 5 [EXPLICIT RELATIONS ONLY]:
  Only create a relation when the text EXPLICITLY states a connection between two entities.
  Two entities appearing near each other is NOT sufficient to create a relation.
RULE 6 [DIRECTIONAL KEYWORDS — 1 to 2 words only]:
  Keywords must describe the relationship type, not the entity type.
  GOOD: "son", "older brother", "teacher", "disciple", "spouse", "employer"
  BAD : "family relationship", "sibling", "person", "human", "related", "associated"
  NOTE: Entity types such as "person", "organization", "concept" are NEVER valid keywords.
RULE 7 [CO-PARENT PATTERN — most common mistake]:
  When text says "C is son/daughter of A and B":
  - A and B are CO-PARENTS of C. They are NOT parent and child of each other.
  - A's description: "A is C's father/mother" — NEVER "A is parent of C and B"
  - B's description: "B is C's father/mother" — NEVER "B is parent of C and A"
  WRONG: 孫悟空's description = "孫悟飯とチチの父親" (implies チチ is also 孫悟空's child)
  RIGHT: 孫悟空's description = "孫悟飯の父親"
RULE 8 [EXTRACT ALL STATED RELATIONS]:
  Extract a relation for EVERY connection explicitly described in the text.
  If an entity description says "A is B's older brother", also output:
    relation<|#|>A<|#|>B<|#|>older brother<|#|>A is B's older brother.
  Do not rely solely on entity descriptions — relations must also be in the relation list.
RULE 9 [RECIPROCAL ROLE INVERSION — most common mistake for sibling/parent relationships]:
  When text says "A is B's [role]", entity B has the INVERSE role relative to A.
  You MUST use the inverse when writing B's entity description.
  Reciprocal role pairs (use these conversions):
    弟 (younger brother) ↔ 兄 (older brother)
    妹 (younger sister)  ↔ 姉 (older sister)
    息子 (son)   → 父 (father) or 母 (mother)
    娘 (daughter)→ 父 (father) or 母 (mother)
    弟子 (disciple) ↔ 師匠 (teacher/master)
    部下 (subordinate) ↔ 上司 (superior)
  Step-by-step example:
    Text: "孫悟天は孫悟飯の弟"
    → role is 弟, so 孫悟天 is 孫悟飯's YOUNGER brother
    → 孫悟飯 has the INVERSE role = 兄 (older brother)
    WRONG description for 孫悟飯: "孫悟飯は孫悟天の弟" ← same role, not inverted!
    RIGHT description for 孫悟飯: "孫悟飯は孫悟天の兄" ← correctly inverted
=== END CRITICAL RULES ===

"""

    key = "entity_extraction_system_prompt"
    if key in PROMPTS and "CRITICAL EXTRACTION RULES" not in PROMPTS[key]:
        PROMPTS[key] = prefix + PROMPTS[key]
        print("[GraphRAG] 抽出プロンプトの先頭に幻覚抑制ルール（9条）を注入しました")

    # ── エンティティ説明要約プロンプトにも注入 ────────────────────────────
    # summarize_entity_descriptions は複数チャンクの説明を1文に統合する LLM 呼び出し。
    # ここで "AとBの父親" のような誤合成が起きるため、専用ルールを注入する。
    #
    # 【誤合成の例】
    #   入力説明①: "孫悟空は孫悟飯の父親"
    #   入力説明②: "孫悟空はチチの夫"
    #   → LLM が "統合" しようとして "孫悟飯とチチの父親" と圧縮してしまう
    #
    # 【防止ルール】
    #   RULE A: 関係は絶対に圧縮するな。"父of孫悟飯" と "夫of チチ" は別の文として書け。
    #   RULE B: "AとBのC" 形式の誰が子で誰が妻か曖昧な表現を禁止する。
    summ_key = "summarize_entity_descriptions"
    summ_prefix = """\
=== CRITICAL MERGE RULES (follow before all other instructions) ===
RULE A [NEVER COMPRESS RELATIONS]:
  Keep every relationship as a SEPARATE sentence. Do NOT merge different relationships into one.
  WRONG: "孫悟空は孫悟飯とチチの父親" (merges parent + spouse into one phrase)
  RIGHT: "孫悟空は孫悟飯の父親。孫悟空はチチの夫。" (two separate facts)
RULE B [PROHIBIT AMBIGUOUS "AとBのC" PATTERN]:
  Never write "XはAとBのC" when A and B have DIFFERENT relationship types with X.
  Only use "AとBの" if A and B truly share the identical role (e.g., both are children of X).
RULE C [NO HALLUCINATION]:
  Use ONLY information from the provided descriptions. Do NOT add facts from memory.
RULE D [PRESERVE DIRECTIONALITY]:
  Sibling roles (兄/弟/姉/妹) must be preserved as-is. Never swap them.
  Parent/child roles (父/母/息子/娘) must be preserved as-is. Never swap them.
=== END CRITICAL MERGE RULES ===

"""
    if summ_key in PROMPTS and "CRITICAL MERGE RULES" not in PROMPTS[summ_key]:
        PROMPTS[summ_key] = summ_prefix + PROMPTS[summ_key]
        print("[GraphRAG] エンティティ要約プロンプト 'summarize_entity_descriptions' に誤合成防止ルールを注入しました")

    # ── クエリ応答プロンプトにも幻覚抑制ルールを注入 ───────────────────────
    # aquery() の回答生成でも同じ llm_func が呼ばれるが、クエリ用プロンプトには
    # CRITICAL EXTRACTION RULES が含まれないため、別途ルールを注入する。
    query_prefix = (
        "=== STRICT ANSWER RULES ===\n"
        "RULE A [NO HALLUCINATION]: Answer ONLY from the provided context/data tables below.\n"
        "  Do NOT use any training knowledge, prior information, or inference not in the context.\n"
        "RULE B [UNKNOWN = SAY SO]: If the answer is not in the provided context,\n"
        "  state explicitly that the information is not available in the provided data.\n"
        "  Do NOT guess or fill in from memory.\n"
        "=== END STRICT ANSWER RULES ===\n\n"
    )
    _RAG_QUERY_PROMPT_KEYS = [
        "local_rag_response",
        "global_map_rag_response",
        "global_reduce_rag_response",
        "naive_rag_response",
    ]
    for qkey in _RAG_QUERY_PROMPT_KEYS:
        if qkey in PROMPTS and "STRICT ANSWER RULES" not in PROMPTS[qkey]:
            PROMPTS[qkey] = query_prefix + PROMPTS[qkey]
            print(f"[GraphRAG] クエリ応答プロンプト '{qkey}' に幻覚抑制ルールを注入しました")


_customize_extraction_prompts()


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
        normalize_on_insert: bool = False,
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
            # 登録時テキスト前処理
            "normalize_on_insert": normalize_on_insert,  # True: 登録前に1文1事実化
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

        # ── 同時 LLM 呼び出し制限 ─────────────────────────────────────────────
        # LightRAG はチャンクを asyncio で並列処理するが、ローカル LM Studio では
        # 複数スロットの KV キャッシュを同時確保しようとして VRAM が不足しやすい。
        # （4スロット × 8192トークン分を同時確保 → VRAM超過 → Channel Error）
        # Semaphore(1) で直列化することで KV キャッシュは常に 1スロット分のみ使用する。
        import asyncio as _asyncio
        _llm_semaphore = _asyncio.Semaphore(1)

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

            # ── エンティティ抽出コールへのグラウンディング ────────────────────
            # system_prompt に我々が注入した "CRITICAL EXTRACTION RULES" が含まれる場合
            # = エンティティ抽出コール（クエリ応答や gleaning 2回目では異なる）
            # チャンクテキストの直前に「このテキストだけを使え」アンカーを追加する。
            # system_prompt の先頭ルールを補強し、小規模モデルの幻覚をさらに抑制する。
            _is_extraction = "CRITICAL EXTRACTION RULES" in (system_prompt or "")
            if _is_extraction and prompt:
                _grounding = (
                    "[MANDATORY: Use ONLY the text provided in this message. "
                    "Do NOT add any fact not explicitly written there.]\n\n"
                )
                prompt = _grounding + prompt

            # prompt 自体に <|#|> が含まれる場合もサニタイズ
            # （gleaning で前回の抽出結果が prompt に直接埋め込まれるケースへの対処）
            safe_prompt = _sanitize(prompt) if (prompt and "<|" in prompt) else prompt
            messages.append({"role": "user", "content": safe_prompt})

            # temperature=0: 抽出タスクはランダム性ゼロが最適（創造性が幻覚になる）
            # model が未指定 or デフォルト値の場合は送らない（LM Studio は省略で自動選択）
            # max_tokens: 抽出出力はエンティティ・関係リストなので 2048 で十分。
            # 4096 にすると LM Studio が (prompt + max_tokens) > n_ctx を事前チェックして
            # コンテキスト超過エラーを起こしやすくなる。
            payload: dict = {"messages": messages, "temperature": 0, "max_tokens": 2048}
            if llm_model and llm_model not in ("", "local-model"):
                payload["model"] = llm_model
            print(f"[GraphRAG LLM] リクエスト送信: model={llm_model or '（自動）'}, messages={len(messages)}")

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            url = f"{base_url}/chat/completions"

            # ── Semaphore で LM Studio への同時接続を 1 件に制限 ─────────────────
            # LightRAG はチャンクを asyncio で並列送信するが、LM Studio がスロットを
            # 複数確保しようとして KV キャッシュ（VRAM）が不足することがある。
            # Semaphore(1) で直列化し、常に 1 スロット分のみ使用させる。
            await _llm_semaphore.acquire()
            try:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    # ── 1回目リクエスト ────────────────────────────────────────────
                    try:
                        resp = await client.post(url, headers=headers, json=payload)
                    except httpx.TransportError as e:
                        # Channel Error / 接続切断 ─ コンテキスト超過時に LM Studio が
                        # 接続をドロップすると、並列リクエストがここに落ちる。
                        # HTTP レスポンスが返らないため status_code では捕捉できない。
                        print(
                            f"[GraphRAG LLM Error] 接続切断 ({type(e).__name__}): {e}\n"
                            "[GraphRAG] ⚠️ Channel Error - チャンクをスキップします\n"
                            "           → LM Studio のモデル Context Length を 8192 以上に設定してください"
                        )
                        return ""

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

                        # ── 2回目リクエスト（sanitize 後）────────────────────────────
                        try:
                            resp = await client.post(url, headers=headers, json=retry_payload)
                        except httpx.TransportError as e:
                            print(
                                f"[GraphRAG LLM Error] 再試行後も接続切断 ({type(e).__name__}): {e}\n"
                                "[GraphRAG] ⚠️ 再試行後も Channel Error - チャンクをスキップします"
                            )
                            return ""

                        # 再試行後も 400 → このチャンクのエンティティ抽出をスキップして継続
                        if resp.status_code == 400:
                            print(f"[GraphRAG LLM Error 再試行] {resp.status_code}: {resp.text[:200]}")
                            print("[GraphRAG] ⚠️ 再試行後も400エラー - チャンクをスキップして次へ継続")
                            return ""  # 空文字 → LightRAG がエンティティなしとして次チャンクへ進む

                    if not resp.is_success:
                        print(f"[GraphRAG LLM Error] {resp.status_code}: {resp.text[:600]}")
                    resp.raise_for_status()
                    resp_json = resp.json()
                    actual_model = resp_json.get("model", "")
                    if actual_model and actual_model != llm_model:
                        print(f"[GraphRAG LLM] ⚠️ 要求モデル={llm_model or '（自動）'} / 実際に使用={actual_model}")
                    content = resp_json["choices"][0]["message"]["content"]
                    return _fix_llm_output(content)  # フィールド数超過による WARNING を抑制
            finally:
                _llm_semaphore.release()

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
            # ── エンティティ説明のLLM要約を無効化 ──────────────────────────────
            # デフォルト: force_llm_summary_on_merge=8, summary_max_tokens=1200
            # → 孫悟空のような頻出エンティティは8件以上の説明が集まり、
            #   LLMが複数の説明を1文に「統合」しようとして
            #   "孫悟飯とチチの父親" のような誤合成を起こす。
            # 対策: 閾値を大幅に引き上げ、<SEP>結合（LLMなし）を使わせる。
            # <SEP>結合は冗長だが正確性を保てる。小規模モデルでの安全な選択。
            force_llm_summary_on_merge=9999,  # 実質的にLLM要約を無効化
            summary_max_tokens=99999,          # トークン上限を無効化
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
            # 登録前テキスト前処理（normalize_on_insert=True のとき）
            if self._config.get("normalize_on_insert", False):
                print(f"[GraphRAG] 登録前テキスト前処理（1文1事実）を適用: {file_path.name}")
                text = await self._normalize_full_text(text)

            await self._rag.ainsert(text)
            # 抽出後の誤合成パターン（"AとBの父親"）を Relations と照合して修正
            await self._fix_compound_parent_descriptions()
            self._docs[file_path.name] = str(file_path)
            self._save_docs_index()
            mode_label = "（前処理あり）" if self._config.get("normalize_on_insert") else ""
            return {"success": True, "file": file_path.name, "message": f"グラフへの追加が完了しました{mode_label}"}
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
                        if self._config.get("normalize_on_insert", False):
                            text = await self._normalize_full_text(text)
                        await self._rag.ainsert(text)
                        self._docs[fname] = fpath
                except Exception:
                    pass
        # 再構築後も誤合成パターンを修正
        await self._fix_compound_parent_descriptions()
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
            # only_need_context=True: LightRAG がグラフから取得した生データ（エンティティ・
            # 関係性・チャンクの一覧テキスト）を返す。LLM による回答生成は行わない。
            #
            # 【なぜ only_need_context=True か】
            # aquery() がデフォルト (False) の場合、LightRAG は内部で llm_func を呼んで
            # グラフデータを元に回答を生成する。この段階で LLM がグラフ外の学習知識を
            # 混入させてハルシネーションを起こすことがある。
            # only_need_context=True にすると、グラフの生データ（事実の列挙）だけが返り、
            # LLM による解釈・合成は行われない。回答の生成はチャット側の LLM に委ねる。
            context = await self._rag.aquery(
                query,
                param=QueryParam(
                    mode=search_mode,
                    only_need_context=True,  # ← グラフ生データのみ取得（LLM呼び出しなし）
                    top_k=30,               # 取得エンティティ数（デフォルト40）
                    chunk_top_k=10,         # 取得チャンク数（デフォルト20）
                    max_entity_tokens=4000, # エンティティ最大トークン（デフォルト6000）
                    max_relation_tokens=4000, # リレーション最大トークン（デフォルト8000）
                    max_total_tokens=8000,  # 合計最大トークン（デフォルト30000）
                    enable_rerank=False,    # rerankモデル未設定のため無効化
                ),
            )
            if context and context.strip():
                print(f"[GraphRAG] 検索成功: {len(context)}文字のコンテキストを取得")
                # 関係性セクションを重点的にログ出力
                rel_start = context.find("Relationships")
                if rel_start >= 0:
                    print(f"[GraphRAG] 検索コンテキスト（Relationships）:\n{'='*60}\n{context[rel_start:rel_start+1200]}\n{'='*60}")
                else:
                    print(f"[GraphRAG] 検索コンテキスト:\n{'='*60}\n{context[:1200]}\n{'='*60}")
                return [{"content": context, "source": f"GraphRAG ({search_mode})", "score": 1.0}]
            print("[GraphRAG] 検索結果が空でした")
            return []
        except Exception as e:
            print(f"[GraphRAG] 検索エラー: {e}")
            return []

    # ── 状態確認 ──────────────────────────────────────────────────────────

    # ── LLM モデル変更 ────────────────────────────────────────────────────

    async def set_llm_model(self, model: str) -> None:
        """
        GraphRAG インデックス作成時に使う LLM モデルを切り替える。

        model: "" → 次回 initialize() 時に自動検出
        model: "some-model-id" → 指定モデルを使用

        LightRAG インスタンスを再生成してストレージを再接続する。
        既存のグラフデータは保持される。
        """
        self._config["llm_model"] = model
        self._rag = self._create_rag_instance()
        if hasattr(self._rag, "initialize_storages"):
            await self._rag.initialize_storages()
        label = model if model else "（自動検出）"
        print(f"[GraphRAG] LLM モデルを変更しました: {label}")

    def get_status(self) -> dict:
        return {
            "total_documents": len(self._docs),
            "documents": sorted(self._docs.keys()),
            "search_mode": self.search_mode,
            "mode": "GraphRAG (LightRAG)",
            "llm_model": self._config.get("llm_model", ""),
            "normalize_on_insert": self._config.get("normalize_on_insert", False),
        }

    # ── デバッグ用: チャンクごとのLLM入出力を可視化 ───────────────────────

    # ── テキスト前処理（複合文 → 単純文） ──────────────────────────────

    async def _normalize_full_text(self, text: str) -> str:
        """
        ドキュメント全体を段落単位で 1文1事実化する。

        add_document() で normalize_on_insert=True のときに呼ばれる。
        段落（空行区切り）ごとに _normalize_text() を適用して結合して返す。
        短い段落（80文字以下）は複合文の可能性が低いためスキップして処理を高速化する。

        Returns:
            正規化済みテキスト。段落ごとの LLM 失敗は元テキストで継続。
        """
        paragraphs = text.split("\n\n")
        normalized: list[str] = []
        total = len(paragraphs)
        for i, para in enumerate(paragraphs):
            stripped = para.strip()
            if not stripped:
                normalized.append(para)
                continue
            # 短い段落はスキップ（複合文になりにくい）
            if len(stripped) <= 80:
                normalized.append(para)
                continue
            print(f"[GraphRAG] テキスト前処理 [{i+1}/{total}]: {stripped[:40]}...")
            norm_para = await self._normalize_text(stripped)
            normalized.append(norm_para)
        result = "\n\n".join(normalized)
        print(f"[GraphRAG] テキスト前処理完了: {len(text)}文字 → {len(result)}文字")
        return result

    async def _fix_compound_parent_descriptions(self) -> None:
        """
        グラフ構築後に実行するエンティティ説明の後処理。

        小規模 LLM が抽出ステップで生成する「AとBのC（父親など）」という
        誤合成パターンを Relations データと照合して修正する。

        【問題パターン】
            孫悟空の説明: "孫悟空は孫悟飯とチチの父親"
            ↑ チチは孫悟空の妻なのに子ども扱いになっている

        【修正戦略（ホワイトリスト方式）】
        Z=「父親/母親」のとき:
          X・Y それぞれについてグラフのエッジを検索し、
          「親子関係」を示すキーワードのエッジが存在するエンティティのみ保持。
          親子エッジがない（=配偶者・兄弟など）エンティティは除外して分離。
        """
        import re as _re

        print("[GraphRAG] description後処理: 誤合成パターンのスキャン開始")

        # 親子関係を示す役割語
        _PARENT_ROLES = r"(?:父親?|母親?|保護者|親)"
        _CHILD_ROLES  = r"(?:息子|娘|子供?|子ども(?:たち)?)"
        _FAMILY_ROLES = _PARENT_ROLES + r"|" + _CHILD_ROLES + r"|(?:兄|弟|姉|妹)"

        # "AはXとYのZ" のパターン
        _COMPOUND_PAT = _re.compile(
            r"^(.+?)は(.+?)と(.+?)の(" + _FAMILY_ROLES + r")"
        )

        # 親子関係を示すキーワード（グラフのエッジの description/keywords に含まれる）
        _PARENT_CHILD_KW = {
            "父", "母", "父親", "母親", "息子", "娘", "子", "親子",
            "父子", "母子", "son", "daughter", "father", "mother", "child", "parent",
        }

        # ① グラフの全ノード名を取得
        try:
            entity_names: list[str] = await self._rag.get_graph_labels()
            if not isinstance(entity_names, list):
                entity_names = list(entity_names) if entity_names else []
            print(f"[GraphRAG] description後処理: {len(entity_names)}エンティティを取得")
        except Exception as e:
            print(f"[GraphRAG] description後処理: ノード一覧取得失敗 ({e})")
            return

        if not entity_names:
            print("[GraphRAG] description後処理: エンティティなし")
            return

        # ② 全エッジを一括取得してキャッシュ（エンティティ名 → 隣接エンティティ集合）
        # 辺ラベルで「親子エッジ」の対を記録: {(src, tgt)} or {(tgt, src)} if parent-child
        parent_child_pairs: set[frozenset[str]] = set()
        try:
            # ルートノードを適当に1つ選んで深さ99で取得する代わりに
            # 全エンティティ分ループして1-hop エッジを集める
            for ename in entity_names:
                try:
                    kg = await self._rag.get_knowledge_graph(
                        node_label=ename, max_depth=1, max_nodes=200
                    )
                    for edge in (kg.edges or []):
                        eprops = edge.properties or {}
                        edge_desc = (eprops.get("description", "") or "").lower()
                        edge_kw   = (eprops.get("keywords",    "") or "").lower()
                        edge_text = edge_desc + " " + edge_kw
                        if any(kw in edge_text for kw in _PARENT_CHILD_KW):
                            parent_child_pairs.add(frozenset([edge.source, edge.target]))
                except Exception:
                    pass
        except Exception as e:
            print(f"[GraphRAG] description後処理: エッジ取得エラー ({e})")

        print(f"[GraphRAG] description後処理: 親子エッジペア {len(parent_child_pairs)}組を検出")

        fixed_count = 0
        sep = "<SEP>"

        for ename in entity_names:
            try:
                info = await self._rag.get_entity_info(ename)
            except Exception:
                continue

            if not info:
                continue

            node_data = (info.get("graph_data") or {}) if isinstance(info, dict) else {}
            raw_desc  = (node_data.get("description", "") or "") if isinstance(node_data, dict) else ""
            if not raw_desc:
                continue

            fragments    = raw_desc.split(sep)
            new_fragments: list[str] = []
            changed = False

            for frag in fragments:
                frag_stripped = frag.strip()
                m = _COMPOUND_PAT.match(frag_stripped)
                if not m:
                    new_fragments.append(frag)
                    continue

                subj, ent_a, ent_b, role = m.groups()
                subj_s = subj.strip()
                a_s    = ent_a.strip()
                b_s    = ent_b.strip()

                print(f"[GraphRAG] description後処理: パターン検出 '{frag_stripped}'")
                print(f"  subj={subj_s}, A={a_s}, B={b_s}, role={role}")

                # ホワイトリスト方式: 親子エッジが確認できるエンティティのみ残す
                a_has_parent_child = frozenset([subj_s, a_s]) in parent_child_pairs
                b_has_parent_child = frozenset([subj_s, b_s]) in parent_child_pairs

                print(f"  A({a_s})親子エッジ: {a_has_parent_child}, B({b_s})親子エッジ: {b_has_parent_child}")

                # 両方とも親子エッジあり → 正当な複数の子 → 変更不要
                if a_has_parent_child and b_has_parent_child:
                    new_fragments.append(frag)
                    continue

                # 片方しか親子エッジがない → 誤合成
                parts: list[str] = []
                if a_has_parent_child:
                    parts.append(f"{subj_s}は{a_s}の{role}")
                elif not a_has_parent_child and not b_has_parent_child:
                    # 両方とも親子エッジなし → エンティティ説明から判断
                    # この場合はフォールバックとして元のフラグメントを保持
                    new_fragments.append(frag)
                    continue
                if b_has_parent_child:
                    parts.append(f"{subj_s}は{b_s}の{role}")

                if parts:
                    new_frag = sep.join(parts)
                    new_fragments.append(new_frag)
                    print(f"[GraphRAG] description修正: '{frag_stripped}' → '{new_frag}'")
                    changed = True
                else:
                    new_fragments.append(frag)

            if changed:
                new_desc = sep.join(new_fragments)
                try:
                    await self._rag.aedit_entity(ename, {"description": new_desc})
                    fixed_count += 1
                except Exception as e2:
                    print(f"[GraphRAG] description修正失敗 '{ename}': {e2}")

        print(f"[GraphRAG] description後処理完了: {len(entity_names)}件スキャン, {fixed_count}件修正")

    async def _normalize_text(self, text: str, model: str = "") -> str:
        """
        複合文を「1文1事実」のシンプルな文列に変換する前処理。

        エンティティ抽出の前に適用することで、8B クラスの小規模 LLM が
        「AとBのC（役割）」「Xで、Y」などの複合構文を誤って解析する問題を回避する。

        【変換例】
          元: 孫悟飯は孫悟空とチチの息子で、孫悟天の兄。
          後: 孫悟飯は孫悟空の息子である。
              孫悟飯はチチの息子である。
              孫悟飯は孫悟天の兄である。

        【設計方針】
        - タスクを「書き換え」に限定（知識は不要）→ 8B モデルでも比較的高精度
        - ドメイン非依存（人物・組織・技術文書など任意の入力に適用可能）
        - _debug_llm_call を経由するため、デバッガーと本番で同じモデルを使用
        - 失敗した場合は元のテキストを返して処理を継続（エラー非透過）

        Args:
            text:  正規化対象のテキスト（通常はチャンク単位）
            model: 使用モデル。空 = RAG デフォルト, 文字列 = 指定モデル

        Returns:
            正規化後のテキスト。LLM 呼び出し失敗時は元テキストをそのまま返す。
        """
        system_prompt = (
            "You are a text simplification assistant.\n"
            "Task: Rewrite the input text as simple, independent sentences.\n"
            "\n"
            "Rules:\n"
            "  1. Each output sentence must express EXACTLY ONE fact or relationship.\n"
            "  2. Preserve ALL information from the original — do NOT omit, add, or change any facts.\n"
            "  3. Compound subjects: split into separate sentences.\n"
            "     Input : 'C is [role] of A and B'\n"
            "     Output: 'C is [role] of A.' AND 'C is [role] of B.' (two sentences)\n"
            "  4. Compound predicates: split into separate sentences.\n"
            "     Input : 'X is Y, and does Z'\n"
            "     Output: 'X is Y.' AND 'X does Z.' (two sentences)\n"
            "  5. Output ONLY the simplified sentences, one per line. No explanations.\n"
            "  6. Keep the SAME language as the input."
        )
        user_prompt = f"Simplify into one-fact-per-sentence:\n\n{text}"

        try:
            result = await self._debug_llm_call(user_prompt, system_prompt, model=model)
            normalized = result.strip()
            return normalized if normalized else text
        except Exception as e:
            print(f"[GraphRAG] テキスト正規化失敗（元テキストを使用）: {e}")
            return text

    async def _debug_llm_call(
        self,
        user_prompt: str,
        system_prompt: str,
        model: str = "",
    ) -> str:
        """
        デバッグ用 LLM 呼び出し。

        model が空の場合は通常の llm_model_func（本番と同じ _fix_llm_output 適用済み）を使う。
        model が指定された場合は、そのモデルで直接 HTTP 呼び出しを行い
        生応答をそのまま返す（_fix_llm_output は適用しない）。
        デバッガーでは「LLM が実際に何を返したか」を見たいため、修正なしの方が有用。
        """
        if not model:
            # 本番と同じ llm_func（_fix_llm_output 適用済み）
            return await self._rag.llm_model_func(
                user_prompt,
                system_prompt=system_prompt,
            )

        # モデル指定あり → 直接 HTTP 呼び出し（生応答を返す）
        cfg = self._config
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                resp = await client.post(
                    f"{cfg['lm_studio_base_url']}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            except httpx.TransportError as e:
                return f"[Channel Error] 接続切断: {type(e).__name__}: {e}"
            if not resp.is_success:
                return f"[ERROR {resp.status_code}] {resp.text[:300]}"
            return resp.json()["choices"][0]["message"]["content"]

    async def debug_extract(
        self,
        file_content: str,
        max_chars: int = 200,
        overlap_sentences: int = 1,
        max_chunks: int = 5,
        model: str = "",
        normalize: bool = False,
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
        import time as _time

        for i, chunk in enumerate(chunks[:max_chunks]):
            chunk_text: str = chunk["content"]
            chunk_tokens: int = chunk["tokens"]

            # ── 前処理: 複合文 → 単純文（オプション）──────────────────────
            normalize_elapsed: float | None = None
            normalized_text: str | None = None
            if normalize:
                t_norm = _time.perf_counter()
                normalized_text = await self._normalize_text(chunk_text, model=model)
                normalize_elapsed = round(_time.perf_counter() - t_norm, 2)

            # 抽出に使うテキスト（前処理ありなら正規化済みテキストを使う）
            extraction_text = normalized_text if (normalize and normalized_text) else chunk_text

            # llm_func のグラウンディングと同じ前置きを input_text に埋め込む。
            # デバッガーで「実際に LLM に送っているテキスト」を可視化するため、
            # ユーザープロンプト側に明示する（llm_func 側のアンカーと二重適用になるが
            # 小規模モデルへの強調として意図的）。
            _grounded_input = (
                "[Use ONLY the following text. Do NOT add any fact not written here.]\n\n"
                + extraction_text
            )
            user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
                **{**context_base, "input_text": _grounded_input}
            )

            t0 = _time.perf_counter()
            try:
                raw_response: str = await self._debug_llm_call(
                    user_prompt,
                    system_prompt=system_prompt,
                    model=model,
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
                "normalized_text": normalized_text,        # None = 前処理なし
                "normalize_elapsed_sec": normalize_elapsed, # None = 前処理なし
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


# 役割語の相互関係マップ（A が B の [key] → B は A の [value]）
_RECIPROCAL_ROLES: dict[str, str] = {
    "弟":   "兄",
    "兄":   "弟",
    "妹":   "姉",
    "姉":   "妹",
    "弟子": "師匠",
    "師匠": "弟子",
    "部下": "上司",
    "上司": "部下",
}


def _fix_reciprocal_descriptions(
    entities: list[dict],
    relations: list[dict],
) -> list[dict]:
    """
    「AはBのX（役割）」という entity description が、
    relation の方向と矛盾している場合に説明文を修正する。

    【修正の仕組み】
    1. 各 relation から「srcはtgtのkeyword」の事実を構築
    2. 各 entity の description で「EntityはXのY（役割）」パターンを検出
    3. 検出された役割が relation と逆向きになっている場合、_RECIPROCAL_ROLES で修正

    例:
      relation: 孫悟天 → 孫悟飯, keywords=弟  （孫悟天は孫悟飯の弟）
      entity 孫悟飯の description: 「孫悟飯は孫悟天の弟です。」← 矛盾（relation で弟は孫悟天側）
      修正後:「孫悟飯は孫悟天の兄です。」（弟 → 兄 に変換）

    【限界】
    - 役割語が description に日本語で明示されている場合のみ機能する
    - マップにない役割語（父/母など親子は息子/娘側のみに定義）は修正不能
    - description のパースが文字列マッチに依存するため、表現が違うと検出できない
    """
    import re as _re

    # relation から「src は tgt の [role]」の事実セットを構築
    # key: (src, tgt) → set of role words in that direction
    # 抽出元は keywords と description の両方を使う
    relation_roles: dict[tuple[str, str], set[str]] = {}
    for rel in relations:
        src  = rel.get("source", "").strip()
        tgt  = rel.get("target", "").strip()
        kw   = rel.get("keywords", "")
        desc = rel.get("description", "")
        if not src or not tgt:
            continue

        roles_found: set[str] = set()

        # ① キーワードを分割して直接追加 + 複合語内の役割語も抽出
        #    例: "兄弟関係" → "兄弟関係" を追加し、さらに "弟"・"兄" も追加
        for k in _re.split(r"[,、・/\s]+", kw):
            k = k.strip()
            if not k:
                continue
            roles_found.add(k)
            # 複合キーワードに役割語が含まれるか確認（例: "兄弟関係" に "弟" が含まれる）
            for role in _RECIPROCAL_ROLES:
                if role in k and k != role:
                    roles_found.add(role)

        # ② description の「src は tgt の[役割]」パターンから役割語を抽出
        #    例: "孫悟天は孫悟飯の弟です" → role="弟" を (孫悟天, 孫悟飯) に追加
        if desc and src and tgt:
            m = _re.search(
                rf"{_re.escape(src)}は{_re.escape(tgt)}の(\S+?)(?:です|。|ます|でした|だ)",
                desc,
            )
            if m:
                r = m.group(1).strip("。、")
                if r in _RECIPROCAL_ROLES:
                    roles_found.add(r)

        for r in roles_found:
            relation_roles.setdefault((src, tgt), set()).add(r)

    # エンティティ名リスト（検索用）
    all_names = [e.get("name", "").strip() for e in entities]

    fixed: list[dict] = []
    for ent in entities:
        ent_name = ent.get("name", "").strip()
        desc = ent.get("description", "")

        new_desc = desc
        corrected = False

        for role, inv_role in _RECIPROCAL_ROLES.items():
            if corrected:
                break
            for other in all_names:
                if not other or other == ent_name:
                    continue
                # \S+ が日本語で貪欲マッチする問題を避けるため、
                # 既知エンティティ名 + "の" + 役割語 を直接文字列検索する
                search_str = f"{other}の{role}"
                if search_str not in desc:
                    continue

                # desc に「otherの{role}」が含まれる → 方向を relation と照合
                # direct: (ent_name → other) に role がある → description が正しい方向
                direct_ok = role in relation_roles.get((ent_name, other), set())
                # reverse: (other → ent_name) に role がある → ent_name は other の role
                #           つまり description は逆向きになっている
                reverse_has_role = role in relation_roles.get((other, ent_name), set())

                if reverse_has_role and not direct_ok:
                    # 矛盾: other が ent_name の role なのに、description は逆
                    new_desc = new_desc.replace(search_str, f"{other}の{inv_role}")
                    print(
                        f"[GraphRAG] description 修正: {ent_name} "
                        f"「{other}の{role}」→「{other}の{inv_role}」"
                    )
                    corrected = True
                    break

        fixed.append({**ent, "description": new_desc})
    return fixed


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

    # 相互関係の逆転を後処理で修正（例: 「孫悟飯は孫悟天の弟」→「孫悟飯は孫悟天の兄」）
    entities = _fix_reciprocal_descriptions(entities, relations)

    return entities, relations

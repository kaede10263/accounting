import asyncio
import base64
import hmac
import json
import logging
import os
import re
import ssl
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

import certifi
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field


load_dotenv()

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
CHART_DIR = Path(os.getenv("CHART_DIR", "static/charts"))
STATIC_DIR = CHART_DIR.parent
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")


@dataclass
class ProcessingContext:
    actor_user_id: str
    chat_id: str
    reply_token: str
    source_type: str
    raw_text: str
    status: str
    started_at: datetime
    interim_replied: bool = False
    final_replied: bool = False


in_flight_tasks: dict[str, dict[str, object]] = {}
pending_setting_changes_by_user: dict[str, dict[str, str]] = {}
pending_chart_cleanup_by_user: set[str] = set()

Currency = Literal["TWD", "USD", "JPY", "CNY", "EUR", "OTHER"]
Intent = Literal["create", "update", "delete", "summary", "list", "chat"]
Action = Literal[
    "create_expense",
    "query_expenses",
    "list_expenses",
    "top_expense",
    "delete_expense",
    "create_payable",
    "query_payables",
    "mark_payable_paid",
    "create_income",
    "query_incomes",
    "list_incomes",
    "query_balance",
    "query_available_cash",
    "ask_available_investment_cash",
    "list_duplicates",
    "delete_duplicates",
    "chat",
]
PayableAction = Literal["create_payable", "query_payables", "mark_payable_paid"]
PayableStatus = Literal["unpaid", "paid", "all"]
PlanOperation = Literal["create", "query", "update", "delete", "chat"]
PlanTarget = Literal["expenses", "incomes", "payables", "balance", "duplicates", "chat"]
PlanDateRangeType = Literal[
    "today",
    "yesterday",
    "this_month",
    "last_month",
    "custom",
    "unspecified",
]
PlanAggregation = Literal["none", "sum", "list", "max", "balance", "status"]
Category = Literal[
    "\u9910\u98f2",
    "\u4ea4\u901a",
    "\u8cfc\u7269",
    "\u751f\u6d3b\u7528\u54c1",
    "\u91ab\u7642",
    "\u5a1b\u6a02",
    "\u6559\u80b2",
    "\u65c5\u904a",
    "\u623f\u79df",
    "\u6c34\u8cbb",
    "\u96fb\u8cbb",
    "\u74e6\u65af\u8cbb",
    "\u96fb\u8a71\u8cbb",
    "\u7db2\u8def\u8cbb",
    "\u4fdd\u96aa",
    "\u4fe1\u7528\u5361",
    "\u8cb8\u6b3e",
    "\u4fdd\u6bcd\u8cbb",
    "\u5176\u4ed6",
    "餐飲",
    "交通",
    "購物",
    "生活用品",
    "醫療",
    "娛樂",
    "教育",
    "旅遊",
    "房租",
    "水電瓦斯",
    "保險",
    "信用卡",
    "貸款",
    "保母費",
    "幼稚園",
    "管理費",
    "其他",
]


class ExpenseEntry(BaseModel):
    date: str = Field(description="Expense date in YYYY-MM-DD format.")
    time: str | None = Field(default=None, description="Expense time in HH:MM format, if any.")
    amount: int = Field(ge=0, description="Expense amount as an integer.")
    currency: Currency
    category: Category
    merchant: str | None = Field(description="Merchant or payee, if any.")
    note: str | None = Field(description="Extra note, if any.")
    confidence: float = Field(ge=0, le=1)


ExpenseDateRangeType = Literal[
    "today",
    "yesterday",
    "last_2_days",
    "this_week",
    "this_month",
    "last_6_months",
    "specific_months",
    "custom",
    "unspecified",
]
ExpenseQueryMode = Literal["list_detail", "aggregate", "grouped_aggregate"]
ExpenseMetric = Literal["sum", "count", "avg", "max", "min"]
ExpenseGroupBy = Literal["day", "week", "month", "category", "merchant"]
ChartType = Literal["none", "line", "bar", "pie"]
ExpenseAggregation = Literal["sum", "count", "list", "top", "sum_and_ratio", "category_breakdown"]
RatioDenominator = Literal["all_expenses", "filtered_expenses", "included_expenses", "none"]
ExpenseSortBy = Literal["date", "time", "amount", "created_at", "group", "total", "count"]
SortDirection = Literal["asc", "desc"]


class DateRange(BaseModel):
    start_date: str = Field(description="YYYY-MM-DD")
    end_date: str = Field(description="YYYY-MM-DD")


class ExpenseQuery(BaseModel):
    mode: ExpenseQueryMode = "aggregate"
    metric: ExpenseMetric = "sum"
    group_by: list[ExpenseGroupBy] = Field(default_factory=list)
    date_range_type: ExpenseDateRangeType
    date_ranges: list[DateRange] = Field(default_factory=list)
    category: Category | None = None
    include_categories: list[Category] = Field(default_factory=list)
    exclude_categories: list[Category] = Field(default_factory=list)
    merchant: str | None = None
    include_keywords: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)
    min_amount: int | None = None
    max_amount: int | None = None
    include_ratio: bool = False
    wants_chart: bool = False
    chart_type: ChartType = "none"
    aggregation: ExpenseAggregation = "sum"
    ratio_denominator: RatioDenominator = "none"
    limit: int = Field(default=15, ge=1, le=100)
    sort_by: ExpenseSortBy = "date"
    sort_direction: SortDirection = "desc"
    confidence: float = Field(ge=0, le=1)
    reason: str | None = None


EXPENSE_CATEGORY_VALUES = [
    "\u9910\u98f2",
    "\u4ea4\u901a",
    "\u8cfc\u7269",
    "\u751f\u6d3b\u7528\u54c1",
    "\u91ab\u7642",
    "\u5a1b\u6a02",
    "\u6559\u80b2",
    "\u65c5\u904a",
    "\u623f\u79df",
    "\u6c34\u8cbb",
    "\u96fb\u8cbb",
    "\u74e6\u65af\u8cbb",
    "\u96fb\u8a71\u8cbb",
    "\u7db2\u8def\u8cbb",
    "\u4fdd\u96aa",
    "\u4fe1\u7528\u5361",
    "\u8cb8\u6b3e",
    "\u4fdd\u6bcd\u8cbb",
    "\u5176\u4ed6",
]


class DeleteResult(BaseModel):
    deleted: bool
    expense: ExpenseEntry | None = None
    expense_id: int | None = None
    reason: str | None = None
    similar_count: int = 0


class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    reason: str | None = None


class AgentPlan(BaseModel):
    operation: PlanOperation
    target: PlanTarget
    should_mutate_db: bool
    confidence: float = Field(ge=0, le=1)
    date_range_type: PlanDateRangeType
    start_date: str | None = None
    end_date: str | None = None
    category: str | None = None
    merchant: str | None = None
    keywords: list[str] = Field(default_factory=list)
    amount: int | None = None
    currency: Currency
    note: str | None = None
    aggregation: PlanAggregation
    reason: str | None = None


class ActionRoute(BaseModel):
    action: Action
    should_mutate_db: bool
    confidence: float = Field(ge=0, le=1)
    reason: str | None = None
    item_type: str | None = None
    owner: str | None = None
    bank: str | None = None
    amount: int | None = None
    due_date: str | None = None
    date_text: str | None = None
    category: str | None = None
    income_type: str | None = None
    status: str | None = None
    purchase_purpose: str | None = None


class PayableQuery(BaseModel):
    item_type: str | None = None
    owner: str | None = None
    bank: str | None = None
    status: PayableStatus = "all"
    date_range_type: Literal["this_month", "custom", "unspecified"] = "this_month"
    start_date: str | None = None
    end_date: str | None = None
    confidence: float = Field(ge=0, le=1)
    reason: str | None = None


app = FastAPI(title="LINE Accounting Bot", version="0.1.0")
CHART_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
logger = logging.getLogger("line-accounting-bot")
pending_delete_by_user: dict[str, int] = {}
pending_expense_by_user: dict[str, dict[str, object]] = {}
pending_duplicate_delete_by_user: dict[str, list[tuple[str, int]]] = {}
last_query_by_user: dict[str, dict[str, object]] = {}


def get_app_env() -> str:
    return os.getenv("APP_ENV", "dev").lower()


def get_db_path() -> str:
    app_env = get_app_env()
    explicit_path = os.getenv("DB_PATH") or os.getenv("DATABASE_URL") or os.getenv("DATABASE_PATH")
    if explicit_path:
        if app_env == "test" and os.path.basename(explicit_path) == "accounting.db":
            raise RuntimeError("APP_ENV=test cannot use prod DB accounting.db.")
        return explicit_path
    if app_env == "test":
        return "accounting_test.db"
    if app_env == "prod":
        return "accounting.db"
    return "accounting_dev.db"


def get_ssl_verify() -> ssl.SSLContext | str | bool:
    if get_ssl_verify_setting().lower() in {"0", "false", "no", "off"}:
        return False

    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return certifi.where()


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.row_factory = sqlite3.Row
    return conn


def get_setting(key: str, default: str | None = None) -> str | None:
    try:
        with get_db() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            return str(row["value"]) if row else default
    except sqlite3.Error:
        return default


def set_setting(key: str, value: str) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, value, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def get_effective_setting(key: str, env_name: str | None = None, default: str | None = None) -> str | None:
    setting_value = get_setting(key)
    if setting_value is not None:
        return setting_value
    return os.getenv(env_name or key, default)


def get_current_model() -> str:
    return str(get_effective_setting("OPENAI_MODEL", "OPENAI_MODEL", "gpt-4.1-mini"))


def get_ssl_verify_setting() -> str:
    return str(get_effective_setting("SSL_VERIFY", "SSL_VERIFY", "true"))


def get_reminder_interval_hours() -> int:
    value = str(get_effective_setting("REMINDER_INTERVAL_HOURS", "REMINDER_INTERVAL_HOURS", "12"))
    try:
        return max(1, min(168, int(value)))
    except ValueError:
        return 12


def get_processing_timeout_seconds() -> int:
    value = str(get_effective_setting("PROCESSING_TIMEOUT_SECONDS", "PROCESSING_TIMEOUT_SECONDS", "3"))
    try:
        return max(1, min(30, int(value)))
    except ValueError:
        return 3


def get_public_base_url() -> str | None:
    return get_effective_setting("PUBLIC_BASE_URL", "PUBLIC_BASE_URL", PUBLIC_BASE_URL)


def get_chart_retention_days() -> int:
    value = str(get_effective_setting("CHART_RETENTION_DAYS", "CHART_RETENTION_DAYS", "7"))
    try:
        return max(0, int(value))
    except ValueError:
        return 7


def chart_cache_key(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return sha256(raw.encode("utf-8")).hexdigest()[:16]


def chart_storage_stats() -> dict[str, float | int]:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    files = list(CHART_DIR.glob("*.png"))
    total_bytes = 0
    for path in files:
        try:
            total_bytes += path.stat().st_size
        except OSError:
            logger.exception("Failed to stat chart file: %s", path)
    return {"count": len(files), "size_bytes": total_bytes, "size_mb": total_bytes / (1024 * 1024)}


def cleanup_old_charts(days: int = 7) -> int:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff_ts = now_ts - max(0, days) * 86400
    deleted_count = 0
    for path in CHART_DIR.glob("*.png"):
        try:
            if days <= 0 or path.stat().st_mtime <= cutoff_ts:
                path.unlink()
                deleted_count += 1
        except OSError:
            logger.exception("Failed to delete old chart file: %s", path)
    return deleted_count


def log_usage(
    provider: str,
    event_type: str,
    detail: str | None = None,
    success: bool = True,
    latency_ms: int | None = None,
) -> None:
    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO usage_logs (event_type, provider, detail, success, latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    provider,
                    detail,
                    1 if success else 0,
                    latency_ms,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
    except sqlite3.Error:
        logger.exception("Failed to write usage log provider=%s event_type=%s", provider, event_type)


def init_db() -> None:
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                provider TEXT NOT NULL,
                detail TEXT,
                success INTEGER NOT NULL DEFAULT 1,
                latency_ms INTEGER,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line_user_id TEXT,
                message_id TEXT,
                raw_text TEXT NOT NULL,
                date TEXT NOT NULL,
                expense_time TEXT,
                amount INTEGER NOT NULL,
                currency TEXT NOT NULL,
                category TEXT NOT NULL,
                merchant TEXT,
                note TEXT,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(expenses)").fetchall()
        }
        if "expense_time" not in columns:
            conn.execute("ALTER TABLE expenses ADD COLUMN expense_time TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expenses_date ON expenses(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expenses_message_id ON expenses(message_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS payables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line_user_id TEXT NOT NULL,
                item_type TEXT NOT NULL,
                amount INTEGER NOT NULL,
                currency TEXT NOT NULL,
                due_date TEXT NOT NULL,
                owner TEXT,
                bank TEXT,
                note TEXT,
                message_id TEXT,
                status TEXT NOT NULL DEFAULT 'unpaid',
                paid_at TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        payable_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(payables)").fetchall()
        }
        if "owner" not in payable_columns:
            conn.execute("ALTER TABLE payables ADD COLUMN owner TEXT")
        if "bank" not in payable_columns:
            conn.execute("ALTER TABLE payables ADD COLUMN bank TEXT")
        if "message_id" not in payable_columns:
            conn.execute("ALTER TABLE payables ADD COLUMN message_id TEXT")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_payables_message_id
            ON payables(line_user_id, message_id)
            WHERE message_id IS NOT NULL
            """
        )
        try:
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_payables_dedupe
                ON payables(line_user_id, item_type, amount, due_date, COALESCE(owner, ''), COALESCE(bank, ''), status)
                WHERE status = 'unpaid'
                """
            )
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            logger.warning("Could not create idx_payables_dedupe; create_payable will use manual dedupe.")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS payable_reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                payable_id INTEGER NOT NULL,
                remind_days_before INTEGER NOT NULL,
                reminded_on TEXT NOT NULL,
                UNIQUE(payable_id, remind_days_before, reminded_on)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS payable_drafts (
                line_user_id TEXT PRIMARY KEY,
                item_type TEXT NOT NULL,
                amount INTEGER NOT NULL,
                currency TEXT NOT NULL,
                owner TEXT,
                bank TEXT,
                note TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line_user_id TEXT,
                raw_text TEXT NOT NULL,
                income_date TEXT NOT NULL,
                amount INTEGER NOT NULL,
                currency TEXT NOT NULL,
                income_type TEXT NOT NULL,
                item_name TEXT,
                owner TEXT,
                category TEXT,
                note TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        income_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(incomes)").fetchall()
        }
        if "item_name" not in income_columns:
            conn.execute("ALTER TABLE incomes ADD COLUMN item_name TEXT")
        if "owner" not in income_columns:
            conn.execute("ALTER TABLE incomes ADD COLUMN owner TEXT")
        if "category" not in income_columns:
            conn.execute("ALTER TABLE incomes ADD COLUMN category TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_payables_due_date ON payables(due_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incomes_date ON incomes(income_date)")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_incomes_dedupe
            ON incomes(line_user_id, income_date, item_name, amount, owner, category)
            """
        )
        conn.commit()


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Starting app env=%s db_path=%s", get_app_env(), get_db_path())
    init_db()
    try:
        backfill_payable_metadata()
        backfill_utility_categories()
        deleted_charts = cleanup_old_charts(get_chart_retention_days())
        logger.info("Cleaned old chart cache files count=%s", deleted_charts)
    except sqlite3.OperationalError as exc:
        if "database is locked" in str(exc).lower():
            logger.warning("Skip startup backfill because SQLite database is locked; app will continue.")
        else:
            raise
    asyncio.create_task(reminder_loop())


def verify_line_signature(body: bytes, signature: str | None) -> bool:
    load_dotenv(override=True)
    channel_secret = os.getenv("LINE_CHANNEL_SECRET", "")
    if not channel_secret or not signature:
        return False

    digest = hmac.new(channel_secret.encode("utf-8"), body, sha256).digest()
    expected_signature = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected_signature, signature)


def get_openai_client() -> OpenAI:
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    return OpenAI(
        api_key=api_key,
        http_client=httpx.Client(verify=get_ssl_verify(), timeout=30),
    )


def create_openai_chat_completion(**kwargs):
    model = kwargs.get("model") or get_current_model()
    kwargs["model"] = model
    start = datetime.now(timezone.utc)
    try:
        response = get_openai_client().chat.completions.create(**kwargs)
        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        log_usage("openai", "chat_completion", detail=str(model), success=True, latency_ms=latency_ms)
        return response
    except Exception:
        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        log_usage("openai", "chat_completion", detail=str(model), success=False, latency_ms=latency_ms)
        raise


def parse_expense_text(text: str) -> ExpenseEntry:
    today = date.today().isoformat()
    schema = {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Expense date in YYYY-MM-DD format."},
            "time": {"type": ["string", "null"], "description": "Expense time in HH:MM format, if any."},
            "amount": {"type": "integer", "minimum": 0},
            "currency": {"type": "string", "enum": ["TWD", "USD", "JPY", "CNY", "EUR", "OTHER"]},
            "category": {
                "type": "string",
                "enum": EXPENSE_CATEGORY_VALUES,
            },            "merchant": {"type": ["string", "null"]},
            "note": {"type": ["string", "null"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["date", "time", "amount", "currency", "category", "merchant", "note", "confidence"],
        "additionalProperties": False,
    }

    log_usage("openai", "chat_completion", detail=get_current_model())
    response = get_openai_client().chat.completions.create(
        model=get_current_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "你是台灣記帳 Bot 的支出解析器，只輸出 schema JSON。"
                    f"今天是 {today}。"
                    "分類必須使用 enum。電話費、手機費、電信費、網路費要分成電話費或網路費；"
                    "電費、台電、電力費要分成電費；水費、自來水要分成水費；"
                    "瓦斯費、天然氣、桶裝瓦斯要分成瓦斯費。"
                    "不要把電話費、水費、電費、瓦斯費、網路費歸到生活用品。"                    "你是台灣使用者的記帳文字解析器。"
                    "請把使用者的自然語言消費記錄轉成結構化資料。"
                    "若文字沒有明確日期，date 使用今天："
                    f"{today}。"
                    "若有明確時間，time 使用 24 小時制 HH:MM；若沒有明確時間，time 為 null。"
                    "若沒有幣別，currency 使用 TWD。"
                    "merchant 是店家、收款方或對象；note 放無法歸類但有用的補充資訊。"
                    "只輸出符合 schema 的結果。"
                ),
            },
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "expense_entry", "strict": True, "schema": schema},
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI returned an empty response.")
    return ExpenseEntry.model_validate_json(content)


def classify_intent_with_openai(text: str) -> IntentResult:
    schema = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["create", "update", "delete", "summary", "list", "chat"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": ["string", "null"]},
        },
        "required": ["intent", "confidence", "reason"],
        "additionalProperties": False,
    }
    log_usage("openai", "chat_completion", detail=get_current_model())
    response = get_openai_client().chat.completions.create(
        model=get_current_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "你是 LINE 記帳 Bot 的意圖分類器。"
                    "請只判斷使用者意圖，不要產生記帳資料。"
                    "create=新增一筆支出，通常包含金額。"
                    "update=修改既有記帳。"
                    "delete=刪除既有記帳。"
                    "summary=查詢總額或統計。"
                    "list=列出明細。"
                    "chat=一般聊天、說明、或無法確定是否要動資料庫。"
                    "如果使用者是在問問題，不要分類成 create。"
                ),
            },
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "intent_result", "strict": True, "schema": schema},
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI returned an empty intent response.")
    return IntentResult.model_validate_json(content)


def classify_intent(text: str) -> IntentResult:
    normalized = text.strip().lower()
    has_amount = re.search(r"\d+", text) is not None
    is_question = any(mark in text for mark in ("?", "？", "嗎", "多少", "哪", "什麼", "如何", "怎麼"))

    try:
        result = classify_intent_with_openai(text)
        if result.intent == "create" and not has_amount:
            return IntentResult(intent="chat", confidence=0.70, reason="create without amount is unsafe")
        if result.intent == "create" and is_question:
            return IntentResult(intent="chat", confidence=0.70, reason="question should not create expense")
        if result.intent == "chat" and is_list_intent(text):
            return IntentResult(intent="list", confidence=0.80, reason="local correction after OpenAI chat")
        if result.intent in {"chat", "list"} and is_summary_intent(text):
            return IntentResult(intent="summary", confidence=0.80, reason="local correction after OpenAI chat")
        if result.intent == "chat" and is_delete_intent(text):
            return IntentResult(intent="delete", confidence=0.80, reason="local correction after OpenAI chat")
        if result.intent == "chat" and is_update_intent(text):
            return IntentResult(intent="update", confidence=0.80, reason="local correction after OpenAI chat")
        return result
    except Exception:
        logger.exception("OpenAI intent classification failed; using local fallback for text: %s", text)

    if is_delete_intent(text):
        return IntentResult(intent="delete", confidence=0.80, reason="local fallback delete keyword")
    if is_update_intent(text):
        return IntentResult(intent="update", confidence=0.75, reason="local fallback update keyword")
    if is_list_intent(text):
        return IntentResult(intent="list", confidence=0.80, reason="local fallback list keyword")
    if is_summary_intent(text):
        return IntentResult(intent="summary", confidence=0.80, reason="local fallback summary keyword")
    if has_amount and not is_question:
        return IntentResult(intent="create", confidence=0.70, reason="local fallback amount")
    if normalized in {"help", "hi", "hello", "嗨", "你好", "說明"}:
        return IntentResult(intent="chat", confidence=0.80, reason="local fallback chat/help keyword")

    return IntentResult(intent="chat", confidence=0.50, reason="local fallback default")


def is_question_text(text: str) -> bool:
    stripped = text.strip()
    normalized = re.sub(r"\s+", "", stripped)
    if is_paid_text(stripped) and not any(keyword in normalized for keyword in ("\u55ce", "\u662f\u5426", "?", "\uff1f")):
        return False
    if any(mark in stripped for mark in ("?", "？", "嗎", "么", "是不是", "有沒有")):
        return True
    question_terms = ("多少", "哪", "哪些", "有哪", "有什麼", "怎麼", "如何", "是否", "了嗎")
    return any(term in stripped for term in question_terms)


def build_agent_plan_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["create", "query", "update", "delete", "chat"]},
            "target": {
                "type": "string",
                "enum": ["expenses", "incomes", "payables", "balance", "duplicates", "chat"],
            },
            "should_mutate_db": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "date_range_type": {
                "type": "string",
                "enum": ["today", "yesterday", "this_month", "last_month", "custom", "unspecified"],
            },
            "start_date": {"type": ["string", "null"]},
            "end_date": {"type": ["string", "null"]},
            "category": {"type": ["string", "null"]},
            "merchant": {"type": ["string", "null"]},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "amount": {"type": ["integer", "null"]},
            "currency": {"type": "string", "enum": ["TWD", "USD", "JPY", "CNY", "EUR", "OTHER"]},
            "note": {"type": ["string", "null"]},
            "aggregation": {"type": "string", "enum": ["none", "sum", "list", "max", "balance", "status"]},
            "reason": {"type": ["string", "null"]},
        },
        "required": [
            "operation",
            "target",
            "should_mutate_db",
            "confidence",
            "date_range_type",
            "start_date",
            "end_date",
            "category",
            "merchant",
            "keywords",
            "amount",
            "currency",
            "note",
            "aggregation",
            "reason",
        ],
        "additionalProperties": False,
    }


def build_agent_planner_prompt(today: str) -> str:
    return f"""
重要規則：
- 「提醒我 7/8 要繳房貸 62218」是 create payables，不是 create expenses。
- 「7/8 房貸 62218」是 create payables，不是 create expenses。
- 「房貸 62218 7/8」是 create payables，不是 create expenses。
- 「信貸 8000 7/16」是 create payables，不是 create expenses。
- payables 的 item_type 要保留原詞：房貸、信貸、貸款、信用卡、保母費等。

範例：
使用者：提醒我 7/8 要繳房貸 62218
輸出：{{"operation":"create","target":"payables","should_mutate_db":true,"confidence":0.96,"date_range_type":"custom","start_date":"2026-07-08","end_date":"2026-07-08","category":"房貸","merchant":null,"keywords":[],"amount":62218,"currency":"TWD","note":"房貸","aggregation":"none","reason":"建立房貸待繳提醒"}}

使用者：信貸 8000 7/16
輸出：{{"operation":"create","target":"payables","should_mutate_db":true,"confidence":0.95,"date_range_type":"custom","start_date":"2026-07-16","end_date":"2026-07-16","category":"信貸","merchant":null,"keywords":[],"amount":8000,"currency":"TWD","note":"信貸","aggregation":"none","reason":"建立信貸待繳提醒"}}

你是台灣 LINE 記帳 Bot 的任務規劃器。
今天是 {today}。
你的工作是把使用者訊息轉成固定 JSON AgentPlan。
你不能回答使用者，也不能產生 SQL。
Python 程式會根據你的 JSON 執行資料庫操作。

資料庫概念：
- expenses：已發生支出，例如早餐、午餐、交通、加油、買衣服、醫療、娛樂。
- incomes：收入，例如薪水、薪資、業外收入、政府補助。
- payables：未繳款或待繳款，例如信用卡、房貸、保母費、幼稚園、管理費。
- balance：收支狀態，不是單一 table，而是 incomes - expenses - unpaid payables。
- duplicates：重複資料檢查，例如重複的薪水、重複支出。

規則：
1. 使用者問「花多少」「多少錢」「明細」「列出」「最貴」通常是 query expenses。
2. 使用者問「收入」「薪水」「薪資收入」通常是 query/create incomes，依語氣判斷。
3. 使用者問「未繳」「還沒繳」「繳了嗎」「待繳」通常是 query payables。
4. 使用者問「透支」「超支」「盈餘」「剩多少」「還有多少錢」「夠不夠」「可動用多少」是 query balance。
5. 使用者問「列出重複」「重複資料」是 query duplicates。
6. 使用者明確說「刪除」「移除」「只留一筆其他刪除」是 delete，但要由 Python 做確認流程。
7. 任何問句都不可設定 should_mutate_db=true。
8. 只有明確記帳語句才可以 should_mutate_db=true，例如「午餐 120」「薪水 50000」「信用卡 1804 7/15」。
9. 大額支出、category=其他、confidence<0.85 時，Python 會再確認；你仍可輸出 create expenses，但 confidence 要反映不確定性。
10. 不要產生 SQL。
11. 不要自己計算總額。
12. 不要自由回答，只輸出 JSON。

分類規則：
- 餐飲：早餐、午餐、晚餐、飲料、咖啡、點心、宵夜、餐費
- 交通：交通、加油、停車、捷運、公車、計程車、Uber、高鐵、台鐵、火車
- 購物：購物、買衣服、衣服、治裝、服飾、鞋子、褲子、外套
- 生活用品：全聯、家樂福、超市、日用品、衛生紙、牛奶若語境是家庭採買
- 醫療：看醫生、掛號、藥、牙醫
- 信用卡：信用卡、卡費
- 貸款：房貸、信貸、貸款
- 保母費：保母、保母費
- 其他：無法判斷分類時

範例：
使用者：今天交通花了多少錢?
輸出：
{{"operation":"query","target":"expenses","should_mutate_db":false,"confidence":0.95,"date_range_type":"today","start_date":null,"end_date":null,"category":"交通","merchant":null,"keywords":[],"amount":null,"currency":"TWD","note":null,"aggregation":"sum","reason":"查詢今天交通分類支出總額"}}

使用者：這個月是否透支?
輸出：
{{"operation":"query","target":"balance","should_mutate_db":false,"confidence":0.96,"date_range_type":"this_month","start_date":null,"end_date":null,"category":null,"merchant":null,"keywords":[],"amount":null,"currency":"TWD","note":null,"aggregation":"balance","reason":"查詢本月收支是否不足"}}

使用者：有多少錢可以買玩具?
輸出：
{{"operation":"query","target":"balance","should_mutate_db":false,"confidence":0.93,"date_range_type":"this_month","start_date":null,"end_date":null,"category":null,"merchant":null,"keywords":[],"amount":null,"currency":"TWD","note":"買玩具","aggregation":"balance","reason":"使用者詢問目前可動用金額"}}

使用者：薪水 50000
輸出：
{{"operation":"create","target":"incomes","should_mutate_db":true,"confidence":0.94,"date_range_type":"today","start_date":null,"end_date":null,"category":null,"merchant":null,"keywords":[],"amount":50000,"currency":"TWD","note":"薪水","aggregation":"none","reason":"新增薪水收入"}}

使用者：買車 100萬
輸出：
{{"operation":"create","target":"expenses","should_mutate_db":true,"confidence":0.72,"date_range_type":"today","start_date":null,"end_date":null,"category":"其他","merchant":null,"keywords":[],"amount":1000000,"currency":"TWD","note":"買車","aggregation":"none","reason":"大額支出且可能是計畫，建議由 Python 進入確認流程"}}
""".strip()


def plan_item_type(plan: AgentPlan) -> str | None:
    for value in (plan.category, plan.note, plan.merchant):
        if value:
            return value
    return None


def agent_plan_to_action_route(plan: AgentPlan, raw_text: str) -> ActionRoute:
    should_mutate_db = plan.should_mutate_db and plan.operation in {"create", "update", "delete"}
    if plan.operation == "query":
        should_mutate_db = False

    reason = plan.reason or "OpenAI AgentPlan"

    if plan.operation == "chat" or plan.target == "chat":
        return ActionRoute(action="chat", should_mutate_db=False, confidence=plan.confidence, reason=reason)

    if plan.operation == "delete":
        if plan.target == "duplicates":
            action: Action = "delete_duplicates"
        elif plan.target == "expenses":
            action = "delete_expense"
        else:
            action = "chat"
        return ActionRoute(
            action=action,
            should_mutate_db=should_mutate_db if action == "delete_expense" else False,
            confidence=plan.confidence,
            reason=reason,
            category=plan.category,
            amount=plan.amount,
            purchase_purpose=plan.note,
        )

    if plan.operation == "update":
        action = "mark_payable_paid" if plan.target == "payables" else "chat"
        return ActionRoute(
            action=action,
            should_mutate_db=should_mutate_db,
            confidence=plan.confidence,
            reason=reason,
            item_type=plan_item_type(plan),
            amount=plan.amount,
            due_date=plan.end_date or plan.start_date,
            category=plan.category,
        )

    if plan.operation == "create":
        if plan.target == "incomes":
            action = "create_income"
        elif plan.target == "payables":
            action = "create_payable"
        elif plan.target == "expenses":
            action = "create_expense"
        else:
            action = "chat"
        return ActionRoute(
            action=action,
            should_mutate_db=should_mutate_db,
            confidence=plan.confidence,
            reason=reason,
            item_type=plan_item_type(plan) if plan.target == "payables" else None,
            amount=plan.amount,
            due_date=plan.end_date or plan.start_date,
            category=plan.category,
            purchase_purpose=plan.note,
        )

    if plan.target == "duplicates":
        action = "list_duplicates"
    elif plan.target == "payables":
        action = "query_payables"
    elif plan.target == "balance":
        if plan.note or any(keyword in raw_text for keyword in ("可動用", "可以買", "還有多少錢", "剩多少", "買")):
            action = "query_available_cash"
        else:
            action = "query_balance"
    elif plan.target == "expenses":
        if plan.aggregation == "max":
            action = "top_expense"
        elif plan.aggregation == "list":
            action = "list_expenses"
        else:
            action = "query_expenses"
    elif plan.target == "incomes":
        action = "list_incomes" if plan.aggregation == "list" else "query_incomes"
    else:
        action = "chat"

    return ActionRoute(
        action=action,
        should_mutate_db=False,
        confidence=plan.confidence,
        reason=reason,
        item_type=plan_item_type(plan) if plan.target == "payables" else None,
        amount=plan.amount,
        due_date=plan.end_date or plan.start_date,
        category=plan.category,
        income_type=plan.category or plan.note or plan.merchant,
        status="unpaid" if plan.target == "payables" else None,
        purchase_purpose=plan.note,
    )


def route_action_with_openai(text: str) -> ActionRoute:
    today = date.today().isoformat()
    log_usage("openai", "chat_completion", detail=get_current_model())
    response = get_openai_client().chat.completions.create(
        model=get_current_model(),
        messages=[
            {"role": "system", "content": build_agent_planner_prompt(today)},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "agent_plan",
                "strict": True,
                "schema": build_agent_plan_schema(),
            },
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI returned an empty agent plan.")
    plan = AgentPlan.model_validate_json(content)
    return agent_plan_to_action_route(plan, text)

def fallback_action_route(text: str) -> ActionRoute:
    has_amount = parse_amount(text) is not None
    payable_amount = parse_payable_amount(text)
    if is_delete_duplicate_text(text):
        return ActionRoute(action="delete_duplicates", should_mutate_db=False, confidence=0.90, reason="local fallback duplicate delete")
    if is_duplicate_data_text(text):
        return ActionRoute(action="list_duplicates", should_mutate_db=False, confidence=0.90, reason="local fallback duplicate list")
    if is_available_investment_cash_query(text):
        return ActionRoute(action="ask_available_investment_cash", should_mutate_db=False, confidence=0.85, reason="local fallback investment cash query")
    if is_available_purchase_cash_query(text):
        return ActionRoute(
            action="query_available_cash",
            should_mutate_db=False,
            confidence=0.85,
            reason="local fallback purchase cash query",
            purchase_purpose=extract_purchase_purpose(text),
        )
    if is_balance_query(text):
        return ActionRoute(action="query_balance", should_mutate_db=False, confidence=0.80, reason="local fallback balance query")
    if is_question_text(text):
        if get_payable_type(text):
            return ActionRoute(
                action="query_payables",
                should_mutate_db=False,
                confidence=0.65,
                reason="local fallback payable question",
                item_type=get_payable_type(text),
                owner=get_owner(text),
                bank=get_bank(text),
                status="unpaid",
            )
        if "貴" in text:
            return ActionRoute(action="top_expense", should_mutate_db=False, confidence=0.65, reason="local fallback top query")
        return ActionRoute(action="query_expenses", should_mutate_db=False, confidence=0.60, reason="local fallback question")
    if get_payable_type(text) and payable_amount is not None:
        return ActionRoute(
            action="create_payable",
            should_mutate_db=True,
            confidence=0.70,
            reason="local fallback payable with amount",
            item_type=get_payable_type(text),
            owner=get_owner(text),
            bank=get_bank(text),
            amount=payable_amount,
            due_date=parse_due_date(text),
        )
    if is_delete_intent(text):
        return ActionRoute(action="delete_expense", should_mutate_db=True, confidence=0.70, reason="local fallback delete")
    if is_list_intent(text):
        return ActionRoute(action="list_expenses", should_mutate_db=False, confidence=0.70, reason="local fallback list")
    if is_summary_intent(text):
        return ActionRoute(action="query_expenses", should_mutate_db=False, confidence=0.70, reason="local fallback summary")
    if has_amount:
        return ActionRoute(action="create_expense", should_mutate_db=True, confidence=0.60, reason="local fallback amount")
    return ActionRoute(action="chat", should_mutate_db=False, confidence=0.50, reason="local fallback chat")


def is_general_unpaid_payable_query(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return any(keyword in normalized for keyword in ("\u9084\u6709\u54ea\u4e9b\u6c92\u7e73", "\u54ea\u4e9b\u6c92\u7e73", "\u76ee\u524d\u672a\u7e73", "\u672a\u7e73\u8cbb\u7528", "\u5f85\u7e73"))


def is_payable_create_text(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    if any(keyword in normalized for keyword in ("\u63d0\u9192\u6211", "\u8981\u7e73", "\u5f85\u7e73", "\u7e73\u8cbb")):
        return True
    return get_payable_type(text) is not None and parse_payable_amount(text) is not None and parse_due_date(text) is not None


def is_statistical_expense_query(text: str) -> bool:
    clean_text, wants_chart, _ = extract_expense_chart_request(text)
    normalized = re.sub(r"\s+", "", clean_text)
    group_by = parse_expense_group_by(clean_text)
    include_ratio = is_ratio_query(clean_text) or group_by == ["category"]
    if wants_chart:
        _, wants_chart, _ = extract_expense_chart_request(text, group_by, include_ratio)
    if wants_chart or group_by:
        return True
    return any(keyword in normalized for keyword in ("\u7d71\u8a08", "\u7e3d\u5171", "\u7e3d\u82b1\u8cbb", "\u5e73\u5747", "\u4f54\u6bd4", "\u6bd4\u4f8b"))


def route_action(text: str, line_user_id: str | None) -> ActionRoute:
    pending = get_payable_draft(line_user_id) if line_user_id else None
    due_date = parse_due_date(text)
    payable_type = get_payable_type(text)
    payable_amount = parse_payable_amount(text)

    if is_paid_text(text) and payable_type and not is_question_text(text):
        return ActionRoute(
            action="mark_payable_paid",
            should_mutate_db=True,
            confidence=0.98,
            reason="deterministic payable paid update",
            item_type=payable_type,
            owner=get_owner(text),
            bank=get_bank(text),
            status="paid",
        )

    if payable_type and payable_amount is not None and is_payable_create_text(text):
        return ActionRoute(
            action="create_payable",
            should_mutate_db=True,
            confidence=0.95,
            reason="deterministic payable reminder create",
            item_type=payable_type,
            amount=payable_amount,
            due_date=due_date,
            owner=get_owner(text),
            bank=get_bank(text),
        )

    if is_question_text(text) and (payable_type or is_general_unpaid_payable_query(text)):
        return ActionRoute(
            action="query_payables",
            should_mutate_db=False,
            confidence=0.95,
            reason="deterministic payable question",
            item_type=payable_type,
            status="all" if payable_type else "unpaid",
        )

    if is_income_list_query(text):
        return ActionRoute(
            action="list_incomes",
            should_mutate_db=False,
            confidence=0.95,
            reason="deterministic income list query",
            income_type=get_income_type(text),
        )

    if is_income_query(text):
        return ActionRoute(
            action="query_incomes",
            should_mutate_db=False,
            confidence=0.95,
            reason="deterministic income aggregate query",
            income_type=get_income_type(text),
        )

    try:
        route = route_action_with_openai(text)
    except Exception:
        logger.exception("OpenAI action routing failed; using local fallback for text: %s", text)
        route = fallback_action_route(text)

    if is_delete_duplicate_text(text):
        return ActionRoute(
            action="delete_duplicates",
            should_mutate_db=False,
            confidence=max(route.confidence, 0.90),
            reason="local correction: duplicate delete",
        )

    if is_duplicate_data_text(text):
        return ActionRoute(
            action="list_duplicates",
            should_mutate_db=False,
            confidence=max(route.confidence, 0.90),
            reason="local correction: duplicate list",
        )

    if is_statistical_expense_query(text):
        return ActionRoute(
            action="query_expenses",
            should_mutate_db=False,
            confidence=max(route.confidence, 0.90),
            reason="local correction: statistical expense query",
            category=route.category,
        )

    if is_available_investment_cash_query(text):
        return ActionRoute(
            action="ask_available_investment_cash",
            should_mutate_db=False,
            confidence=max(route.confidence, 0.90),
            reason="local correction: investment cash query",
        )

    if is_available_purchase_cash_query(text):
        return ActionRoute(
            action="query_available_cash",
            should_mutate_db=False,
            confidence=max(route.confidence, 0.90),
            reason="local correction: purchase cash query",
            purchase_purpose=route.purchase_purpose or extract_purchase_purpose(text),
        )

    if is_balance_query(text):
        return ActionRoute(
            action="query_balance",
            should_mutate_db=False,
            confidence=max(route.confidence, 0.85),
            reason="local correction: income/balance query",
        )

    if pending and due_date:
        return ActionRoute(
            action="create_payable",
            should_mutate_db=True,
            confidence=max(route.confidence, 0.85),
            reason="pending payable draft completed with due date",
            item_type=str(pending["item_type"]),
            amount=int(pending["amount"]),
            due_date=due_date,
        )

    if is_question_text(text) and route.should_mutate_db:
        if route.action in {"mark_payable_paid", "create_payable"} or get_payable_type(text):
            return ActionRoute(
                action="query_payables",
                should_mutate_db=False,
                confidence=max(route.confidence, 0.80),
                reason="safety: question cannot mutate payable data",
                item_type=route.item_type or get_payable_type(text),
                owner=route.owner or get_owner(text),
                bank=route.bank or get_bank(text),
                status=route.status or "unpaid",
            )
        return ActionRoute(action="chat", should_mutate_db=False, confidence=0.70, reason="safety: question cannot mutate data")

    if route.action == "create_expense" and parse_amount(text) is None:
        return ActionRoute(action="chat", should_mutate_db=False, confidence=0.70, reason="safety: expense without amount")

    if route.action == "create_payable":
        route.item_type = route.item_type or get_payable_type(text)
        route.amount = route.amount if route.amount is not None else parse_payable_amount(text)
        route.due_date = route.due_date or due_date
        route.owner = route.owner or get_owner(text)
        route.bank = route.bank or get_bank(text)
        if not route.item_type or route.amount is None:
            return ActionRoute(action="chat", should_mutate_db=False, confidence=0.65, reason="safety: incomplete payable")

    return route


def infer_category(text: str) -> Category:
    category_keywords: list[tuple[str, Category]] = [
        ("\u96fb\u8a71\u8cbb", "\u96fb\u8a71\u8cbb"),
        ("\u624b\u6a5f\u8cbb", "\u96fb\u8a71\u8cbb"),
        ("\u96fb\u4fe1\u8cbb", "\u96fb\u8a71\u8cbb"),
        ("\u7db2\u8def\u8cbb", "\u7db2\u8def\u8cbb"),
        ("\u5bec\u983b", "\u7db2\u8def\u8cbb"),
        ("\u96fb\u8cbb", "\u96fb\u8cbb"),
        ("\u53f0\u96fb", "\u96fb\u8cbb"),
        ("\u6c34\u8cbb", "\u6c34\u8cbb"),
        ("\u81ea\u4f86\u6c34", "\u6c34\u8cbb"),
        ("\u74e6\u65af\u8cbb", "\u74e6\u65af\u8cbb"),
        ("\u5929\u7136\u6c23", "\u74e6\u65af\u8cbb"),
        ("\u6876\u88dd\u74e6\u65af", "\u74e6\u65af\u8cbb"),
        ("保母", "保母費"),
        ("\u4ea4\u901a", "\u4ea4\u901a"),
        ("\u52a0\u6cb9", "\u4ea4\u901a"),
        ("\u505c\u8eca", "\u4ea4\u901a"),
        ("\u6377\u904b", "\u4ea4\u901a"),
        ("\u516c\u8eca", "\u4ea4\u901a"),
        ("\u8a08\u7a0b\u8eca", "\u4ea4\u901a"),
        ("\u9ad8\u9435", "\u4ea4\u901a"),
        ("\u53f0\u9435", "\u4ea4\u901a"),
        ("\u706b\u8eca", "\u4ea4\u901a"),
        ("\u6cb9\u9322", "\u4ea4\u901a"),
        ("\u6cb9\u8cc7", "\u4ea4\u901a"),
        ("\u9910\u98f2", "\u9910\u98f2"),
        ("\u5348\u9910", "\u9910\u98f2"),
        ("\u665a\u9910", "\u9910\u98f2"),
        ("\u65e9\u9910", "\u9910\u98f2"),
        ("\u9ede\u5fc3", "\u9910\u98f2"),
        ("\u98f2\u6599", "\u9910\u98f2"),
        ("\u8cfc\u7269", "\u8cfc\u7269"),
        ("\u8cb7\u8863\u670d", "\u8cfc\u7269"),
        ("貸款", "貸款"),
        ("信用卡", "信用卡"),
        ("幼稚園", "幼稚園"),
        ("管理費", "管理費"),
        ("房租", "房租"),
        ("午餐", "餐飲"),
        ("晚餐", "餐飲"),
        ("早餐", "餐飲"),
        ("點心", "餐飲"),
        ("消夜", "餐飲"),
        ("宵夜", "餐飲"),
        ("咖啡", "餐飲"),
        ("飲料", "餐飲"),
        ("餐費", "餐飲"),
        ("餐飲", "餐飲"),
        ("捷運", "交通"),
        ("公車", "交通"),
        ("計程車", "交通"),
        ("全聯", "生活用品"),
        ("家樂福", "生活用品"),
        ("水費", "水電瓦斯"),
        ("電費", "水電瓦斯"),
        ("瓦斯", "水電瓦斯"),
        ("保險", "保險"),
        ("信用卡", "信用卡"),
    ]
    for keyword, category in category_keywords:
        if keyword in text:
            return category
    return "其他"


def parse_expense_text_fallback(text: str) -> ExpenseEntry:
    amount_match = re.search(r"(\d+)", text)
    if not amount_match:
        raise ValueError("No amount found in text.")

    amount = int(amount_match.group(1))
    description = (text[: amount_match.start()] + text[amount_match.end() :]).strip()
    target_date = date.today()
    if "昨天" in text:
        target_date = target_date - timedelta(days=1)

    return ExpenseEntry(
        date=target_date.isoformat(),
        time=parse_time_fallback(text),
        amount=amount,
        currency="TWD",
        category=infer_category(text),
        merchant=None,
        note=description or None,
        confidence=0.90,
    )


def parse_time_fallback(text: str) -> str | None:
    hour_match = re.search(
        r"(上午|早上|中午|下午|晚上|凌晨)?\s*(\d{1,2}|[一二三四五六七八九十十一十二兩]+)\s*[點:：](?:\s*(\d{1,2})\s*分?)?",
        text,
    )
    if not hour_match:
        return None

    period = hour_match.group(1) or ""
    hour = parse_hour_text(hour_match.group(2))
    if hour is None:
        return None
    minute = int(hour_match.group(3) or 0)

    if period in {"下午", "晚上"} and hour < 12:
        hour += 12
    elif period == "中午" and hour < 12:
        hour += 12
    elif period == "凌晨" and hour == 12:
        hour = 0

    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def parse_hour_text(text: str) -> int | None:
    if text.isdigit():
        return int(text)

    chinese_hours = {
        "一": 1,
        "二": 2,
        "兩": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "十一": 11,
        "十二": 12,
    }
    return chinese_hours.get(text)


def parse_amount(text: str) -> int | None:
    cleaned = re.sub(r"20\d{2}\s*\u5e74\s*\d{1,2}\s*\u6708\s*\d{1,2}\s*(?:\u65e5|\u865f)?", " ", text)
    cleaned = re.sub(r"\d{1,2}\s*\u6708\s*\d{1,2}\s*(?:\u65e5|\u865f)?", " ", cleaned)
    cleaned = re.sub(r"\d{1,2}\s*\u6708(?!\s*(?:\u842c|\u5343|\u767e|\u5143|\u584a|k|K))", " ", cleaned)
    cleaned = re.sub(r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}", " ", cleaned)
    cleaned = re.sub(r"\d{1,2}\s*(?:/|月)\s*\d{1,2}\s*(?:日|號)?", " ", cleaned)
    cleaned = re.sub(r"\d{1,2}\s*(?:日|號)", " ", cleaned)

    chinese_match = re.search(r"([一二三四五六七八九十兩]+)\s*萬", cleaned)
    if chinese_match:
        value = parse_chinese_number(chinese_match.group(1))
        return value * 10000 if value is not None else None

    ten_thousand_match = re.search(r"(\d+(?:\.\d+)?)\s*\u842c", cleaned.replace(",", ""))
    if ten_thousand_match:
        return int(float(ten_thousand_match.group(1)) * 10000)

    match = re.search(r"(\d+)", cleaned.replace(",", ""))
    if match:
        return int(match.group(1))
    return None


def parse_payable_amount(text: str) -> int | None:
    cleaned = re.sub(r"20\d{2}\s*\u5e74\s*\d{1,2}\s*\u6708\s*\d{1,2}\s*(?:\u65e5|\u865f)?", " ", text)
    cleaned = re.sub(r"\d{1,2}\s*\u6708\s*\d{1,2}\s*(?:\u65e5|\u865f)?", " ", cleaned)
    cleaned = re.sub(r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}", " ", cleaned)
    cleaned = re.sub(r"\b\d{1,2}\s*/\s*\d{1,2}\b\s*(?:\u865f|\u65e5)?", " ", cleaned)
    cleaned = re.sub(r"\b\d{1,2}\s*(?:\u865f|\u65e5)\b", " ", cleaned)
    return parse_amount(cleaned)


def parse_chinese_number(text: str) -> int | None:
    if text in {"一", "二", "兩", "三", "四", "五", "六", "七", "八", "九"}:
        return parse_hour_text(text)
    if text == "十":
        return 10
    if text.startswith("十"):
        tail = parse_hour_text(text[1:])
        return 10 + (tail or 0)
    if text.endswith("十"):
        head = parse_hour_text(text[:-1])
        return (head or 0) * 10
    if "十" in text:
        head_text, tail_text = text.split("十", 1)
        head = parse_hour_text(head_text) or 1
        tail = parse_hour_text(tail_text) or 0
        return head * 10 + tail
    return None


def parse_due_date(text: str) -> str | None:
    today = date.today()

    iso_match = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", text)
    if iso_match:
        return date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))).isoformat()

    month_day_match = re.search(r"(\d{1,2})\s*(?:/|月)\s*(\d{1,2})\s*(?:日|號)?", text)
    if month_day_match:
        due = date(today.year, int(month_day_match.group(1)), int(month_day_match.group(2)))
        if due < today:
            due = date(today.year + 1, due.month, due.day)
        return due.isoformat()

    day_match = re.search(r"(\d{1,2})\s*(?:日|號)", text)
    if day_match:
        day = int(day_match.group(1))
        due = date(today.year, today.month, day)
        if due < today:
            next_month = today.month + 1
            next_year = today.year
            if next_month == 13:
                next_month = 1
                next_year += 1
            due = date(next_year, next_month, day)
        return due.isoformat()

    if "今天" in text:
        return today.isoformat()
    if "明天" in text:
        return (today + timedelta(days=1)).isoformat()
    if "後天" in text:
        return (today + timedelta(days=2)).isoformat()
    return None


def is_balance_query(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    keywords = (
        "\u6536\u652f",
        "\u6536\u5165\u652f\u51fa",
        "\u76c8\u9918",
        "\u7d50\u9918",
        "\u4e0d\u8db3",
        "\u5920\u4e0d\u5920",
        "\u5920\u55ce",
        "\u9084\u5dee",
        "\u900f\u652f",
        "\u8d85\u652f",
        "\u8d64\u5b57",
        "\u5269\u591a\u5c11",
        "\u9084\u6709\u591a\u5c11\u9322",
        "\u53ef\u52d5\u7528\u591a\u5c11",
    )
    return any(keyword in normalized for keyword in keywords)


def is_available_investment_cash_query(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    investment_terms = ("\u80a1\u7968", "\u8cb7\u80a1", "\u6295\u8cc7", "ETF", "etf", "\u57fa\u91d1")
    if not any(term in normalized for term in investment_terms):
        return False
    advice_terms = (
        "\u5efa\u8b70",
        "\u5efa\u8b70\u6295\u5165",
        "\u5efa\u8b70\u8cb7",
        "\u9810\u7b97\u6293\u591a\u5c11",
        "\u6bd4\u4f8b\u591a\u5c11",
        "\u6295\u5165\u6bd4\u4f8b",
        "\u914d\u7f6e\u6bd4\u4f8b",
    )
    return any(term in normalized for term in advice_terms)


def is_available_purchase_cash_query(text: str) -> bool:
    if is_available_investment_cash_query(text):
        return False
    normalized = re.sub(r"\s+", "", text)
    patterns = (
        r"(?:\u6709|\u9084\u6709|\u5269|\u9084\u5269)\u591a\u5c11(?:\u9322)?\u53ef\u4ee5\u8cb7.+",
        r"\u9084\u5269\u591a\u5c11\u53ef\u4ee5\u8cb7.+",
        r"\u9019\u500b\u6708\u53ef\u4ee5\u8cb7.+(?:\u55ce|\?)?",
        r"\u53ef\u4ee5\u8cb7.+(?:\u55ce|\?)?",
        r"\u53ef\u4ee5\u6295\u8cc7\u591a\u5c11",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def extract_purchase_purpose(text: str) -> str | None:
    normalized = re.sub(r"\s+", "", text)
    cleaned = re.sub(r"[?？。！!]", "", normalized)
    patterns = (
        r"(?:\u6709|\u9084\u6709|\u5269|\u9084\u5269)\u591a\u5c11(?:\u9322)?\u53ef\u4ee5\u8cb7(.+)",
        r"\u9019\u500b\u6708\u53ef\u4ee5\u8cb7(.+?)(?:\u55ce)?$",
        r"\u53ef\u4ee5\u8cb7(.+?)(?:\u55ce)?$",
        r"\u53ef\u4ee5\u6295\u8cc7(.+)?",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            purpose = match.group(1).strip()
            return purpose or None
    return None


def wants_balance_details(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    if any(keyword in normalized for keyword in ("\u5217\u51fa\u660e\u7d30", "\u6536\u652f\u660e\u7d30", "\u6536\u5165\u660e\u7d30", "\u652f\u51fa\u660e\u7d30")):
        return True
    if "\u6709\u54ea\u4e9b\u652f\u51fa" in normalized or "\u54ea\u4e9b\u652f\u51fa" in normalized:
        return True
    return "\u70ba\u4ec0\u9ebc" in normalized and "\u9019\u9ebc\u591a" in normalized


def is_duplicate_data_text(text: str) -> bool:
    return "\u91cd\u8907" in text


def is_delete_duplicate_text(text: str) -> bool:
    return is_duplicate_data_text(text) and any(keyword in text for keyword in ("\u522a\u9664", "\u6e05\u6389", "\u6e05\u9664", "\u79fb\u9664"))


def is_confirm_delete_duplicates_text(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return normalized in {"\u78ba\u8a8d\u522a\u9664\u91cd\u8907\u8cc7\u6599", "\u78ba\u8a8d\u522a\u9664\u91cd\u8907", "\u522a\u9664\u91cd\u8907\u8cc7\u6599"}


def resolve_income_owner(text: str) -> str | None:
    for owner in ("\u8001\u516c", "\u8001\u5a46", "\u5152\u5b50", "\u5973\u5152"):
        if owner in text:
            return owner
    return get_owner(text)


def resolve_income_item_name(text: str, income_type: str) -> str:
    if "\u85aa\u6c34" in text or "\u85aa\u8cc7" in text:
        return "\u85aa\u6c34"
    if "\u88dc\u52a9" in text:
        return "\u653f\u5e9c\u88dc\u52a9"
    if "\u696d\u5916" in text:
        return "\u696d\u5916\u6536\u5165"
    return income_type


def parse_month_number_from_text(text: str) -> int | None:
    digit_match = re.search(r"(\d{1,2})\s*\u6708", text)
    if digit_match:
        month = int(digit_match.group(1))
        return month if 1 <= month <= 12 else None
    chinese_months = {
        "\u4e00": 1,
        "\u4e8c": 2,
        "\u4e09": 3,
        "\u56db": 4,
        "\u4e94": 5,
        "\u516d": 6,
        "\u4e03": 7,
        "\u516b": 8,
        "\u4e5d": 9,
        "\u5341": 10,
        "\u5341\u4e00": 11,
        "\u5341\u4e8c": 12,
    }
    for label, month in sorted(chinese_months.items(), key=lambda item: len(item[0]), reverse=True):
        if f"{label}\u6708" in text:
            return month
    return None


def get_month_range(text: str) -> tuple[date, date]:
    today = date.today()
    month = parse_month_number_from_text(text)
    if month:
        start_date = date(today.year, month, 1)
    else:
        start_date = today.replace(day=1)

    if start_date.month == 12:
        end_date = date(start_date.year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(start_date.year, start_date.month + 1, 1) - timedelta(days=1)
    return start_date, end_date


def normalize_payable_item_type(text: str | None) -> str | None:
    if not text:
        return None
    if "\u4fe1\u8cb8" in text:
        return "\u4fe1\u8cb8"
    if "\u623f\u8cb8" in text:
        return "\u623f\u8cb8"
    if "\u4fe1\u7528\u5361" in text or "\u5361\u8cbb" in text:
        return "\u4fe1\u7528\u5361"
    if "\u623f\u79df" in text:
        return "\u623f\u79df"
    if "\u8cb8\u6b3e" in text:
        return "\u8cb8\u6b3e"
    if "\u4fdd\u6bcd\u8cbb" in text or "\u4fdd\u6bcd" in text:
        return "\u4fdd\u6bcd\u8cbb"
    if "\u5e7c\u7a1a\u5712" in text or "\u5e7c\u5152\u5712" in text:
        return "\u5e7c\u7a1a\u5712"
    if "\u7ba1\u7406\u8cbb" in text:
        return "\u7ba1\u7406\u8cbb"
    return None


def get_payable_type(text: str) -> str | None:
    normalized = normalize_payable_item_type(text)
    if normalized:
        return normalized
    aliases = (
        ("\u623f\u8cb8", ("\u623f\u8cb8", "\u623f\u5c4b\u8cb8\u6b3e")),
        ("\u623f\u79df", ("\u623f\u79df", "\u79df\u91d1")),
        ("\u4fe1\u7528\u5361", ("\u4fe1\u7528\u5361", "\u5361\u8cbb", "\u5237\u5361")),
        ("\u4fe1\u8cb8", ("\u4fe1\u8cb8", "\u4fe1\u7528\u8cb8\u6b3e")),
        ("\u8cb8\u6b3e", ("\u8cb8\u6b3e",)),
        ("\u4fdd\u6bcd\u8cbb", ("\u4fdd\u6bcd\u8cbb", "\u4fdd\u6bcd")),
        ("\u5e7c\u7a1a\u5712", ("\u5e7c\u7a1a\u5712", "\u5e7c\u5152\u5712", "\u5b78\u8cbb")),
        ("\u7ba1\u7406\u8cbb", ("\u7ba1\u7406\u8cbb",)),
    )
    for item_type, keywords in aliases:
        if any(keyword in text for keyword in keywords):
            return item_type
    keywords = (
        "信用卡",
        "房貸",
        "信貸",
        "貸款",
        "保母費",
        "保母",
        "幼稚園",
        "管理費",
    )
    for keyword in keywords:
        if keyword in text:
            return "保母費" if keyword == "保母" else keyword
    return None


def get_owner(text: str) -> str | None:
    for owner in ("老婆", "老公", "兒子", "女兒"):
        if owner in text:
            return owner
    return None


def get_bank(text: str) -> str | None:
    banks = (
        "玉山",
        "中國信託",
        "中信",
        "台新",
        "國泰",
        "富邦",
        "永豐",
        "兆豐",
        "第一",
        "華南",
        "元大",
        "聯邦",
        "星展",
        "匯豐",
        "台北富邦",
    )
    for bank in banks:
        if bank in text:
            return "中國信託" if bank == "中信" else bank
    return None


def backfill_payable_metadata() -> None:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, note FROM payables WHERE owner IS NULL OR bank IS NULL"
        ).fetchall()
        for row in rows:
            note = row["note"] or ""
            conn.execute(
                """
                UPDATE payables
                SET owner = COALESCE(owner, ?),
                    bank = COALESCE(bank, ?)
                WHERE id = ?
                """,
                (get_owner(note), get_bank(note), row["id"]),
            )
        conn.commit()


def backfill_utility_categories() -> None:
    rules = {
        "\u96fb\u8a71\u8cbb": ["\u96fb\u8a71\u8cbb", "\u624b\u6a5f\u8cbb", "\u96fb\u4fe1\u8cbb"],
        "\u7db2\u8def\u8cbb": ["\u7db2\u8def\u8cbb", "\u5bec\u983b"],
        "\u96fb\u8cbb": ["\u96fb\u8cbb", "\u53f0\u96fb", "\u96fb\u529b\u8cbb"],
        "\u6c34\u8cbb": ["\u6c34\u8cbb", "\u81ea\u4f86\u6c34"],
        "\u74e6\u65af\u8cbb": ["\u74e6\u65af\u8cbb", "\u5929\u7136\u6c23", "\u6876\u88dd\u74e6\u65af"],
    }
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, raw_text, note, merchant, category
            FROM expenses
            WHERE category IN (?, ?, ?)
            """,
            ("\u751f\u6d3b\u7528\u54c1", "\u5176\u4ed6", "瘞湧?行"),
        ).fetchall()
        for row in rows:
            text = " ".join(str(row[key] or "") for key in ("raw_text", "note", "merchant"))
            for category, keywords in rules.items():
                if any(keyword in text for keyword in keywords):
                    conn.execute("UPDATE expenses SET category = ? WHERE id = ?", (category, row["id"]))
                    break
        conn.commit()


def get_income_type(text: str) -> str | None:
    keywords = ("薪資收入", "薪水", "薪資", "業外收入", "政府補助", "補助")
    for keyword in keywords:
        if keyword in text:
            if keyword in {"薪水", "薪資"}:
                return "薪資收入"
            if keyword == "補助":
                return "政府補助"
            return keyword
    return None


def get_income_type(text: str) -> str | None:
    keywords = ("\u85aa\u8cc7\u6536\u5165", "\u85aa\u6c34", "\u85aa\u8cc7", "\u696d\u5916\u6536\u5165", "\u653f\u5e9c\u88dc\u52a9", "\u88dc\u52a9")
    for keyword in keywords:
        if keyword in text:
            if keyword in {"\u85aa\u6c34", "\u85aa\u8cc7"}:
                return "\u85aa\u8cc7\u6536\u5165"
            if keyword == "\u88dc\u52a9":
                return "\u653f\u5e9c\u88dc\u52a9"
            return keyword
    return None


def is_income_query(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    income_terms = ("\u6536\u5165", "\u85aa\u6c34", "\u85aa\u8cc7", "\u85aa\u8cc7\u6536\u5165", "\u696d\u5916\u6536\u5165", "\u653f\u5e9c\u88dc\u52a9")
    balance_terms = ("\u6536\u652f", "\u7d50\u9918", "\u76c8\u9918", "\u900f\u652f", "\u8d85\u652f", "\u5269\u591a\u5c11", "\u9084\u6709\u591a\u5c11\u9322", "\u53ef\u52d5\u7528", "\u5920\u4e0d\u5920", "\u4e0d\u8db3")
    if not any(term in normalized for term in income_terms):
        return False
    if any(term in normalized for term in balance_terms):
        return False
    return True


def is_income_list_query(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return is_income_query(text) and any(term in normalized for term in ("\u6e05\u55ae", "\u660e\u7d30", "\u5217\u51fa", "\u6709\u54ea\u4e9b"))


def is_paid_text(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    keywords = (
        "\u5df2\u7e73",
        "\u7e73\u4e86",
        "\u7e73\u6e05",
        "\u4ed8\u4e86",
        "\u4ed8\u6b3e\u4e86",
    )
    return any(keyword in normalized for keyword in keywords)


def resolve_payable_item_type(raw_text: str, route: ActionRoute | None = None) -> str | None:
    normalized = normalize_payable_item_type(raw_text)
    if normalized:
        return normalized
    if route and route.item_type:
        return normalize_payable_item_type(route.item_type) or route.item_type
    detected = get_payable_type(raw_text)
    if detected:
        return detected
    aliases = (
        ("\u4fe1\u7528\u5361", ("\u4fe1\u7528\u5361", "\u5361\u8cbb", "\u5237\u5361")),
        ("\u623f\u8cb8", ("\u623f\u8cb8", "\u623f\u5c4b\u8cb8\u6b3e")),
        ("\u623f\u79df", ("\u623f\u79df", "\u79df\u91d1")),
        ("\u4fe1\u8cb8", ("\u4fe1\u8cb8", "\u4fe1\u7528\u8cb8\u6b3e")),
        ("\u8cb8\u6b3e", ("\u8cb8\u6b3e",)),
        ("\u4fdd\u6bcd\u8cbb", ("\u4fdd\u6bcd\u8cbb", "\u4fdd\u6bcd")),
        ("\u5e7c\u7a1a\u5712", ("\u5e7c\u7a1a\u5712", "\u5e7c\u5152\u5712", "\u5b78\u8cbb")),
        ("\u7ba1\u7406\u8cbb", ("\u7ba1\u7406\u8cbb",)),
    )
    for item_type, keywords in aliases:
        if any(keyword in raw_text for keyword in keywords):
            return item_type
    return None


def payable_type_aliases() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        ("\u4fe1\u7528\u5361", ("\u4fe1\u7528\u5361", "\u5361\u8cbb", "\u5237\u5361")),
        ("\u623f\u8cb8", ("\u623f\u8cb8", "\u623f\u5c4b\u8cb8\u6b3e")),
        ("\u623f\u79df", ("\u623f\u79df", "\u79df\u91d1")),
        ("\u4fe1\u8cb8", ("\u4fe1\u8cb8", "\u4fe1\u7528\u8cb8\u6b3e")),
        ("\u8cb8\u6b3e", ("\u8cb8\u6b3e",)),
        ("\u4fdd\u6bcd\u8cbb", ("\u4fdd\u6bcd\u8cbb", "\u4fdd\u6bcd")),
        ("\u5e7c\u7a1a\u5712", ("\u5e7c\u7a1a\u5712", "\u5e7c\u5152\u5712", "\u5b78\u8cbb")),
        ("\u7ba1\u7406\u8cbb", ("\u7ba1\u7406\u8cbb",)),
    )


def resolve_payable_item_types(raw_text: str, route: ActionRoute | None = None) -> list[str]:
    candidates: list[str] = []

    route_item = route.item_type if route and route.item_type else None
    if route_item:
        for part in re.split(r"[,，、/]|(?:\s*(?:跟|和|及|與|and)\s*)", route_item):
            part = part.strip()
            if part:
                candidates.append(part)

    for item_type, keywords in payable_type_aliases():
        if any(keyword in raw_text for keyword in keywords):
            candidates.append(item_type)

    if not candidates:
        item_type = resolve_payable_item_type(raw_text, route)
        if item_type:
            candidates.append(item_type)

    result: list[str] = []
    for candidate in candidates:
        normalized = None
        for item_type, keywords in payable_type_aliases():
            if candidate == item_type or any(keyword in candidate for keyword in keywords):
                normalized = item_type
                break
        normalized = normalized or candidate
        if normalized not in result:
            result.append(normalized)
    return result


def format_month_day(value: str) -> str:
    parsed = datetime.strptime(value, "%Y-%m-%d").date()
    return f"{parsed.month}/{parsed.day}"


def current_month_range() -> tuple[date, date]:
    today = date.today()
    start_date = today.replace(day=1)
    if today.month == 12:
        end_date = date(today.year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(today.year, today.month + 1, 1) - timedelta(days=1)
    return start_date, end_date


def parse_payable_query_fallback(text: str, route: ActionRoute | None = None) -> PayableQuery:
    item_type = (route.item_type if route else None) or resolve_payable_item_type(text)
    normalized = re.sub(r"\s+", "", text)
    status: PayableStatus = "all" if item_type else "unpaid"
    if any(keyword in normalized for keyword in ("\u9084\u6c92\u7e73", "\u672a\u7e73", "\u6c92\u7e73", "\u5f85\u7e73")):
        status = "unpaid"
    if any(keyword in normalized for keyword in ("\u5df2\u7e73", "\u7e73\u4e86")) and not normalized.endswith("\u55ce?"):
        status = "paid"

    start_date, end_date = current_month_range()
    return PayableQuery(
        item_type=item_type,
        owner=(route.owner if route else None) or get_owner(text),
        bank=(route.bank if route else None) or get_bank(text),
        status=status,
        date_range_type="this_month",
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        confidence=0.75,
        reason="local payable query fallback",
    )


def parse_payable_query_with_openai(text: str) -> PayableQuery:
    start_date, end_date = current_month_range()
    schema = {
        "type": "object",
        "properties": {
            "item_type": {"type": ["string", "null"]},
            "owner": {"type": ["string", "null"]},
            "bank": {"type": ["string", "null"]},
            "status": {"type": "string", "enum": ["unpaid", "paid", "all"]},
            "date_range_type": {"type": "string", "enum": ["this_month", "custom", "unspecified"]},
            "start_date": {"type": ["string", "null"]},
            "end_date": {"type": ["string", "null"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": ["string", "null"]},
        },
        "required": [
            "item_type",
            "owner",
            "bank",
            "status",
            "date_range_type",
            "start_date",
            "end_date",
            "confidence",
            "reason",
        ],
        "additionalProperties": False,
    }
    log_usage("openai", "chat_completion", detail=get_current_model())
    response = get_openai_client().chat.completions.create(
        model=get_current_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "你是台灣 LINE 記帳 Bot 的待繳款查詢解析器，只輸出 JSON，不要回答使用者。"
                    f"本月範圍是 {start_date.isoformat()} 到 {end_date.isoformat()}。"
                    "房貸繳了嗎、這個月房貸繳了嗎，都輸出 item_type=房貸,status=all,date_range_type=this_month。"
                    "房租繳了嗎輸出 item_type=房租,status=all,date_range_type=this_month。房租和房貸是不同項目。"
                    "信用卡繳了嗎輸出 item_type=信用卡,status=all,date_range_type=this_month。"
                    "這個月還有哪些沒繳、目前未繳費用有哪些，輸出 item_type=null,status=unpaid,date_range_type=this_month。"
                    "問句不可判成 create_payable 或 mark_payable_paid。只有已繳 房貸、房貸繳了、我繳了房貸才是已繳動作，但本函式只輸出 query 欄位。"
                ),
            },
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "payable_query", "strict": True, "schema": schema},
        },
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI returned an empty payable query.")
    query = PayableQuery.model_validate_json(content)
    if query.date_range_type in {"this_month", "unspecified"} or not query.start_date or not query.end_date:
        query.start_date = start_date.isoformat()
        query.end_date = end_date.isoformat()
        query.date_range_type = "this_month"
    query.item_type = resolve_payable_item_type(query.item_type or text) or query.item_type
    return query


def parse_payable_query(text: str, route: ActionRoute | None = None) -> PayableQuery:
    try:
        query = parse_payable_query_with_openai(text)
        fallback = parse_payable_query_fallback(text, route)
        query.item_type = fallback.item_type or query.item_type
        query.owner = fallback.owner or query.owner
        query.bank = fallback.bank or query.bank
        if is_general_unpaid_payable_query(text):
            query.status = "unpaid"
        elif query.item_type and is_question_text(text):
            query.status = "all"
        if query.confidence >= 0.65:
            return query
    except Exception:
        logger.exception("OpenAI payable query parsing failed; using fallback for text: %s", text)
    return parse_payable_query_fallback(text, route)


def payable_label(row: sqlite3.Row) -> str:
    meta = " ".join(value for value in (row["owner"], row["bank"]) if value)
    return f"{meta} {row['item_type']}".strip()


def build_existing_payable_reply(row: sqlite3.Row, title: str) -> str:
    return (
        f"{title}\n"
        f"\u7de8\u865f\uff1a{row['id']}\n"
        f"\u9805\u76ee\uff1a{row['item_type']}\n"
        f"\u91d1\u984d\uff1aTWD {int(row['amount'])}\n"
        f"\u671f\u9650\uff1a{row['due_date']}"
    )


def dedupe_payables(line_user_id: str | None = None) -> int:
    where = ["status = 'unpaid'"]
    params: list[str | None] = []
    if line_user_id:
        where.append("line_user_id = ?")
        params.append(line_user_id)

    with get_db() as conn:
        rows = conn.execute(
            f"""
            SELECT id, line_user_id, item_type, amount, due_date, owner, bank
            FROM payables
            WHERE {' AND '.join(where)}
            ORDER BY id ASC
            """,
            params,
        ).fetchall()

        seen: dict[tuple[object, ...], int] = {}
        delete_ids: list[int] = []
        for row in rows:
            key = (
                row["line_user_id"],
                row["item_type"],
                int(row["amount"]),
                row["due_date"],
                row["owner"] or "",
                row["bank"] or "",
            )
            if key in seen:
                delete_ids.append(int(row["id"]))
            else:
                seen[key] = int(row["id"])

        for row_id in delete_ids:
            conn.execute("DELETE FROM payables WHERE id = ?", (row_id,))
        conn.commit()
    return len(delete_ids)


def create_payable(
    line_user_id: str,
    item_type: str,
    amount: int,
    due_date: str,
    note: str | None = None,
    message_id: str | None = None,
) -> str | dict[str, str | None]:
    item_type = normalize_payable_item_type(item_type) or item_type
    owner = get_owner(note or "")
    bank = get_bank(note or "")
    with get_db() as conn:
        if message_id:
            existing = conn.execute(
                """
                SELECT * FROM payables
                WHERE line_user_id = ? AND message_id = ?
                LIMIT 1
                """,
                (line_user_id, message_id),
            ).fetchone()
            if existing:
                return build_existing_payable_reply(existing, "\u9019\u7b46\u5f85\u7e73\u6b3e\u5df2\u7d93\u5efa\u7acb\u904e")

        existing = conn.execute(
            """
            SELECT * FROM payables
            WHERE line_user_id = ?
              AND item_type = ?
              AND amount = ?
              AND due_date = ?
              AND COALESCE(owner, '') = COALESCE(?, '')
              AND COALESCE(bank, '') = COALESCE(?, '')
              AND status = 'unpaid'
            ORDER BY id ASC
            LIMIT 1
            """,
            (line_user_id, item_type, amount, due_date, owner, bank),
        ).fetchone()
        if existing:
            return build_existing_payable_reply(existing, "\u9019\u7b46\u5f85\u7e73\u6b3e\u5df2\u7d93\u5b58\u5728")

        conn.execute(
            """
            INSERT INTO payables (
                line_user_id, item_type, amount, currency, due_date, owner, bank, note, message_id, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'unpaid', ?)
            """,
            (
                line_user_id,
                item_type,
                amount,
                "TWD",
                due_date,
                owner,
                bank,
                note,
                message_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    due = datetime.strptime(due_date, "%Y-%m-%d").date()
    reminder_dates = [
        (due - timedelta(days=3)).isoformat(),
        (due - timedelta(days=2)).isoformat(),
        (due - timedelta(days=1)).isoformat(),
    ]
    reminder_separator = "\u3001"
    owner_line = f"\u4eba\u54e1\uff1a{owner}\n" if owner else ""
    bank_line = f"\u9280\u884c\uff1a{bank}\n" if bank else ""
    return (
        "\u5df2\u5efa\u7acb\u5f85\u7e73\u63d0\u9192\n"
        f"\u9805\u76ee\uff1a{item_type}\n"
        f"{owner_line}"
        f"{bank_line}"
        f"\u91d1\u984d\uff1aTWD {amount}\n"
        f"\u671f\u9650\uff1a{due_date}\n"
        f"\u63d0\u9192\u65e5\u671f\uff1a{reminder_separator.join(reminder_dates)}\n"
        "\u63d0\u9192\u65b9\u5f0f\uff1aLINE push"
    )

def get_payable_draft(line_user_id: str) -> sqlite3.Row | None:
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM payable_drafts WHERE line_user_id = ?",
            (line_user_id,),
        ).fetchone()


def save_payable_draft(
    line_user_id: str,
    item_type: str,
    amount: int,
    note: str,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO payable_drafts (
                line_user_id, item_type, amount, currency, owner, bank, note, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(line_user_id) DO UPDATE SET
                item_type = excluded.item_type,
                amount = excluded.amount,
                currency = excluded.currency,
                owner = excluded.owner,
                bank = excluded.bank,
                note = excluded.note,
                created_at = excluded.created_at
            """,
            (
                line_user_id,
                item_type,
                amount,
                "TWD",
                get_owner(note),
                get_bank(note),
                note,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def clear_payable_draft(line_user_id: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM payable_drafts WHERE line_user_id = ?", (line_user_id,))
        conn.commit()


def save_income(raw_text: str, line_user_id: str | None) -> str | None:
    income_type = get_income_type(raw_text)
    amount = parse_amount(raw_text)
    if not income_type or amount is None:
        return None

    income_date = date.today()
    if "??訾?" in raw_text:
        income_date -= timedelta(days=1)
    owner = resolve_income_owner(raw_text)
    item_name = resolve_income_item_name(raw_text, income_type)
    category = income_type
    start_date = income_date.replace(day=1)
    if income_date.month == 12:
        end_date = date(income_date.year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(income_date.year, income_date.month + 1, 1) - timedelta(days=1)

    with get_db() as conn:
        existing = conn.execute(
            """
            SELECT id, amount
            FROM incomes
            WHERE income_date = ?
              AND income_date BETWEEN ? AND ?
              AND COALESCE(item_name, income_type) = ?
              AND amount = ?
              AND COALESCE(owner, '') = COALESCE(?, '')
              AND COALESCE(category, income_type) = ?
              AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
            ORDER BY id ASC
            LIMIT 1
            """,
            (
                income_date.isoformat(),
                start_date.isoformat(),
                end_date.isoformat(),
                item_name,
                amount,
                owner,
                category,
                line_user_id,
                line_user_id,
            ),
        ).fetchone()
        if existing:
            return (
                "\u9019\u7b46\u6536\u5165\u5df2\u7d93\u8a18\u9304\u904e\u4e86\n"
                f"\u65e5\u671f\uff1a{income_date.isoformat()}\n"
                f"\u985e\u578b\uff1a{income_type}\n"
                f"\u91d1\u984d\uff1aTWD {amount}"
            )

        conn.execute(
            """
            INSERT INTO incomes (
                line_user_id, raw_text, income_date, amount, currency, income_type, item_name, owner, category, note, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                line_user_id,
                raw_text,
                income_date.isoformat(),
                amount,
                "TWD",
                income_type,
                item_name,
                owner,
                category,
                raw_text,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    return (
        "\u5df2\u8a18\u9304\u6536\u5165\n"
        f"\u65e5\u671f\uff1a{income_date.isoformat()}\n"
        f"\u985e\u578b\uff1a{income_type}\n"
        f"\u91d1\u984d\uff1aTWD {amount}"
    )


def mark_payable_paid(raw_text: str, line_user_id: str | None) -> str | None:
    if not line_user_id or not is_paid_text(raw_text):
        return None

    dedupe_payables(line_user_id)
    item_type = normalize_payable_item_type(raw_text) or resolve_payable_item_type(raw_text)
    if not item_type:
        return "\u6211\u4e0d\u592a\u78ba\u5b9a\u4f60\u8981\u6a19\u8a18\u54ea\u4e00\u500b\u9805\u76ee\u5df2\u7e73\uff0c\u8acb\u56de\u8986\u4f8b\u5982\uff1a\u5df2\u7e73 \u623f\u8cb8\u3002"

    owner = get_owner(raw_text)
    bank = get_bank(raw_text)
    today = date.today()
    start_date = today.replace(day=1)
    if today.month == 12:
        end_date = date(today.year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(today.year, today.month + 1, 1) - timedelta(days=1)

    with get_db() as conn:
        where = [
            "line_user_id = ?",
            "item_type = ?",
            "status = 'unpaid'",
            "due_date BETWEEN ? AND ?",
        ]
        params: list[str | int | None] = [line_user_id, item_type, start_date.isoformat(), end_date.isoformat()]
        if owner:
            where.append("owner = ?")
            params.append(owner)
        if bank:
            where.append("bank = ?")
            params.append(bank)
        where_sql = " AND ".join(where)

        row = conn.execute(
            f"""
            SELECT * FROM payables
            WHERE {where_sql}
            ORDER BY ABS(julianday(due_date) - julianday(?)) ASC, due_date ASC, id ASC
            LIMIT 1
            """,
            [*params, today.isoformat()],
        ).fetchone()

        if not row:
            existing_paid = conn.execute(
                """
                SELECT id FROM payables
                WHERE line_user_id = ?
                  AND item_type = ?
                  AND status = 'paid'
                  AND due_date BETWEEN ? AND ?
                LIMIT 1
                """,
                (line_user_id, item_type, start_date.isoformat(), end_date.isoformat()),
            ).fetchone()
            if existing_paid:
                return f"\u9019\u500b\u6708{item_type}\u5df2\u7d93\u6a19\u8a18\u70ba\u5df2\u7e73\u3002"
            return f"\u6211\u627e\u4e0d\u5230\u9019\u500b\u6708\u7684{item_type}\u5f85\u7e73\u9805\u76ee\u3002"

        duplicate_rows = conn.execute(
            """
            SELECT id FROM payables
            WHERE line_user_id = ?
              AND item_type = ?
              AND amount = ?
              AND due_date = ?
              AND COALESCE(owner, '') = COALESCE(?, '')
              AND COALESCE(bank, '') = COALESCE(?, '')
              AND status = 'unpaid'
            """,
            (line_user_id, item_type, row["amount"], row["due_date"], row["owner"], row["bank"]),
        ).fetchall()
        update_ids = [int(duplicate["id"]) for duplicate in duplicate_rows] or [int(row["id"])]
        placeholders = ",".join("?" for _ in update_ids)
        conn.execute(
            f"UPDATE payables SET status = 'paid', paid_at = ? WHERE id IN ({placeholders})",
            [datetime.now(timezone.utc).isoformat(), *update_ids],
        )
        conn.commit()

    return (
        "\u5df2\u6a19\u8a18\u70ba\u5df2\u7e73\n"
        f"\u9805\u76ee\uff1a{item_type}\n"
        f"\u91d1\u984d\uff1aTWD {int(row['amount'])}\n"
        f"\u671f\u9650\uff1a{row['due_date']}"
    )


def format_optional(label: str, value: str | None) -> str:
    return f"{label}\uff1a{value}\n" if value else ""

def is_payable_query(text: str) -> bool:
    return (
        get_payable_type(text) is not None
        and any(keyword in text for keyword in ("還沒繳", "未繳", "待繳", "繳費", "到期", "期限"))
    )


def build_payable_list_reply(raw_text: str, line_user_id: str | None) -> str | None:
    if not line_user_id or not is_payable_query(raw_text):
        return None

    item_type = get_payable_type(raw_text)
    owner = get_owner(raw_text)
    bank = get_bank(raw_text)
    today = date.today()
    start_date = today.replace(day=1)
    if today.month == 12:
        end_date = date(today.year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(today.year, today.month + 1, 1) - timedelta(days=1)

    where = [
        "line_user_id = ?",
        "status = 'unpaid'",
        "due_date BETWEEN ? AND ?",
    ]
    params: list[str | int | None] = [line_user_id, start_date.isoformat(), end_date.isoformat()]
    if item_type:
        where.append("item_type = ?")
        params.append(item_type)
    if owner:
        where.append("owner = ?")
        params.append(owner)
    if bank:
        where.append("bank = ?")
        params.append(bank)

    with get_db() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM payables
            WHERE {where_sql}
            ORDER BY due_date ASC, id ASC
            """,
            params,
        ).fetchall()

    title = f"本月未繳項目：{item_type or '全部'}"
    if not rows:
        return f"{title}\n目前查無未繳資料。"

    total = sum(int(row["amount"]) for row in rows)
    lines = [title]
    for row in rows:
        meta = " ".join(value for value in (row["owner"], row["bank"]) if value)
        meta_text = f" {meta}" if meta else ""
        lines.append(f"{row['due_date']} {row['item_type']}{meta_text} TWD {row['amount']}")
    lines.append(f"合計：TWD {total}")
    lines.append("若已繳，請回覆：已繳 項目/銀行/人員")
    return "\n".join(lines)


def build_one_payable_query_reply(
    line_user_id: str,
    item_type: str,
    owner: str | None,
    bank: str | None,
    start_date: date,
    end_date: date,
) -> str:
    where = ["line_user_id = ?", "item_type = ?", "due_date BETWEEN ? AND ?"]
    params: list[str | int | None] = [line_user_id, item_type, start_date.isoformat(), end_date.isoformat()]
    if owner:
        where.append("owner = ?")
        params.append(owner)
    if bank:
        where.append("bank = ?")
        params.append(bank)

    with get_db() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM payables
            WHERE {where_sql}
            ORDER BY due_date ASC, id ASC
            """,
            params,
        ).fetchall()

    title_parts = [part for part in (owner, bank, item_type) if part]
    title_item = " ".join(title_parts) if title_parts else item_type
    if not rows:
        return f"\u9019\u500b\u6708{title_item}\u6c92\u6709\u7e73\u8cbb\u7d00\u9304\u3002"

    unpaid_rows = [row for row in rows if row["status"] == "unpaid"]
    duplicate_keys: dict[tuple[str, str, int], set[str]] = {}
    for row in rows:
        key = (str(row["due_date"]), str(row["item_type"]), int(row["amount"]))
        duplicate_keys.setdefault(key, set()).add(str(row["status"]))

    duplicate_lines = []
    for due_date, duplicated_item, amount in sorted(duplicate_keys):
        statuses = duplicate_keys[(due_date, duplicated_item, amount)]
        if "paid" in statuses and "unpaid" in statuses:
            duplicate_lines.append(f"\u53e6\u5916\u6211\u6709\u770b\u5230\u53ef\u80fd\u91cd\u8907\u7684\u8cc7\u6599\uff1a{format_month_day(due_date)} {duplicated_item} {amount} \u5143\uff0c\u540c\u6642\u6709\u5df2\u7e73\u548c\u672a\u7e73\u3002")

    if not unpaid_rows:
        reply = f"\u9019\u500b\u6708{title_item}\u5df2\u7d93\u7e73\u4e86\uff0c\u76ee\u524d\u6c92\u6709\u672a\u7e73\u9805\u76ee\u3002"
        if duplicate_lines:
            reply += "\n" + "\n".join(duplicate_lines)
        return reply

    if len(unpaid_rows) == 1:
        row = unpaid_rows[0]
        reply = f"\u9019\u500b\u6708{title_item}\u9084\u6709\u4e00\u7b46\u6c92\u7e73\uff0c\u671f\u9650\u662f {format_month_day(row['due_date'])}\uff0c\u91d1\u984d\u662f {int(row['amount'])} \u5143\u3002"
    else:
        lines = [f"\u9019\u500b\u6708{title_item}\u9084\u6709 {len(unpaid_rows)} \u7b46\u6c92\u7e73\uff1a"]
        for row in unpaid_rows:
            lines.append(f"{format_month_day(row['due_date'])} \u5230\u671f\uff0c{int(row['amount'])} \u5143")
        reply = "\n".join(lines)

    if duplicate_lines:
        reply += "\n" + "\n".join(duplicate_lines)
    return reply


def build_payable_query_reply(
    raw_text: str,
    line_user_id: str | None,
    route: ActionRoute | None = None,
) -> str:
    if not line_user_id:
        return "\u6211\u9700\u8981 LINE \u4f7f\u7528\u8005\u8cc7\u8a0a\u624d\u80fd\u5e6b\u4f60\u67e5\u7e73\u8cbb\u9805\u76ee\u3002"

    try:
        dedupe_payables(line_user_id)
        query = parse_payable_query(raw_text, route)
        query.item_type = normalize_payable_item_type(query.item_type) or query.item_type
        start_date, end_date = current_month_range()
        start = query.start_date or start_date.isoformat()
        end = query.end_date or end_date.isoformat()

        where = ["line_user_id = ?", "due_date BETWEEN ? AND ?"]
        params: list[str | int | None] = [line_user_id, start, end]
        if query.item_type:
            where.append("item_type = ?")
            params.append(query.item_type)
        if query.owner:
            where.append("owner = ?")
            params.append(query.owner)
        if query.bank:
            where.append("bank = ?")
            params.append(query.bank)
        if query.status in {"unpaid", "paid"}:
            where.append("status = ?")
            params.append(query.status)
        where_sql = " AND ".join(where)

        with get_db() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM payables
                WHERE {where_sql}
                ORDER BY due_date ASC, id ASC
                """,
                params,
            ).fetchall()

        item_label = query.item_type or "\u5f85\u7e73\u9805\u76ee"
        if not rows:
            if query.item_type:
                return f"\u9019\u500b\u6708\u6c92\u6709{item_label}\u7e73\u8cbb\u7d00\u9304\u3002"
            return "\u9019\u500b\u6708\u76ee\u524d\u6c92\u6709\u672a\u7e73\u9805\u76ee\u3002"

        unpaid_rows = [row for row in rows if row["status"] == "unpaid"]
        paid_rows = [row for row in rows if row["status"] == "paid"]

        if query.status == "all" and query.item_type:
            if unpaid_rows:
                if len(unpaid_rows) == 1:
                    row = unpaid_rows[0]
                    return f"\u9019\u500b\u6708{item_label}\u9084\u6c92\u7e73\uff0c\u671f\u9650\u662f {format_month_day(row['due_date'])}\uff0c\u91d1\u984d {int(row['amount'])} \u5143\u3002"
                lines = [f"\u9019\u500b\u6708{item_label}\u9084\u6709 {len(unpaid_rows)} \u7b46\u6c92\u7e73\uff1a"]
                for row in unpaid_rows:
                    lines.append(f"{format_month_day(row['due_date'])} \u5230\u671f\uff0c{int(row['amount'])} \u5143")
                return "\n".join(lines)
            if paid_rows:
                return f"\u9019\u500b\u6708{item_label}\u5df2\u7d93\u7e73\u4e86\u3002"
            return f"\u9019\u500b\u6708\u6c92\u6709{item_label}\u7e73\u8cbb\u7d00\u9304\u3002"

        if query.status == "paid":
            if query.item_type:
                return f"\u9019\u500b\u6708{item_label}\u5df2\u7d93\u7e73\u4e86\u3002"
            return f"\u9019\u500b\u6708\u5df2\u7e73\u9805\u76ee\u5171 {len(rows)} \u7b46\u3002"

        total = sum(int(row["amount"]) for row in unpaid_rows or rows)
        lines = [f"\u9019\u500b\u6708\u9084\u6c92\u7e73\u7684\u9805\u76ee\uff1a"]
        for row in unpaid_rows or rows:
            meta = " ".join(value for value in (row["owner"], row["bank"]) if value)
            meta_text = f" {meta}" if meta else ""
            lines.append(f"{format_month_day(row['due_date'])} {row['item_type']}{meta_text} {int(row['amount'])} \u5143")
        lines.append(f"\u5408\u8a08\uff1aTWD {total}")
        return "\n".join(lines)
    except Exception as exc:
        logger.exception("Payable query failed for text: %s", raw_text)
        return f"\u67e5\u8a62\u5f85\u7e73\u6b3e\u5931\u6557\uff1a{str(exc)[:80]}"


def handle_payable_setup(raw_text: str, line_user_id: str | None, message_id: str | None = None) -> str | None:
    if not line_user_id:
        return None

    pending = get_payable_draft(line_user_id)
    due_date = parse_due_date(raw_text)
    if pending and due_date:
        clear_payable_draft(line_user_id)
        return create_payable(
            line_user_id,
            str(pending["item_type"]),
            int(pending["amount"]),
            due_date,
            str(pending["note"] or ""),
            message_id,
        )

    item_type = get_payable_type(raw_text)
    amount = parse_amount(raw_text)
    if not item_type or amount is None:
        return None

    due_date = parse_due_date(raw_text)
    if due_date:
        return create_payable(line_user_id, item_type, amount, due_date, raw_text, message_id)

    save_payable_draft(line_user_id, item_type, amount, raw_text)
    return (
        f"{item_type} TWD {amount} 的繳費期限是哪一天？\n"
        "可以回覆例如：7/15、7月15日、15號、明天、後天。"
    )


async def push_line_message(line_user_id: str, text: str) -> None:
    await push_line_messages(line_user_id, [{"type": "text", "text": text}])


async def push_line_messages(line_user_id: str, messages: list[dict]) -> None:
    start = datetime.now(timezone.utc)
    if get_app_env() == "test":
        logger.info("APP_ENV=test; skip LINE push to %s messages=%s", line_user_id, messages)
        log_usage("line", "push", success=True, latency_ms=0)
        if any(message.get("type") == "image" for message in messages):
            log_usage("line", "image", success=True, latency_ms=0)
        return
    load_dotenv(override=True)
    channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
    if not channel_access_token:
        raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN is not configured.")

    headers = {
        "Authorization": f"Bearer {channel_access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "to": line_user_id,
        "messages": messages,
    }
    try:
        async with httpx.AsyncClient(timeout=10, verify=get_ssl_verify()) as client:
            response = await client.post(LINE_PUSH_URL, headers=headers, json=payload)
            response.raise_for_status()
        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        log_usage("line", "push", success=True, latency_ms=latency_ms)
        if any(message.get("type") == "image" for message in messages):
            log_usage("line", "image", success=True, latency_ms=latency_ms)
    except Exception:
        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        log_usage("line", "push", success=False, latency_ms=latency_ms)
        raise


async def reminder_loop() -> None:
    while True:
        try:
            await send_due_reminders()
        except Exception:
            logger.exception("Failed to send due reminders.")
        await asyncio.sleep(get_reminder_interval_hours() * 3600)


async def send_due_reminders() -> None:
    dedupe_payables()
    today = date.today()
    targets = {
        (today + timedelta(days=3)).isoformat(): 3,
        (today + timedelta(days=2)).isoformat(): 2,
        (today + timedelta(days=1)).isoformat(): 1,
    }
    logger.info("send_due_reminders today=%s target_dates=%s", today.isoformat(), list(targets.keys()))
    rows = []
    with get_db() as conn:
        for target_date, days_before in targets.items():
            rows.extend(
                conn.execute(
                    """
                    SELECT p.*
                    FROM payables p
                    LEFT JOIN payable_reminders r
                      ON r.payable_id = p.id
                     AND r.remind_days_before = ?
                     AND r.reminded_on = ?
                    WHERE p.status = 'unpaid'
                      AND p.due_date = ?
                      AND r.id IS NULL
                    """,
                    (days_before, today.isoformat(), target_date),
                ).fetchall()
            )
    logger.info("send_due_reminders found rows count=%d", len(rows))

    for row in rows:
        days_before = targets[row["due_date"]]
        reminded_on = today.isoformat()
        logger.info(
            "send_due_reminders payable id=%s line_user_id=%s due_date=%s days_before=%s",
            row["id"],
            row["line_user_id"],
            row["due_date"],
            days_before,
        )
        with get_db() as conn:
            existing = conn.execute(
                """
                SELECT id FROM payable_reminders
                WHERE payable_id = ? AND remind_days_before = ? AND reminded_on = ?
                """,
                (row["id"], days_before, reminded_on),
            ).fetchone()
            if existing:
                continue

        owner_text = format_optional("\u4eba\u54e1", row["owner"])
        bank_text = format_optional("\u9280\u884c", row["bank"])
        message = (
            "\u7e73\u8cbb\u63d0\u9192\n"
            f"\u9805\u76ee\uff1a{row['item_type']}\n"
            f"{owner_text}"
            f"{bank_text}"
            f"\u91d1\u984d\uff1aTWD {row['amount']}\n"
            f"\u671f\u9650\uff1a{row['due_date']}\n"
            f"\u9084\u6709 {days_before} \u5929\u5230\u671f\u3002\n"
            f"\u82e5\u5df2\u7e73\uff0c\u8acb\u56de\u8986\uff1a\u5df2\u7e73 {row['item_type']}"
        )
        try:
            await push_line_message(row["line_user_id"], message)
            logger.info("send_due_reminders push success payable_id=%s", row["id"])
        except Exception:
            logger.exception("send_due_reminders push failed payable_id=%s", row["id"])
            continue

        with get_db() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO payable_reminders (payable_id, remind_days_before, reminded_on)
                    VALUES (?, ?, ?)
                    """,
                    (row["id"], days_before, reminded_on),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                continue

def is_delete_intent(text: str) -> bool:
    return any(keyword in text.lower() for keyword in ("刪除", "删除", "取消", "移除", "delete"))


def is_delete_all_intent(text: str) -> bool:
    return is_delete_intent(text) and any(keyword in text for keyword in ("所有", "全部", "全刪", "清空"))


def is_confirm_delete_all_intent(text: str) -> bool:
    return any(keyword in text for keyword in ("確認", "確定")) and is_delete_all_intent(text)


def is_confirm_delete_intent(text: str) -> bool:
    return any(keyword in text for keyword in ("確認刪除", "確定刪除"))


def is_update_intent(text: str) -> bool:
    return any(keyword in text.lower() for keyword in ("修改", "改成", "更正", "更新", "edit", "update"))


def is_summary_intent(text: str) -> bool:
    return any(
        keyword in text
        for keyword in ("總共", "花費多少", "多少錢", "花多少", "總花費", "合計", "統計", "最貴", "最高")
    )


def is_list_intent(text: str) -> bool:
    return any(keyword in text for keyword in ("有哪一些", "有哪些", "列出", "明細", "清單"))


def get_query_dates(text: str) -> list[str]:
    today = date.today()
    dates: list[date] = []

    if any(keyword in text for keyword in ("這兩天", "近兩天", "昨天跟今天", "昨天和今天")):
        dates.extend([today - timedelta(days=1), today])
    else:
        if "昨天" in text:
            dates.append(today - timedelta(days=1))
        if "今天" in text or not dates:
            dates.append(today)

    unique_dates: list[date] = []
    for target_date in dates:
        if target_date not in unique_dates:
            unique_dates.append(target_date)

    return [target_date.isoformat() for target_date in unique_dates]


def get_query_dates_for_list(text: str) -> list[str]:
    if any(keyword in text for keyword in ("有哪一些", "有哪些")) and not any(
        keyword in text for keyword in ("今天", "昨天", "這兩天", "近兩天")
    ):
        today = date.today()
        return [(today - timedelta(days=1)).isoformat(), today.isoformat()]

    return get_query_dates(text)


def get_query_category(text: str, route_category: str | None = None) -> Category | None:
    if route_category:
        category = infer_category(route_category)
        if category != "其他":
            return category
        return route_category  # type: ignore[return-value]
    category = infer_category(text)
    return None if category == "其他" else category


def normalize_expense_category(value: str | None) -> Category | None:
    if not value:
        return None
    normalized = re.sub(r"\s+", "", value)
    aliases: tuple[tuple[Category, tuple[str, ...]], ...] = (
        ("\u9910\u98f2", ("\u9910\u98f2", "\u98f2\u98df", "\u5403\u98ef", "\u9910\u8cbb", "\u65e9\u9910", "\u5348\u9910", "\u665a\u9910", "\u5bb5\u591c", "\u98f2\u6599", "\u5496\u5561", "\u9ede\u5fc3")),
        ("\u4ea4\u901a", ("\u4ea4\u901a", "\u52a0\u6cb9", "\u505c\u8eca", "\u6377\u904b", "\u516c\u8eca", "\u8a08\u7a0b\u8eca", "Uber", "\u9ad8\u9435", "\u53f0\u9435", "\u706b\u8eca")),
        ("\u8cfc\u7269", ("\u8cfc\u7269", "\u8cb7\u8863\u670d", "\u8863\u670d", "\u6cbb\u88dd", "\u670d\u98fe", "\u978b\u5b50", "\u8932\u5b50", "\u5916\u5957")),
        ("\u751f\u6d3b\u7528\u54c1", ("\u751f\u6d3b\u7528\u54c1", "\u5168\u806f", "\u5bb6\u6a02\u798f", "\u8d85\u5e02", "\u65e5\u7528\u54c1", "\u885b\u751f\u7d19")),
        ("\u6c34\u8cbb", ("\u6c34\u8cbb", "\u81ea\u4f86\u6c34")),
        ("\u96fb\u8cbb", ("\u96fb\u8cbb", "\u53f0\u96fb", "\u96fb\u529b\u8cbb")),
        ("\u74e6\u65af\u8cbb", ("\u74e6\u65af\u8cbb", "\u5929\u7136\u6c23", "\u6876\u88dd\u74e6\u65af")),
        ("\u96fb\u8a71\u8cbb", ("\u96fb\u8a71\u8cbb", "\u624b\u6a5f\u8cbb", "\u96fb\u4fe1\u8cbb")),
        ("\u7db2\u8def\u8cbb", ("\u7db2\u8def\u8cbb", "\u5bec\u983b")),
        ("\u8cb8\u6b3e", ("\u8cb8\u6b3e", "\u623f\u8cb8", "\u4fe1\u8cb8")),
        ("\u4fe1\u7528\u5361", ("\u4fe1\u7528\u5361", "\u5361\u8cbb")),
        ("\u4fdd\u6bcd\u8cbb", ("\u4fdd\u6bcd\u8cbb", "\u4fdd\u6bcd")),
    )
    for category, keywords in aliases:
        if any(keyword in normalized for keyword in keywords):
            return category
    category = infer_category(value)
    return None if category == "其他" else category


def normalize_category_list(values: list[Category] | list[str]) -> list[Category]:
    result: list[Category] = []
    for value in values:
        category = normalize_expense_category(str(value))
        if category and category not in result:
            result.append(category)
    return result


def extract_exclude_categories(text: str) -> list[Category]:
    normalized = re.sub(r"\s+", "", text)
    result: list[Category] = []
    patterns = (
        r"(?:\u9664\u4e86|\u6263\u6389|\u4e0d\u542b|\u6392\u9664)([\u4e00-\u9fffA-Za-z]+?)(?:\u5916|\u7684|\u82b1|\u7e3d|\u591a|\uff0c|,|\?|？|$)",
    )
    for pattern in patterns:
        for match in re.findall(pattern, normalized):
            category = normalize_expense_category(match)
            if category and category not in result:
                result.append(category)
    return result


def extract_expense_exclude_keywords(text: str) -> list[str]:
    normalized = re.sub(r"\s+", "", text)
    result: list[str] = []
    keyword_items = (
        "\u623f\u8cb8",
        "\u4fe1\u8cb8",
    )
    exclude_prefixes = ("\u4e0d\u8981\u8a08\u7b97", "\u4e0d\u7b97", "\u9664\u4e86", "\u6263\u6389", "\u4e0d\u542b", "\u6392\u9664")
    for keyword in keyword_items:
        if any(f"{prefix}{keyword}" in normalized for prefix in exclude_prefixes):
            result.append(keyword)

    if any(f"{prefix}\u8cb8\u6b3e" in normalized for prefix in exclude_prefixes):
        result.append("\u8cb8\u6b3e")

    deduped: list[str] = []
    for keyword in result:
        if keyword not in deduped:
            deduped.append(keyword)
    return deduped


def should_use_filtered_ratio_denominator(text: str, query: ExpenseQuery) -> bool:
    normalized = re.sub(r"\s+", "", text)
    explicit_exclusion = any(keyword in normalized for keyword in ("\u4e0d\u8981\u8a08\u7b97", "\u4e0d\u7b97", "\u9664\u4e86", "\u6263\u6389", "\u4e0d\u542b", "\u6392\u9664"))
    return query.aggregation == "category_breakdown" and explicit_exclusion and bool(query.exclude_categories or query.exclude_keywords)


def is_ratio_query(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return any(keyword in normalized for keyword in ("\u4f54\u6bd4", "\u5360\u6bd4", "\u6bd4\u4f8b", "\u5360\u5168\u90e8\u591a\u5c11"))


def is_category_breakdown_query(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return any(
        keyword in normalized
        for keyword in (
            "\u5404\u985e\u5225\u82b1\u8cbb\u4f54\u6bd4",
            "\u5404\u5206\u985e\u82b1\u8cbb\u4f54\u6bd4",
            "\u5206\u985e\u4f54\u6bd4",
            "\u6bcf\u500b\u985e\u5225\u82b1\u591a\u5c11",
            "\u5404\u985e\u5225\u652f\u51fa",
            "\u5404\u5206\u985e\u652f\u51fa",
            "\u5404\u985e\u5225\u7684\u82b1\u8cbb\u4f54\u6bd4",
            "\u6240\u6709\u82b1\u8cbb\u6bd4\u4f8b",
            "\u5168\u90e8\u82b1\u8cbb\u6bd4\u4f8b",
            "\u652f\u51fa\u6bd4\u4f8b",
            "\u82b1\u8cbb\u6bd4\u4f8b",
        )
    )


def month_date_range(year: int, month: int) -> DateRange:
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    return DateRange(start_date=start.isoformat(), end_date=end.isoformat())


def extract_query_months(raw_text: str) -> list[int]:
    normalized = re.sub(r"\s+", "", raw_text)
    months = [int(value) for value in re.findall(r"(\d{1,2})\s*\u6708", normalized)]
    chinese_months = {
        "\u4e00": 1,
        "\u4e8c": 2,
        "\u5169": 2,
        "\u4e09": 3,
        "\u56db": 4,
        "\u4e94": 5,
        "\u516d": 6,
        "\u4e03": 7,
        "\u516b": 8,
        "\u4e5d": 9,
        "\u5341": 10,
        "\u5341\u4e00": 11,
        "\u5341\u4e8c": 12,
    }
    for match in re.findall(r"(\u5341\u4e00|\u5341\u4e8c|\u5341|[\u4e00\u4e8c\u5169\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d])\s*\u6708", normalized):
        month = chinese_months.get(match)
        if month:
            months.append(month)

    result: list[int] = []
    for month in months:
        if 1 <= month <= 12 and month not in result:
            result.append(month)
    return result


def extract_expense_specific_day(raw_text: str) -> DateRange | None:
    normalized = re.sub(r"\s+", "", raw_text)
    match = re.search(r"(\d{1,2})\u6708(\d{1,2})(?:\u865f|\u65e5)?", normalized)
    if not match:
        chinese_months = {
            "\u4e00": 1,
            "\u4e8c": 2,
            "\u5169": 2,
            "\u4e09": 3,
            "\u56db": 4,
            "\u4e94": 5,
            "\u516d": 6,
            "\u4e03": 7,
            "\u516b": 8,
            "\u4e5d": 9,
            "\u5341": 10,
            "\u5341\u4e00": 11,
            "\u5341\u4e8c": 12,
        }
        match = re.search(r"(\u5341\u4e00|\u5341\u4e8c|\u5341|[\u4e00\u4e8c\u5169\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d])\u6708(\d{1,2})(?:\u865f|\u65e5)?", normalized)
        if not match:
            return None
        month = chinese_months.get(match.group(1))
        day = int(match.group(2))
    else:
        month = int(match.group(1))
        day = int(match.group(2))
    try:
        value = date(date.today().year, month, day)
    except ValueError:
        return None
    return DateRange(start_date=value.isoformat(), end_date=value.isoformat())


def parse_expense_sort(raw_text: str) -> tuple[ExpenseSortBy, SortDirection]:
    normalized = re.sub(r"\s+", "", raw_text)
    if any(keyword in normalized for keyword in ("\u6309\u91d1\u984d\u7531\u5c0f\u5230\u5927", "\u91d1\u984d\u5c0f\u5230\u5927", "\u7531\u4f4e\u5230\u9ad8", "\u4fbf\u5b9c\u6392\u524d\u9762")):
        return "amount", "asc"
    if any(keyword in normalized for keyword in ("\u6309\u91d1\u984d\u6392\u5e8f", "\u6309\u91d1\u984d\u7531\u5927\u5230\u5c0f", "\u91d1\u984d\u5927\u5230\u5c0f", "\u7531\u9ad8\u5230\u4f4e", "\u6700\u8cb4\u6392\u524d\u9762")):
        return "amount", "desc"
    if any(keyword in normalized for keyword in ("\u6642\u9593\u65e9\u5230\u665a",)):
        return "time", "asc"
    if any(keyword in normalized for keyword in ("\u6309\u6642\u9593\u6392\u5e8f", "\u6642\u9593\u665a\u5230\u65e9")):
        return "time", "desc"
    if any(keyword in normalized for keyword in ("\u820a\u5230\u65b0", "\u65e5\u671f\u7531\u65e9\u5230\u665a")):
        return "date", "asc"
    if any(keyword in normalized for keyword in ("\u6309\u65e5\u671f\u6392\u5e8f", "\u65b0\u5230\u820a", "\u6700\u8fd1\u7684\u6392\u524d\u9762")):
        return "date", "desc"
    return "date", "desc"


def parse_expense_chart_type(raw_text: str, group_by: list[ExpenseGroupBy], include_ratio: bool) -> tuple[bool, ChartType]:
    normalized = re.sub(r"\s+", "", raw_text).lower()
    if any(keyword in normalized for keyword in ("\u6298\u7dda\u5716", "\u6298\u7dda", "linechart", "line")):
        return True, "line"
    if any(keyword in normalized for keyword in ("\u9577\u689d\u5716", "\u67f1\u72c0\u5716", "\u689d\u5f62\u5716", "barchart", "bar")):
        return True, "bar"
    if any(keyword in normalized for keyword in ("\u5713\u9905\u5716", "\u5713\u5f62\u5716", "\u9905\u5716", "piechart", "pie")):
        return True, "pie"
    if any(keyword in normalized for keyword in ("\u756b\u5716", "\u5716\u8868", "\u5716\u5f62", "\u5716")):
        if group_by and group_by[0] in {"day", "month"}:
            return True, "line"
        if group_by and group_by[0] == "category" and include_ratio:
            return True, "pie"
        return True, "bar"
    return False, "none"


def infer_expense_chart_type(group_by: list[ExpenseGroupBy], include_ratio: bool) -> ChartType:
    if group_by and group_by[0] in {"day", "month"}:
        return "line"
    if group_by and group_by[0] == "category" and include_ratio:
        return "pie"
    return "bar"


def extract_expense_chart_request(
    raw_text: str,
    group_by: list[ExpenseGroupBy] | None = None,
    include_ratio: bool = False,
) -> tuple[str, bool, ChartType]:
    clean_text = raw_text
    chart_type: ChartType = "none"
    patterns: list[tuple[str, ChartType]] = [
        (r"(?:[+＋,，、\s]*(?:折線圖|折線|line\s*chart|line)[\s。？?]*)", "line"),
        (r"(?:[+＋,，、\s]*(?:長條圖|柱狀圖|條形圖|bar\s*chart|bar)[\s。？?]*)", "bar"),
        (r"(?:[+＋,，、\s]*(?:圓餅圖|圓形圖|餅圖|pie\s*chart|pie)[\s。？?]*)", "pie"),
    ]
    for pattern, detected_type in patterns:
        if re.search(pattern, clean_text, flags=re.IGNORECASE):
            if chart_type == "none":
                chart_type = detected_type
            clean_text = re.sub(pattern, " ", clean_text, flags=re.IGNORECASE)

    generic_pattern = r"(?:[+＋,，、\s]*(?:圖表|畫圖|圖形|圖)[\s。？?]*)"
    if re.search(generic_pattern, clean_text, flags=re.IGNORECASE):
        if chart_type == "none":
            chart_type = infer_expense_chart_type(group_by or [], include_ratio)
        clean_text = re.sub(generic_pattern, " ", clean_text, flags=re.IGNORECASE)

    clean_text = re.sub(r"\s+", " ", clean_text).strip(" +＋,，、。?？")
    return clean_text or raw_text.strip(), chart_type != "none", chart_type


def parse_expense_group_by(raw_text: str) -> list[ExpenseGroupBy]:
    normalized = re.sub(r"\s+", "", raw_text)
    if any(keyword in normalized for keyword in ("\u6bcf\u5929", "\u4e00\u5929", "\u6bcf\u65e5", "\u6309\u5929", "\u9010\u65e5")):
        return ["day"]
    if any(keyword in normalized for keyword in ("\u6bcf\u6708", "\u6bcf\u500b\u6708", "\u6309\u6708", "\u5404\u6708", "\u6708\u4efd", "\u9010\u6708")):
        return ["month"]
    if is_category_breakdown_query(raw_text) or any(keyword in normalized for keyword in ("\u5404\u985e\u5225", "\u5404\u5206\u985e", "\u5206\u985e\u4f54\u6bd4", "\u985e\u5225\u4f54\u6bd4")):
        return ["category"]
    if any(keyword in normalized for keyword in ("\u5e97\u5bb6", "\u5546\u5bb6", "\u5e97\u540d", "\u5546\u6236")):
        return ["merchant"]
    return []


def parse_expense_metric(raw_text: str) -> ExpenseMetric:
    normalized = re.sub(r"\s+", "", raw_text)
    if any(keyword in normalized for keyword in ("\u5e73\u5747", "\u5e73\u5747\u82b1", "avg")):
        return "avg"
    if any(keyword in normalized for keyword in ("\u6700\u8cb4", "\u6700\u9ad8", "\u6700\u5927", "max")):
        return "max"
    if any(keyword in normalized for keyword in ("\u6700\u4fbf\u5b9c", "\u6700\u4f4e", "\u6700\u5c0f", "min")):
        return "min"
    if any(keyword in normalized for keyword in ("\u5e7e\u7b46", "\u7b46\u6578", "\u6b21\u6578", "count")):
        return "count"
    return "sum"


def infer_expense_query_mode(raw_text: str, group_by: list[ExpenseGroupBy]) -> ExpenseQueryMode:
    normalized = re.sub(r"\s+", "", raw_text)
    if any(keyword in normalized for keyword in ("\u660e\u7d30", "\u5217\u51fa", "\u6709\u54ea\u4e9b")):
        return "list_detail"
    if group_by:
        return "grouped_aggregate"
    return "aggregate"


def extract_expense_month_sequence(raw_text: str) -> list[int]:
    match = re.search(r"((?:\d{1,2}\s*){2,})\u6708", raw_text)
    if not match:
        return []
    months = [int(value) for value in re.findall(r"\d{1,2}", match.group(1))]
    result: list[int] = []
    for month in months:
        if 1 <= month <= 12 and month not in result:
            result.append(month)
    return result


def default_expense_query_dates(date_range_type: ExpenseDateRangeType, raw_text: str) -> list[DateRange]:
    today = date.today()
    if date_range_type == "today":
        return [DateRange(start_date=today.isoformat(), end_date=today.isoformat())]
    if date_range_type == "yesterday":
        yesterday = today - timedelta(days=1)
        return [DateRange(start_date=yesterday.isoformat(), end_date=yesterday.isoformat())]
    if date_range_type == "last_2_days":
        yesterday = today - timedelta(days=1)
        return [DateRange(start_date=yesterday.isoformat(), end_date=today.isoformat())]
    if date_range_type == "this_week":
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        return [DateRange(start_date=start.isoformat(), end_date=end.isoformat())]
    if date_range_type == "this_month":
        start = today.replace(day=1)
        return [month_date_range(today.year, today.month)]
    if date_range_type == "last_6_months":
        return [DateRange(start_date=(today - timedelta(days=182)).isoformat(), end_date=today.isoformat())]

    months = extract_query_months(raw_text)
    if months:
        return [month_date_range(today.year, month) for month in months if 1 <= month <= 12]

    dates = get_query_dates(raw_text)
    return [DateRange(start_date=value, end_date=value) for value in dates]


def parse_expense_query_fallback(text: str) -> ExpenseQuery:
    preliminary_group_by = parse_expense_group_by(text)
    preliminary_include_ratio = is_ratio_query(text) or preliminary_group_by == ["category"]
    clean_text, wants_chart, chart_type = extract_expense_chart_request(text, preliminary_group_by, preliminary_include_ratio)
    normalized = re.sub(r"\s+", "", clean_text)
    specific_day = extract_expense_specific_day(clean_text)
    months = extract_expense_month_sequence(clean_text) or extract_query_months(clean_text)
    group_by = parse_expense_group_by(clean_text)
    metric = parse_expense_metric(clean_text)
    mode = infer_expense_query_mode(clean_text, group_by)
    include_ratio = is_ratio_query(clean_text) or group_by == ["category"]
    if specific_day:
        date_range_type: ExpenseDateRangeType = "custom"
    elif "\u6628\u5929" in normalized and "\u4eca\u5929" in normalized:
        date_range_type: ExpenseDateRangeType = "last_2_days"
    elif "\u6628\u5929" in normalized:
        date_range_type = "yesterday"
    elif any(keyword in normalized for keyword in ("\u672c\u9031", "\u9019\u9031", "\u672c\u5468", "\u9019\u5468")):
        date_range_type = "this_week"
    elif "\u9019\u500b\u6708" in normalized or "\u672c\u6708" in normalized:
        date_range_type = "this_month"
    elif "\u6700\u8fd1\u534a\u5e74" in normalized or "\u8fd1\u534a\u5e74" in normalized:
        date_range_type = "last_6_months"
    elif len(months) >= 2:
        date_range_type = "specific_months"
    elif len(months) == 1:
        date_range_type = "specific_months"
    else:
        date_range_type = "today"

    if group_by == ["category"]:
        aggregation: ExpenseAggregation = "category_breakdown"
    elif mode == "list_detail":
        aggregation = "list"
    else:
        aggregation = "sum_and_ratio" if is_ratio_query(clean_text) else "sum"
    if any(keyword in normalized for keyword in ("\u660e\u7d30", "\u5217\u51fa", "\u6709\u54ea\u4e9b")):
        aggregation = "list"
    elif "\u6700\u8cb4" in normalized:
        aggregation = "top"
    elif "\u5e7e\u7b46" in normalized or "\u7b46\u6578" in normalized:
        aggregation = "count"

    exclude_categories = extract_exclude_categories(clean_text)
    exclude_keywords = extract_expense_exclude_keywords(clean_text)
    sort_by, sort_direction = parse_expense_sort(clean_text)
    ratio_denominator: RatioDenominator = "all_expenses" if aggregation in {"sum_and_ratio", "category_breakdown"} else "none"
    if aggregation == "category_breakdown" and (exclude_categories or exclude_keywords):
        ratio_denominator = "filtered_expenses"

    fallback_ranges = [specific_day] if specific_day else (
        [month_date_range(date.today().year, month) for month in months]
        if months
        else default_expense_query_dates(date_range_type, clean_text)
    )

    return ExpenseQuery(
        date_range_type=date_range_type,
        date_ranges=fallback_ranges,
        mode=mode,
        metric=metric,
        group_by=group_by,
        category=None if aggregation == "category_breakdown" else get_query_category(clean_text),
        include_categories=[],
        exclude_categories=exclude_categories,
        merchant=None,
        include_keywords=[],
        keywords=[],
        exclude_keywords=exclude_keywords,
        min_amount=None,
        max_amount=None,
        include_ratio=include_ratio,
        wants_chart=wants_chart,
        chart_type=chart_type,
        aggregation=aggregation,
        ratio_denominator=ratio_denominator,
        limit=100 if mode in {"grouped_aggregate", "aggregate"} else 15,
        sort_by=sort_by,
        sort_direction=sort_direction,
        confidence=0.65,
        reason="local fallback expense query parser",
    )


def parse_expense_query_with_openai(text: str) -> ExpenseQuery:
    today = date.today().isoformat()
    schema = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["list_detail", "aggregate", "grouped_aggregate"]},
            "metric": {"type": "string", "enum": ["sum", "count", "avg", "max", "min"]},
            "group_by": {
                "type": "array",
                "items": {"type": "string", "enum": ["day", "week", "month", "category", "merchant"]},
            },
            "date_range_type": {
                "type": "string",
                "enum": [
                    "today",
                    "yesterday",
                    "last_2_days",
                    "this_week",
                    "this_month",
                    "last_6_months",
                    "specific_months",
                    "custom",
                    "unspecified",
                ],
            },
            "date_ranges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                    },
                    "required": ["start_date", "end_date"],
                    "additionalProperties": False,
                },
            },
            "category": {"type": ["string", "null"], "enum": [*EXPENSE_CATEGORY_VALUES, None]},
            "include_categories": {
                "type": "array",
                "items": {"type": "string", "enum": EXPENSE_CATEGORY_VALUES},
            },
            "exclude_categories": {
                "type": "array",
                "items": {"type": "string", "enum": EXPENSE_CATEGORY_VALUES},
            },
            "merchant": {"type": ["string", "null"]},
            "include_keywords": {"type": "array", "items": {"type": "string"}},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "exclude_keywords": {"type": "array", "items": {"type": "string"}},
            "min_amount": {"type": ["integer", "null"]},
            "max_amount": {"type": ["integer", "null"]},
            "include_ratio": {"type": "boolean"},
            "wants_chart": {"type": "boolean"},
            "chart_type": {"type": "string", "enum": ["none", "line", "bar", "pie"]},
            "aggregation": {"type": "string", "enum": ["sum", "count", "list", "top", "sum_and_ratio", "category_breakdown"]},
            "ratio_denominator": {"type": "string", "enum": ["all_expenses", "filtered_expenses", "none"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            "sort_by": {"type": "string", "enum": ["date", "time", "amount", "created_at"]},
            "sort_direction": {"type": "string", "enum": ["asc", "desc"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": ["string", "null"]},
        },
        "required": [
            "mode",
            "metric",
            "group_by",
            "date_range_type",
            "date_ranges",
            "category",
            "include_categories",
            "exclude_categories",
            "merchant",
            "include_keywords",
            "keywords",
            "exclude_keywords",
            "min_amount",
            "max_amount",
            "include_ratio",
            "wants_chart",
            "chart_type",
            "aggregation",
            "ratio_denominator",
            "limit",
            "sort_by",
            "sort_direction",
            "confidence",
            "reason",
        ],
        "additionalProperties": False,
    }
    log_usage("openai", "chat_completion", detail=get_current_model())
    response = get_openai_client().chat.completions.create(
        model=get_current_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "各類別花費佔比、各分類花費佔比、分類佔比、每個類別花多少、各類別支出、七月各類別的花費佔比，都要輸出 aggregation=category_breakdown。"
                    "category_breakdown 時 category 必須是 null，include_categories 必須是空陣列，ratio_denominator=all_expenses，limit=50。"
                    "各類別不是 include_categories，不要把所有分類塞進 include_categories 後做合計。Python 會負責 SQL GROUP BY category 與比例計算。"                    "你是台灣 LINE 記帳 Bot 的支出查詢解析器，只輸出 structured JSON，不要產生 SQL，也不要計算金額或比例。"
                    f"今天是 {today}。"
                    "分類同義詞：飲食、吃飯、餐費、早餐、午餐、晚餐、宵夜、飲料、咖啡、點心都等於餐飲。"
                    "除了飲食外、扣掉飲食、不含飲食、排除餐飲，要輸出 exclude_categories=[餐飲]。"
                    "除了交通外輸出 exclude_categories=[交通]；除了購物外輸出 exclude_categories=[購物]。"
                    "七月或7月必須輸出 2026-07-01 到 2026-07-31；六月或6月必須輸出 2026-06-01 到 2026-06-30。"
                    "這個月、本月輸出本月1號到本月最後一天；不可以把七月解析成只有2026-07-01一天。"
                    "問佔比是多少、比例是多少、占全部多少時，aggregation=sum_and_ratio。"
                    "如果問除了X外的花費佔比，ratio_denominator=all_expenses；分子套用 exclude_categories，分母是同日期區間全部 expenses。"                    "你是台灣 LINE 記帳 Bot 的支出查詢解析器，只輸出 JSON，不要產生 SQL。"
                    f"今天是 {today}。"
                    "日期規則：今天=today，昨天=yesterday，昨天跟今天=last_2_days，"
                    "這個月/本月=this_month，最近半年/近半年=last_6_months。"
                    "三月和五月這種不連續月份要用 specific_months，date_ranges 分別列出每個月份，"
                    "不要合併成 3/1 到 5/31。沒有指定日期時預設 today。"
                    "分類規則：電話費/手機費/電信費=電話費；網路費/寬頻=網路費；"
                    "電費/台電=電費；水費/自來水=水費；瓦斯費/天然氣/桶裝瓦斯=瓦斯費。"
                    "不要把水電瓦斯電話網路費歸到生活用品。"
                    "aggregation：花多少/多少錢=sum，明細/列出=list，最貴=top，幾筆=count。"
                ),
            },
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "expense_query", "strict": True, "schema": schema},
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI returned an empty expense query.")
    query = ExpenseQuery.model_validate_json(content)
    if not query.date_ranges:
        query.date_ranges = default_expense_query_dates(query.date_range_type, text)
    return query


def parse_expense_query(text: str) -> ExpenseQuery:
    preliminary_group_by = parse_expense_group_by(text)
    preliminary_include_ratio = is_ratio_query(text) or preliminary_group_by == ["category"]
    clean_text, wants_chart, chart_type = extract_expense_chart_request(text, preliminary_group_by, preliminary_include_ratio)
    try:
        query = parse_expense_query_with_openai(clean_text)
        specific_day = extract_expense_specific_day(clean_text)
        months = extract_expense_month_sequence(clean_text) or extract_query_months(clean_text)
        text_group_by = parse_expense_group_by(clean_text)
        text_metric = parse_expense_metric(clean_text)
        query.metric = text_metric
        if text_group_by:
            query.group_by = text_group_by
            query.mode = "grouped_aggregate"
        else:
            query.mode = infer_expense_query_mode(clean_text, query.group_by)
        if specific_day:
            query.date_range_type = "custom"
            query.date_ranges = [specific_day]
        elif len(months) >= 1:
            query.date_range_type = "specific_months"
            query.date_ranges = [month_date_range(date.today().year, month) for month in months]
        elif any(keyword in re.sub(r"\s+", "", clean_text) for keyword in ("\u672c\u9031", "\u9019\u9031", "\u672c\u5468", "\u9019\u5468")):
            query.date_range_type = "this_week"
            query.date_ranges = default_expense_query_dates("this_week", clean_text)
        elif query.date_range_type != "custom":
            query.date_ranges = default_expense_query_dates(query.date_range_type, clean_text)
        fallback_category = get_query_category(clean_text)
        text_exclude_categories = extract_exclude_categories(clean_text)
        text_exclude_keywords = extract_expense_exclude_keywords(clean_text)
        if is_category_breakdown_query(clean_text):
            query.aggregation = "category_breakdown"
            query.mode = "grouped_aggregate"
            query.group_by = ["category"]
            query.include_ratio = True
            query.category = None
            query.include_categories = []
            query.limit = 100
        elif text_exclude_categories:
            query.exclude_categories = text_exclude_categories
            query.category = None
        elif fallback_category:
            query.category = fallback_category
        if text_exclude_categories:
            query.exclude_categories = text_exclude_categories
            if "\u623f\u8cb8" in text_exclude_keywords or "\u4fe1\u8cb8" in text_exclude_keywords:
                query.exclude_categories = [category for category in query.exclude_categories if category != "\u8cb8\u6b3e"]
        if text_exclude_keywords:
            query.exclude_keywords = text_exclude_keywords
            if "\u8cb8\u6b3e" in text_exclude_keywords and "\u623f\u8cb8" not in text_exclude_keywords and "\u4fe1\u8cb8" not in text_exclude_keywords:
                query.exclude_categories = [*query.exclude_categories, "\u8cb8\u6b3e"]
        query.category = normalize_expense_category(query.category)
        query.include_categories = normalize_category_list(query.include_categories)
        query.exclude_categories = normalize_category_list(query.exclude_categories)
        query.include_keywords = list(dict.fromkeys([*query.include_keywords, *query.keywords]))
        query.wants_chart = wants_chart
        query.chart_type = chart_type if wants_chart else "none"
        query.sort_by, query.sort_direction = parse_expense_sort(clean_text)
        if query.aggregation == "list" and query.limit == 10:
            query.limit = 15
        if query.mode == "list_detail":
            query.aggregation = "list"
            query.group_by = []
            query.wants_chart = False
            query.chart_type = "none"
        elif query.mode == "grouped_aggregate":
            if query.group_by == ["category"]:
                query.aggregation = "category_breakdown"
                query.include_ratio = query.include_ratio or is_ratio_query(clean_text)
            else:
                query.aggregation = "sum"
            if query.limit < 50:
                query.limit = 100
        if should_use_filtered_ratio_denominator(clean_text, query):
            query.ratio_denominator = "filtered_expenses"
        elif is_category_breakdown_query(clean_text) and query.ratio_denominator == "none":
            query.ratio_denominator = "all_expenses"
        if is_ratio_query(clean_text) and query.aggregation != "category_breakdown":
            query.aggregation = "sum_and_ratio"
            if query.ratio_denominator == "none":
                query.ratio_denominator = "all_expenses"
        if query.confidence >= 0.70:
            return query
    except Exception:
        logger.exception("OpenAI expense query parsing failed; using fallback for text: %s", text)
    fallback_query = parse_expense_query_fallback(clean_text)
    fallback_query.wants_chart = wants_chart
    fallback_query.chart_type = chart_type if wants_chart else "none"
    return fallback_query


def utility_category_keywords(category: str | None) -> list[str]:
    mapping = {
        "\u96fb\u8a71\u8cbb": ["\u96fb\u8a71\u8cbb", "\u624b\u6a5f\u8cbb", "\u96fb\u4fe1\u8cbb"],
        "\u7db2\u8def\u8cbb": ["\u7db2\u8def\u8cbb", "\u5bec\u983b"],
        "\u96fb\u8cbb": ["\u96fb\u8cbb", "\u53f0\u96fb", "\u96fb\u529b\u8cbb"],
        "\u6c34\u8cbb": ["\u6c34\u8cbb", "\u81ea\u4f86\u6c34"],
        "\u74e6\u65af\u8cbb": ["\u74e6\u65af\u8cbb", "\u5929\u7136\u6c23", "\u6876\u88dd\u74e6\u65af"],
    }
    return mapping.get(category or "", [])


def category_search_keywords(category: str | None) -> list[str]:
    if category and ("\u4ea4\u901a" in category or "\u9234" in category or "鈭" in category):
        return [
            "\u4ea4\u901a",
            "\u52a0\u6cb9",
            "\u505c\u8eca",
            "\u6377\u904b",
            "\u516c\u8eca",
            "\u8a08\u7a0b\u8eca",
            "Uber",
            "\u9ad8\u9435",
            "\u53f0\u9435",
            "\u706b\u8eca",
        ]
    if category and ("\u9910\u98f2" in category or "擗" in category):
        return [
            "\u9910\u98f2",
            "擗ㄡ",
            "\u65e9\u9910",
            "\u5348\u9910",
            "\u665a\u9910",
            "\u98f2\u6599",
            "\u5496\u5561",
            "\u9ede\u5fc3",
            "\u5bb5\u591c",
            "\u9910\u8cbb",
            "\u9eb5\u5305",
            "\u6d88\u591c",
        ]
    mapping = {
        "\u4ea4\u901a": [
            "\u4ea4\u901a",
            "\u52a0\u6cb9",
            "\u505c\u8eca",
            "\u6377\u904b",
            "\u516c\u8eca",
            "\u8a08\u7a0b\u8eca",
            "Uber",
            "\u9ad8\u9435",
            "\u53f0\u9435",
            "\u706b\u8eca",
        ],
        "\u9910\u98f2": [
            "\u9910\u98f2",
            "擗ㄡ",
            "\u65e9\u9910",
            "\u5348\u9910",
            "\u665a\u9910",
            "\u98f2\u6599",
            "\u5496\u5561",
            "\u9ede\u5fc3",
            "\u5bb5\u591c",
            "\u9910\u8cbb",
        ],
        "\u8cfc\u7269": [
            "\u8cfc\u7269",
            "\u8cb7\u8863\u670d",
            "\u8863\u670d",
            "\u6cbb\u88dd",
            "\u670d\u98fe",
            "\u978b\u5b50",
            "\u8932\u5b50",
            "\u5916\u5957",
        ],
        "\u751f\u6d3b\u7528\u54c1": [
            "\u5168\u806f",
            "\u5bb6\u6a02\u798f",
            "\u8d85\u5e02",
            "\u65e5\u7528\u54c1",
            "\u885b\u751f\u7d19",
        ],
    }
    keywords = utility_category_keywords(category) or mapping.get(category or "", [])
    if category and "\u9910\u98f2" in category:
        for keyword in ("擗ㄡ", "\u9eb5\u5305", "\u5bb5\u591c", "\u6d88\u591c"):
            if keyword not in keywords:
                keywords.append(keyword)
    return keywords


def text_match_sql(fields: tuple[str, ...], keywords: list[str]) -> tuple[str, list[str]]:
    clauses: list[str] = []
    params: list[str] = []
    for keyword in keywords:
        field_clauses = [f"COALESCE({field}, '') LIKE ?" for field in fields]
        clauses.append("(" + " OR ".join(field_clauses) + ")")
        params.extend([f"%{keyword}%"] * len(fields))
    return " AND ".join(clauses), params


def text_any_match_sql(fields: tuple[str, ...], keywords: list[str]) -> tuple[str, list[str]]:
    clauses: list[str] = []
    params: list[str] = []
    for keyword in keywords:
        for field in fields:
            clauses.append(f"COALESCE({field}, '') LIKE ?")
            params.append(f"%{keyword}%")
    return " OR ".join(clauses), params


def payable_item_type_to_expense_category(item_type: str | None) -> str:
    item = normalize_payable_item_type(item_type or "") or item_type or ""
    if item in {"\u623f\u8cb8", "\u4fe1\u8cb8", "\u8cb8\u6b3e"}:
        return "\u8cb8\u6b3e"
    if item in {"\u4fe1\u7528\u5361", "\u5361\u8cbb"}:
        return "\u4fe1\u7528\u5361"
    if item == "\u4fdd\u6bcd\u8cbb":
        return "\u4fdd\u6bcd\u8cbb"
    if item == "\u7ba1\u7406\u8cbb":
        return "\u7ba1\u7406\u8cbb"
    if item in {"\u5e7c\u7a1a\u5712", "\u5e7c\u5152\u5712"}:
        return "\u6559\u80b2"
    if item == "\u623f\u79df":
        return "\u623f\u79df"
    return "\u5176\u4ed6"


def get_actual_spending_rows(start_date: str, end_date: str, line_user_id: str | None) -> list[dict[str, object]]:
    with get_db() as conn:
        expense_rows = conn.execute(
            """
            SELECT id, date, expense_time, amount, currency, category, merchant, note, raw_text, created_at
            FROM expenses
            WHERE date BETWEEN ? AND ?
              AND currency = 'TWD'
              AND amount > 0
              AND raw_text NOT LIKE '刪除%'
              AND raw_text NOT LIKE '删除%'
              AND raw_text NOT LIKE '取消%'
              AND raw_text NOT LIKE '移除%'
              AND LOWER(raw_text) NOT LIKE 'delete%'
              AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
            """,
            (start_date, end_date, line_user_id, line_user_id),
        ).fetchall()
        paid_payable_rows = conn.execute(
            """
            SELECT id,
                   COALESCE(date(paid_at), due_date) AS paid_date,
                   amount,
                   currency,
                   item_type,
                   owner,
                   bank,
                   note,
                   due_date,
                   paid_at,
                   created_at
            FROM payables
            WHERE status = 'paid'
              AND currency = 'TWD'
              AND amount > 0
              AND COALESCE(date(paid_at), due_date) BETWEEN ? AND ?
              AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
            """,
            (start_date, end_date, line_user_id, line_user_id),
        ).fetchall()

    rows: list[dict[str, object]] = []
    for row in expense_rows:
        rows.append(
            {
                "id": int(row["id"]),
                "date": row["date"],
                "expense_time": row["expense_time"],
                "amount": int(row["amount"]),
                "currency": row["currency"],
                "category": row["category"],
                "merchant": row["merchant"],
                "note": row["note"],
                "raw_text": row["raw_text"],
                "created_at": row["created_at"],
                "source": "expense",
                "source_label": "\u4e00\u822c\u652f\u51fa",
            }
        )
    for row in paid_payable_rows:
        item_type = str(row["item_type"] or "")
        meta = " ".join(str(value) for value in (row["owner"], row["bank"]) if value)
        note = str(row["note"] or item_type)
        label = " ".join(part for part in (meta, item_type) if part).strip() or note
        rows.append(
            {
                "id": int(row["id"]),
                "date": row["paid_date"],
                "expense_time": None,
                "amount": int(row["amount"]),
                "currency": row["currency"],
                "category": payable_item_type_to_expense_category(item_type),
                "merchant": meta or None,
                "note": label,
                "raw_text": note,
                "created_at": row["created_at"],
                "source": "paid_payable",
                "source_label": "\u5df2\u7e73\u6b3e\u9805",
                "item_type": item_type,
                "due_date": row["due_date"],
                "paid_at": row["paid_at"],
            }
        )
    return rows


def actual_spending_rows_for_query(
    query: ExpenseQuery,
    line_user_id: str | None,
    *,
    include_exclusions: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ranges = query.date_ranges or default_expense_query_dates(query.date_range_type, "")
    for range_value in ranges:
        rows.extend(get_actual_spending_rows(range_value.start_date, range_value.end_date, line_user_id))
    return [row for row in rows if actual_spending_row_matches_query(row, query, include_exclusions=include_exclusions)]


def actual_spending_row_text(row: dict[str, object]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("raw_text", "merchant", "note", "category", "source_label", "item_type")
    )


def row_matches_any_keyword(row: dict[str, object], keywords: list[str]) -> bool:
    text = actual_spending_row_text(row)
    return any(keyword and keyword in text for keyword in keywords)


def row_matches_all_keywords(row: dict[str, object], keywords: list[str]) -> bool:
    text = actual_spending_row_text(row)
    return all((not keyword) or keyword in text for keyword in keywords)


def actual_spending_row_matches_query(
    row: dict[str, object],
    query: ExpenseQuery,
    *,
    include_exclusions: bool = True,
) -> bool:
    include_categories = normalize_category_list(query.include_categories)
    row_category = str(row.get("category") or "")
    if include_categories and row_category not in include_categories:
        return False
    if not include_categories and query.category:
        alias_keywords = category_search_keywords(query.category)
        if row_category != query.category and not row_matches_any_keyword(row, alias_keywords):
            return False
    if query.merchant and query.merchant not in str(row.get("merchant") or ""):
        return False
    include_keywords = list(dict.fromkeys([*query.include_keywords, *query.keywords]))
    if include_keywords and not row_matches_all_keywords(row, include_keywords):
        return False
    if include_exclusions:
        exclude_categories = normalize_category_list(query.exclude_categories)
        if exclude_categories and row_category in exclude_categories:
            return False
        if query.exclude_keywords and row_matches_any_keyword(row, query.exclude_keywords):
            return False
    amount = int(row.get("amount") or 0)
    if query.min_amount is not None and amount < query.min_amount:
        return False
    if query.max_amount is not None and amount > query.max_amount:
        return False
    return True


def expense_row_group_key(row: dict[str, object], group_by: ExpenseGroupBy) -> str:
    row_date = str(row.get("date") or "")
    if group_by == "month":
        return row_date[:7]
    if group_by == "category":
        return str(row.get("category") or "\u5176\u4ed6")
    if group_by == "merchant":
        return str(row.get("merchant") or row.get("note") or row.get("category") or "\u672a\u6307\u5b9a")
    return row_date


def metric_value_from_amounts(amounts: list[int], metric: ExpenseMetric) -> float:
    if not amounts:
        return 0
    if metric == "count":
        return float(len(amounts))
    if metric == "avg":
        return sum(amounts) / len(amounts)
    if metric == "max":
        return float(max(amounts))
    if metric == "min":
        return float(min(amounts))
    return float(sum(amounts))


def sort_actual_spending_rows(rows: list[dict[str, object]], query: ExpenseQuery) -> list[dict[str, object]]:
    reverse = query.sort_direction == "desc"
    if query.sort_by == "amount":
        return sorted(
            rows,
            key=lambda row: (int(row.get("amount") or 0), str(row.get("date") or ""), str(row.get("expense_time") or ""), int(row.get("id") or 0)),
            reverse=reverse,
        )
    if query.sort_by == "time":
        return sorted(
            rows,
            key=lambda row: (str(row.get("date") or ""), str(row.get("expense_time") or ""), int(row.get("id") or 0)),
            reverse=reverse,
        )
    if query.sort_by == "created_at":
        return sorted(
            rows,
            key=lambda row: (str(row.get("created_at") or ""), int(row.get("id") or 0)),
            reverse=reverse,
        )
    return sorted(
        rows,
        key=lambda row: (str(row.get("date") or ""), str(row.get("expense_time") or ""), int(row.get("id") or 0)),
        reverse=reverse,
    )


def build_expense_where_clauses(
    query: ExpenseQuery,
    line_user_id: str | None,
    *,
    include_exclusions: bool = True,
) -> tuple[str, list[str | int | None]]:
    ranges = query.date_ranges or default_expense_query_dates(query.date_range_type, "")
    date_sql = " OR ".join("(date BETWEEN ? AND ?)" for _ in ranges)
    where = [
        f"({date_sql})",
        "currency = ?",
        "amount > 0",
        "raw_text NOT LIKE '?芷%'",
        "raw_text NOT LIKE '?%'",
        "raw_text NOT LIKE '??%'",
        "raw_text NOT LIKE '蝘駁%'",
        "LOWER(raw_text) NOT LIKE 'delete%'",
        "(? IS NULL OR line_user_id = ? OR line_user_id IS NULL)",
    ]
    params: list[str | int | None] = []
    for range_value in ranges:
        params.extend([range_value.start_date, range_value.end_date])
    params.extend(["TWD", line_user_id, line_user_id])

    include_categories = normalize_category_list(query.include_categories)
    if include_categories:
        where.append(f"category IN ({','.join('?' for _ in include_categories)})")
        params.extend(include_categories)
    elif query.category:
        alias_keywords = category_search_keywords(query.category)
        if alias_keywords:
            text_sql, text_params = text_any_match_sql(("raw_text", "note", "merchant", "category"), alias_keywords)
            where.append(f"(category = ? OR ({text_sql}))")
            params.append(query.category)
            params.extend(text_params)
        else:
            where.append("category = ?")
            params.append(query.category)

    if query.merchant:
        where.append("COALESCE(merchant, '') LIKE ?")
        params.append(f"%{query.merchant}%")
    include_keywords = list(dict.fromkeys([*query.include_keywords, *query.keywords]))
    if include_keywords:
        text_sql, text_params = text_match_sql(("raw_text", "merchant", "note", "category"), include_keywords)
        where.append(text_sql)
        params.extend(text_params)
    if include_exclusions:
        exclude_categories = normalize_category_list(query.exclude_categories)
        if exclude_categories:
            where.append(f"category NOT IN ({','.join('?' for _ in exclude_categories)})")
            params.extend(exclude_categories)
        if query.exclude_keywords:
            text_sql, text_params = text_any_match_sql(("raw_text", "merchant", "note", "category"), query.exclude_keywords)
            where.append(f"NOT ({text_sql})")
            params.extend(text_params)
    if query.min_amount is not None:
        where.append("amount >= ?")
        params.append(query.min_amount)
    if query.max_amount is not None:
        where.append("amount <= ?")
        params.append(query.max_amount)

    return " AND ".join(where), params


def build_expense_query_from_structured(
    query: ExpenseQuery,
    line_user_id: str | None,
) -> tuple[str, list[str | int | None]]:
    return build_expense_where_clauses(query, line_user_id, include_exclusions=True)


def calculate_expense_summary_with_ratio(query: ExpenseQuery, line_user_id: str | None) -> dict:
    filtered_rows = actual_spending_rows_for_query(query, line_user_id, include_exclusions=True)
    denominator_query = query.model_copy(
        update={
            "category": None,
            "include_categories": [],
            "merchant": None,
            "include_keywords": [],
            "keywords": [],
            "min_amount": None,
            "max_amount": None,
        }
    )
    include_denominator_exclusions = query.ratio_denominator in {"filtered_expenses", "included_expenses"}
    denominator_rows = actual_spending_rows_for_query(
        denominator_query,
        line_user_id,
        include_exclusions=include_denominator_exclusions,
    )
    filtered_total = sum(int(row["amount"]) for row in filtered_rows)
    denominator_total = sum(int(row["amount"]) for row in denominator_rows)
    ratio_percent = (filtered_total / denominator_total * 100) if denominator_total else 0
    return {
        "filtered_total": filtered_total,
        "denominator_total": denominator_total,
        "ratio_percent": ratio_percent,
        "count": len(filtered_rows),
        "rows": sort_actual_spending_rows(filtered_rows, query)[: query.limit],
    }


def calculate_category_breakdown(query: ExpenseQuery, line_user_id: str | None) -> dict:
    breakdown_query = query.model_copy(update={"category": None, "include_categories": []})
    filtered_rows = actual_spending_rows_for_query(
        breakdown_query,
        line_user_id,
        include_exclusions=True,
    )
    denominator_query = query.model_copy(
        update={
            "category": None,
            "include_categories": [],
            "merchant": None,
            "include_keywords": [],
            "keywords": [],
            "min_amount": None,
            "max_amount": None,
        }
    )
    include_denominator_exclusions = query.ratio_denominator in {"filtered_expenses", "included_expenses"}
    denominator_rows = actual_spending_rows_for_query(
        denominator_query,
        line_user_id,
        include_exclusions=include_denominator_exclusions,
    )
    denominator_total = sum(int(row["amount"]) for row in denominator_rows)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in filtered_rows:
        grouped.setdefault(str(row.get("category") or "\u5176\u4ed6"), []).append(row)
    result_rows = []
    for category, rows in grouped.items():
        total = sum(int(row["amount"]) for row in rows)
        if total <= 0:
            continue
        ratio_percent = (total / denominator_total * 100) if denominator_total else 0
        result_rows.append(
            {
                "category": category,
                "count": len(rows),
                "total": total,
                "ratio_percent": ratio_percent,
            }
        )
    result_rows.sort(key=lambda row: int(row["total"]), reverse=True)
    return {"rows": result_rows, "denominator_total": denominator_total}


def build_expense_query(
    dates: list[str],
    category: Category | None,
    line_user_id: str | None,
) -> tuple[str, list[str | int | None]]:
    where = [
        f"date IN ({','.join('?' for _ in dates)})",
        "currency = ?",
        "amount > 0",
        "raw_text NOT LIKE '刪除%'",
        "raw_text NOT LIKE '删除%'",
        "raw_text NOT LIKE '取消%'",
        "raw_text NOT LIKE '移除%'",
        "LOWER(raw_text) NOT LIKE 'delete%'",
        "(? IS NULL OR line_user_id = ? OR line_user_id IS NULL)",
    ]
    params: list[str | int | None] = [*dates, "TWD", line_user_id, line_user_id]
    if category:
        where.append("category = ?")
        params.append(category)
    return " AND ".join(where), params


def expense_query_date_label(query: ExpenseQuery) -> str:
    return "\u3001".join(
        item.start_date if item.start_date == item.end_date else f"{item.start_date}~{item.end_date}"
        for item in query.date_ranges
    )


def expense_query_month_label(query: ExpenseQuery) -> str:
    if len(query.date_ranges) == 1:
        start = datetime.strptime(query.date_ranges[0].start_date, "%Y-%m-%d").date()
        end = datetime.strptime(query.date_ranges[0].end_date, "%Y-%m-%d").date()
        full_month = month_date_range(start.year, start.month)
        if full_month.start_date == query.date_ranges[0].start_date and full_month.end_date == query.date_ranges[0].end_date:
            return f"{start.month}\u6708"
    return "\u5168\u90e8"


def build_expense_order_by(query: ExpenseQuery) -> str:
    direction = "ASC" if query.sort_direction == "asc" else "DESC"
    if query.sort_by == "amount":
        return f"amount {direction}, date DESC, COALESCE(expense_time, '') DESC, id DESC"
    if query.sort_by == "time":
        return f"date {direction}, COALESCE(expense_time, '') {direction}, id {direction}"
    if query.sort_by == "created_at":
        return f"created_at {direction}, id {direction}"
    return f"date {direction}, COALESCE(expense_time, '') {direction}, id {direction}"


def format_expense_sort_label(query: ExpenseQuery) -> str:
    if query.sort_by == "amount":
        return "\u91d1\u984d\u7531\u5927\u5230\u5c0f" if query.sort_direction == "desc" else "\u91d1\u984d\u7531\u5c0f\u5230\u5927"
    if query.sort_by == "time":
        return "\u6642\u9593\u7531\u665a\u5230\u65e9" if query.sort_direction == "desc" else "\u6642\u9593\u7531\u65e9\u5230\u665a"
    if query.sort_by == "created_at":
        return "\u5efa\u7acb\u6642\u9593\u7531\u65b0\u5230\u820a" if query.sort_direction == "desc" else "\u5efa\u7acb\u6642\u9593\u7531\u820a\u5230\u65b0"
    return "\u65e5\u671f\u7531\u65b0\u5230\u820a" if query.sort_direction == "desc" else "\u65e5\u671f\u7531\u820a\u5230\u65b0"


def expense_metric_sql(metric: ExpenseMetric) -> str:
    if metric == "count":
        return "COUNT(*)"
    if metric == "avg":
        return "AVG(amount)"
    if metric == "max":
        return "MAX(amount)"
    if metric == "min":
        return "MIN(amount)"
    return "SUM(amount)"


def expense_group_expression(group_by: ExpenseGroupBy) -> str:
    if group_by == "month":
        return "strftime('%Y-%m', date)"
    if group_by == "category":
        return "COALESCE(category, '\u5176\u4ed6')"
    if group_by == "merchant":
        return "COALESCE(NULLIF(merchant, ''), NULLIF(note, ''), category, '\u672a\u6307\u5b9a')"
    return "date"


def generate_month_range(start_date: str, end_date: str) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    current = date(start.year, start.month, 1)
    end_month = date(end.year, end.month, 1)
    months: list[str] = []
    while current <= end_month:
        months.append(f"{current.year:04d}-{current.month:02d}")
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return months


def iter_expense_group_keys(query: ExpenseQuery, group_by: ExpenseGroupBy) -> list[str]:
    keys: list[str] = []
    if group_by == "day":
        for range_value in query.date_ranges:
            start = datetime.strptime(range_value.start_date, "%Y-%m-%d").date()
            end = datetime.strptime(range_value.end_date, "%Y-%m-%d").date()
            current = start
            while current <= end:
                keys.append(current.isoformat())
                current += timedelta(days=1)
    elif group_by == "month":
        if not query.date_ranges:
            return keys
        starts = [datetime.strptime(range_value.start_date, "%Y-%m-%d").date() for range_value in query.date_ranges]
        ends = [datetime.strptime(range_value.end_date, "%Y-%m-%d").date() for range_value in query.date_ranges]
        keys = generate_month_range(min(starts).isoformat(), max(ends).isoformat())
    return keys


def execute_expense_query(query: ExpenseQuery, line_user_id: str | None) -> dict:
    where_sql, params = build_expense_where_clauses(query, line_user_id, include_exclusions=True)
    rows = actual_spending_rows_for_query(query, line_user_id, include_exclusions=True)
    logger.info(
        "expense_query intent=%s metric=%s group_by=%s date_from=%s date_to=%s category=%s user_id=%s sql_where=%s params=%s result_count=%s",
        query.mode,
        query.metric,
        query.group_by,
        query.date_ranges[0].start_date if query.date_ranges else None,
        query.date_ranges[-1].end_date if query.date_ranges else None,
        query.category,
        line_user_id,
        where_sql,
        params,
        len(rows),
    )

    if query.mode == "list_detail":
        sorted_rows = sort_actual_spending_rows(rows, query)[: query.limit]
        logger.info("expense_query list result_count=%d", len(sorted_rows))
        return {"mode": "list_detail", "rows": sorted_rows}

    if query.mode == "grouped_aggregate" and query.group_by:
        group_by = query.group_by[0]
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            grouped.setdefault(expense_row_group_key(row, group_by), []).append(row)
        logger.info("expense_query grouped raw_count=%d group_by=%s", len(grouped), group_by)

        row_map: dict[str, dict[str, object]] = {}
        for group_key, group_rows in grouped.items():
            amounts = [int(row["amount"]) for row in group_rows]
            row_map[group_key] = {
                "group_key": group_key,
                "count": len(group_rows),
                "total": sum(amounts),
                "avg": (sum(amounts) / len(amounts)) if amounts else 0,
                "max": max(amounts) if amounts else 0,
                "min": min(amounts) if amounts else 0,
                "metric_value": metric_value_from_amounts(amounts, query.metric),
            }
        result_rows: list[dict[str, object]] = []
        fill_keys = iter_expense_group_keys(query, group_by)
        if fill_keys:
            for key in fill_keys:
                row = row_map.get(key)
                result_rows.append(
                    {
                        "group": key,
                        "count": int(row["count"]) if row else 0,
                        "total": int(row["total"]) if row else 0,
                        "avg": float(row["avg"]) if row else 0,
                        "max": int(row["max"]) if row else 0,
                        "min": int(row["min"]) if row else 0,
                        "ratio_percent": 0,
                    }
                )
        else:
            for row in row_map.values():
                result_rows.append(
                    {
                        "group": str(row["group_key"]),
                        "count": int(row["count"]),
                        "total": int(row["total"]),
                        "avg": float(row["avg"]),
                        "max": int(row["max"]),
                        "min": int(row["min"]),
                        "ratio_percent": 0,
                    }
                )
            reverse = query.sort_direction == "desc"
            if query.sort_by in {"total", "amount"}:
                result_rows.sort(key=lambda row: (int(row["total"]), str(row["group"])), reverse=reverse)
            elif query.sort_by == "count":
                result_rows.sort(key=lambda row: (int(row["count"]), str(row["group"])), reverse=reverse)
            else:
                result_rows.sort(key=lambda row: str(row["group"]), reverse=reverse)
            result_rows = result_rows[: query.limit]

        denominator_total = 0
        if query.include_ratio:
            denominator_query = query.model_copy(
                update={
                    "category": None,
                    "include_categories": [],
                    "merchant": None,
                    "include_keywords": [],
                    "keywords": [],
                    "min_amount": None,
                    "max_amount": None,
                }
            )
            denominator_rows = actual_spending_rows_for_query(
                denominator_query,
                line_user_id,
                include_exclusions=query.ratio_denominator == "filtered_expenses",
            )
            denominator_total = sum(int(row["amount"]) for row in denominator_rows)
            for row in result_rows:
                total = int(row["total"])
                row["ratio_percent"] = (total / denominator_total * 100) if denominator_total else 0

        total_sum = sum(int(row["total"]) for row in result_rows)
        logger.info("expense_query daily/grouped result=%s", result_rows)
        return {
            "mode": "grouped_aggregate",
            "group_by": group_by,
            "rows": result_rows,
            "total": total_sum,
            "denominator_total": denominator_total,
        }

    amounts = [int(row["amount"]) for row in rows]
    total = sum(amounts)
    count = len(amounts)
    avg = (total / count) if count else 0
    max_amount = max(amounts) if amounts else 0
    min_amount = min(amounts) if amounts else 0
    max_rows = [row for row in rows if int(row["amount"]) == max_amount] if rows else []
    min_rows = [row for row in rows if int(row["amount"]) == min_amount] if rows else []
    max_row = sort_actual_spending_rows(max_rows, query)[0] if max_rows else None
    min_row = sort_actual_spending_rows(min_rows, query)[-1] if min_rows else None
    logger.info("expense_query aggregate result_count=%s total=%s", count, total)
    return {
        "mode": "aggregate",
        "count": count,
        "total": total,
        "avg": float(avg),
        "max": max_amount,
        "min": min_amount,
        "metric_value": metric_value_from_amounts(amounts, query.metric),
        "max_item_name": str((max_row or {}).get("merchant") or (max_row or {}).get("note") or (max_row or {}).get("category") or (max_row or {}).get("raw_text")) if max_row else None,
        "max_item_count": len(max_rows),
        "min_item_name": str((min_row or {}).get("merchant") or (min_row or {}).get("note") or (min_row or {}).get("category") or (min_row or {}).get("raw_text")) if min_row else None,
        "min_item_count": len(min_rows),
    }


def build_expense_result_text(query: ExpenseQuery, result: dict) -> str:
    lines = ["\u67e5\u8a62\u7d50\u679c", f"\u65e5\u671f\uff1a{expense_query_date_label(query)}"]
    if query.category:
        lines.append(f"\u5206\u985e\uff1a{query.category}")
    if query.exclude_keywords:
        lines.append(f"\u6392\u9664\u9805\u76ee\uff1a{'\u3001'.join(query.exclude_keywords)}")

    if result["mode"] == "grouped_aggregate":
        group_by = str(result["group_by"])
        title = {
            "day": "\u6bcf\u65e5\u82b1\u8cbb",
            "month": "\u6bcf\u6708\u82b1\u8cbb",
            "category": "\u5404\u985e\u5225\u4f54\u6bd4" if query.include_ratio else "\u5404\u985e\u5225\u82b1\u8cbb",
            "merchant": "\u5404\u5e97\u5bb6\u82b1\u8cbb",
        }.get(group_by, "\u5206\u7d44\u7d71\u8a08")
        if query.include_ratio:
            denominator_label = "\u6392\u9664\u5f8c\u652f\u51fa" if query.ratio_denominator == "filtered_expenses" else "\u5168\u90e8\u652f\u51fa"
            lines.append(f"{denominator_label}\uff1aTWD {int(result['denominator_total'])}")
        lines.append("")
        lines.append(f"{title}\uff1a")
        for row in result["rows"]:
            line = f"{row['group']}\uff1aTWD {int(row['total'])}"
            if query.include_ratio:
                line += f"\uff0c{float(row['ratio_percent']):.2f}%"
            line += f"\uff0c{int(row['count'])} \u7b46"
            lines.append(line)
        lines.append(f"\u5408\u8a08\uff1aTWD {int(result['total'])}")
        return "\n".join(lines)

    if result["mode"] == "aggregate":
        max_suffix = ""
        if result.get("max_item_name"):
            max_suffix = f"\uff08{result['max_item_name']}"
            if int(result.get("max_item_count", 0)) > 1:
                max_suffix += f"\u7b49 {int(result['max_item_count'])} \u7b46"
            max_suffix += "\uff09"
        min_suffix = ""
        if result.get("min_item_name"):
            min_suffix = f"\uff08{result['min_item_name']}"
            if int(result.get("min_item_count", 0)) > 1:
                min_suffix += f"\u7b49 {int(result['min_item_count'])} \u7b46"
            min_suffix += "\uff09"
        lines.extend(
            [
                f"\u7b46\u6578\uff1a{int(result['count'])}",
                f"\u7e3d\u82b1\u8cbb\uff1aTWD {int(result['total'])}",
                f"\u5e73\u5747\uff1aTWD {float(result['avg']):.0f}",
                f"\u6700\u9ad8\uff1aTWD {int(result['max'])}{max_suffix}",
                f"\u6700\u4f4e\uff1aTWD {int(result['min'])}{min_suffix}",
            ]
        )
        return "\n".join(lines)

    lines.append(f"\u6392\u5e8f\uff1a{format_expense_sort_label(query)}")
    for row in result["rows"]:
        label = row["merchant"] or row["note"] or row["category"]
        time_text = f" {row['expense_time']}" if row["expense_time"] else ""
        lines.append(f"{row['date']}{time_text} {label} TWD {row['amount']}")
    return "\n".join(lines)


def generate_expense_chart(query: ExpenseQuery, result: dict, line_user_id: str | None = None) -> str | None:
    start = datetime.now(timezone.utc)
    if not query.wants_chart or query.chart_type == "none":
        return None
    if result.get("mode") != "grouped_aggregate":
        return None
    if get_app_env() != "test":
        load_dotenv(override=False)
    public_base_url = get_public_base_url()
    if not public_base_url:
        logger.warning("PUBLIC_BASE_URL is not configured; skip chart generation.")
        return None
    rows = list(result.get("rows", []))
    if query.chart_type == "pie":
        rows = [row for row in rows if int(row.get("total", 0)) > 0]
    if not rows:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        logger.exception("matplotlib is not available; skip chart generation.")
        return None

    labels = [str(row["group"]) for row in rows]
    values = [int(row["total"]) for row in rows]
    if sum(values) <= 0:
        return None

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    title = {
        "day": "\u6bcf\u65e5\u82b1\u8cbb",
        "month": "\u6bcf\u6708\u82b1\u8cbb",
        "category": "\u5404\u985e\u5225\u82b1\u8cbb",
        "merchant": "\u5404\u5e97\u5bb6\u82b1\u8cbb",
    }.get(query.group_by[0] if query.group_by else "", "\u82b1\u8cbb\u5716\u8868")
    payload = {
        "chart_type": query.chart_type,
        "date_range": [range_value.model_dump() for range_value in query.date_ranges],
        "category": query.category,
        "include_categories": query.include_categories,
        "exclude_categories": query.exclude_categories,
        "include_keywords": query.include_keywords,
        "keywords": query.keywords,
        "exclude_keywords": query.exclude_keywords,
        "group_by": query.group_by,
        "labels": labels,
        "values": values,
        "title": title,
        "line_user_id": line_user_id,
    }
    filename = f"chart_{chart_cache_key(payload)}.png"
    path = CHART_DIR / filename
    if path.exists():
        return f"{public_base_url.rstrip('/')}/static/charts/{filename}"

    plt.figure(figsize=(8, 4.8))
    if query.chart_type == "pie":
        if query.group_by[:1] != ["category"]:
            return None
        plt.pie(values, labels=labels, autopct="%1.1f%%")
        plt.axis("equal")
    elif query.chart_type == "line":
        plt.plot(labels, values, marker="o")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("TWD")
    else:
        plt.bar(labels, values)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("TWD")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    log_usage("chart", "generate", detail=query.chart_type, success=True, latency_ms=int((datetime.now(timezone.utc) - start).total_seconds() * 1000))
    return f"{public_base_url.rstrip('/')}/static/charts/{filename}"


def build_expense_reply(raw_text: str, line_user_id: str | None, route: ActionRoute | None = None) -> dict[str, str | None]:
    query = parse_expense_query(raw_text)
    clean_text, need_chart, chart_type = extract_expense_chart_request(raw_text, query.group_by, query.include_ratio)
    route_category = get_query_category(raw_text, route.category) if route and route.category else None
    if query.mode != "grouped_aggregate" and route_category and route_category != "\u5176\u4ed6":
        query.category = route_category
    logger.info(
        "expense_reply raw_text=%s clean_text=%s need_chart=%s chart_type=%s intent=%s date_from=%s date_to=%s category=%s user_id=%s",
        raw_text,
        clean_text,
        need_chart,
        chart_type,
        query.mode,
        query.date_ranges[0].start_date if query.date_ranges else None,
        query.date_ranges[-1].end_date if query.date_ranges else None,
        query.category,
        line_user_id,
    )
    result = execute_expense_query(query, line_user_id)
    if query.mode == "grouped_aggregate" and query.group_by[:1] == ["day"]:
        logger.info("expense_reply daily_data=%s", result.get("rows"))
    text = build_expense_result_text(query, result)
    image_url = generate_expense_chart(query, result, line_user_id)
    if query.wants_chart and not image_url:
        text += "\n\n圖表未產生：請確認 PUBLIC_BASE_URL 已設定，且查詢結果有可繪製的金額。"
    return {"text": text, "image_url": image_url}


def build_summary_reply(raw_text: str, line_user_id: str | None, route: ActionRoute | None = None) -> str:
    if any(keyword in raw_text for keyword in ("最貴", "最高")):
        return build_top_expense_reply(raw_text, line_user_id)

    query = parse_expense_query(raw_text)
    if route and route.category and get_query_category(raw_text, route.category) not in {None, "\u5176\u4ed6", "\u5176他", "其他", "?嗡?"}:
        query.category = get_query_category(raw_text, route.category)
    else:
        fallback_category = get_query_category(raw_text)
        if fallback_category:
            query.category = fallback_category
    if query.aggregation not in {"sum_and_ratio", "category_breakdown"}:
        query.aggregation = "sum"
    dates = [item.start_date for item in query.date_ranges]
    category = query.category
    user_key = line_user_id or "_global"
    last_query_by_user[user_key] = {"dates": dates, "category": category}

    if query.aggregation == "category_breakdown":
        summary = calculate_category_breakdown(query, line_user_id)
        month_label = expense_query_month_label(query)
        lines = [
            "\u67e5\u8a62\u7d50\u679c",
            f"\u65e5\u671f\uff1a{expense_query_date_label(query)}",
        ]
        if query.exclude_categories:
            lines.append(f"\u6392\u9664\u5206\u985e\uff1a{'\u3001'.join(query.exclude_categories)}")
        if query.exclude_keywords:
            lines.append(f"\u6392\u9664\u9805\u76ee\uff1a{'\u3001'.join(query.exclude_keywords)}")
        denominator_label = f"{month_label}\u6392\u9664\u5f8c\u652f\u51fa" if query.ratio_denominator == "filtered_expenses" else f"{month_label}\u5168\u90e8\u652f\u51fa"
        lines.extend([
            f"{denominator_label}\uff1aTWD {summary['denominator_total']}",
            "",
            "\u5404\u985e\u5225\u4f54\u6bd4\uff1a",
        ])
        for row in summary["rows"]:
            lines.append(
                f"{row['category']}\uff1aTWD {row['total']}\uff0c{row['ratio_percent']:.2f}%\uff0c{row['count']} \u7b46"
            )
        return "\n".join(lines)

    if query.aggregation == "sum_and_ratio":
        summary = calculate_expense_summary_with_ratio(query, line_user_id)
        lines = [
            "\u67e5\u8a62\u7d50\u679c",
            f"\u65e5\u671f\uff1a{expense_query_date_label(query)}",
        ]
        if query.exclude_categories:
            lines.append(f"\u6392\u9664\u5206\u985e\uff1a{'\u3001'.join(query.exclude_categories)}")
        if query.exclude_keywords:
            lines.append(f"\u6392\u9664\u9805\u76ee\uff1a{'\u3001'.join(query.exclude_keywords)}")
        if query.include_categories:
            lines.append(f"\u5206\u985e\uff1a{'\u3001'.join(query.include_categories)}")
        elif query.category:
            lines.append(f"\u5206\u985e\uff1a{query.category}")
        denominator_label = f"{expense_query_month_label(query)}\u6392\u9664\u5f8c\u652f\u51fa" if query.ratio_denominator == "filtered_expenses" else f"{expense_query_month_label(query)}\u5168\u90e8\u652f\u51fa"
        lines.extend(
            [
                f"\u7e3d\u82b1\u8cbb\uff1aTWD {summary['filtered_total']}",
                f"\u4f54{denominator_label}\uff1a{summary['ratio_percent']:.2f}%",
                f"{denominator_label}\uff1aTWD {summary['denominator_total']}",
            ]
        )
        return "\n".join(lines)

    where_sql, params = build_expense_query_from_structured(query, line_user_id)

    with get_db() as conn:
        row = conn.execute(
            f"""
            SELECT COUNT(*) AS count, COALESCE(SUM(amount), 0) AS total
            FROM expenses
            WHERE {where_sql}
            """,
            params,
        ).fetchone()
        rows = conn.execute(
            f"""
            SELECT id, date, expense_time, amount, currency, category, merchant, note
            FROM expenses
            WHERE {where_sql}
            ORDER BY date DESC, expense_time DESC, id DESC
            LIMIT 10
            """,
            params,
        ).fetchall()

    lines = [
        "查詢結果",
        f"日期：{'、'.join(dates)}",
        f"分類：{category or '全部分類'}",
        f"筆數：{int(row['count'])}",
        f"總花費：TWD {int(row['total'])}",
    ]
    if rows:
        lines.append("明細：")
        for detail in rows:
            label = detail["merchant"] or detail["note"] or detail["category"]
            time_text = f" {detail['expense_time']}" if detail["expense_time"] else ""
            lines.append(f"{detail['date']}{time_text} {label} TWD {detail['amount']}")

    return "\n".join(lines)


def build_top_expense_reply(raw_text: str, line_user_id: str | None, route: ActionRoute | None = None) -> str:
    user_key = line_user_id or "_global"
    previous = last_query_by_user.get(user_key, {})
    query = parse_expense_query(raw_text)
    if route and route.category and get_query_category(raw_text, route.category) not in {None, "\u5176\u4ed6", "\u5176他", "其他", "?嗡?"}:
        query.category = get_query_category(raw_text, route.category)
    else:
        fallback_category = get_query_category(raw_text)
        if fallback_category:
            query.category = fallback_category
    query.aggregation = "top"
    category = query.category
    dates = [item.start_date for item in query.date_ranges]

    if category is None and isinstance(previous.get("category"), str):
        category = previous["category"]  # type: ignore[assignment]
    if not any(keyword in raw_text for keyword in ("今天", "昨天", "這兩天", "近兩天")) and isinstance(previous.get("dates"), list):
        dates = [str(value) for value in previous["dates"]]

    last_query_by_user[user_key] = {"dates": dates, "category": category}
    where_sql, params = build_expense_query_from_structured(query, line_user_id)

    with get_db() as conn:
        row = conn.execute(
            f"""
            SELECT id, date, expense_time, amount, currency, category, merchant, note
            FROM expenses
            WHERE {where_sql}
            ORDER BY amount DESC, date DESC, expense_time DESC, id DESC
            LIMIT 1
            """,
            params,
        ).fetchone()

    if not row:
        return f"查不到符合條件的花費。\n日期：{'、'.join(dates)}\n分類：{category or '全部分類'}"

    label = row["merchant"] or row["note"] or row["category"]
    time_text = f" {row['expense_time']}" if row["expense_time"] else ""
    return (
        "最貴的一筆是：\n"
        f"日期：{row['date']}{time_text}\n"
        f"分類：{row['category']}\n"
        f"項目：{label}\n"
        f"金額：{row['currency']} {row['amount']}"
    )


def build_list_reply(raw_text: str, line_user_id: str | None, route: ActionRoute | None = None) -> str:
    query = parse_expense_query(raw_text)
    if route and route.category and get_query_category(raw_text, route.category) not in {None, "\u5176\u4ed6", "\u5176他", "其他", "?嗡?"}:
        query.category = get_query_category(raw_text, route.category)
    else:
        fallback_category = get_query_category(raw_text)
        if fallback_category:
            query.category = fallback_category
    query.aggregation = "list"
    dates = [item.start_date for item in query.date_ranges]
    category = query.category
    user_key = line_user_id or "_global"
    last_query_by_user[user_key] = {"dates": dates, "category": category}

    where_sql, params = build_expense_query_from_structured(query, line_user_id)
    order_by = build_expense_order_by(query)
    with get_db() as conn:
        rows = conn.execute(
            f"""
            SELECT id, date, expense_time, amount, currency, category, merchant, note
            FROM expenses
            WHERE {where_sql}
            ORDER BY {order_by}
            LIMIT ?
            """,
            [*params, query.limit],
        ).fetchall()

    if not rows:
        return f"查無明細\n日期：{'、'.join(dates)}\n分類：{category or '全部分類'}"

    lines = [
        "花費明細",
        f"日期：{'、'.join(dates)}",
        f"分類：{category or '全部分類'}",
    ]
    lines.append(f"\u6392\u5e8f\uff1a{format_expense_sort_label(query)}")
    total = 0
    for row in rows:
        total += int(row["amount"])
        label = row["merchant"] or row["note"] or row["category"]
        time_text = f" {row['expense_time']}" if row["expense_time"] else ""
        lines.append(f"{row['date']}{time_text} {label} TWD {row['amount']}")

    lines.append(f"小計：TWD {total}")
    if len(rows) == query.limit:
        lines.append("只顯示最近 15 筆。")

    return "\n".join(lines)


def find_expense_candidates(
    raw_text: str,
    line_user_id: str | None,
    *,
    limit: int = 10,
) -> list[sqlite3.Row]:
    dates = get_query_dates(raw_text)
    category = get_query_category(raw_text)
    amount_match = re.search(r"(\d+)", raw_text)
    keywords = get_delete_search_keywords(raw_text)

    where = [
        f"date IN ({','.join('?' for _ in dates)})",
        "currency = ?",
        "amount > 0",
        "raw_text NOT LIKE '刪除%'",
        "raw_text NOT LIKE '删除%'",
        "raw_text NOT LIKE '取消%'",
        "raw_text NOT LIKE '移除%'",
        "LOWER(raw_text) NOT LIKE 'delete%'",
        "(? IS NULL OR line_user_id = ? OR line_user_id IS NULL)",
    ]
    params: list[str | int | None] = [*dates, "TWD", line_user_id, line_user_id]
    if category:
        where.append("category = ?")
        params.append(category)
    if amount_match:
        where.append("amount = ?")
        params.append(int(amount_match.group(1)))

    with get_db() as conn:
        rows = conn.execute(
            f"""
            SELECT id, date, expense_time, amount, currency, category, merchant, note, raw_text
            FROM expenses
            WHERE {where_sql}
            ORDER BY date DESC, id DESC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()

    if not keywords:
        return rows

    filtered_rows = []
    for row in rows:
        searchable = " ".join(
            str(value or "")
            for value in (row["raw_text"], row["merchant"], row["note"], row["category"])
        )
        if all(keyword in searchable for keyword in keywords):
            filtered_rows.append(row)
    return filtered_rows


def get_delete_search_keywords(raw_text: str) -> list[str]:
    cleaned = raw_text
    stopwords = (
        "確認",
        "確定",
        "刪除",
        "删除",
        "取消",
        "移除",
        "delete",
        "今天",
        "昨天",
        "前天",
        "這兩天",
        "近兩天",
        "上午",
        "下午",
        "晚上",
        "早上",
        "中午",
        "凌晨",
        "花費",
        "支出",
        "消費",
        "記帳",
        "的",
    )
    for stopword in stopwords:
        cleaned = cleaned.replace(stopword, " ")
    cleaned = re.sub(r"[\d#：:，,。.\s]+", " ", cleaned)
    return [token for token in cleaned.split() if token]


def build_delete_candidates_reply(raw_text: str, line_user_id: str | None) -> str:
    rows = find_expense_candidates(raw_text, line_user_id)
    dates = get_query_dates(raw_text)
    if not rows:
        return f"找不到可刪除的花費。\n日期：{'、'.join(dates)}"

    if len(rows) == 1:
        row = rows[0]
        if line_user_id:
            pending_delete_by_user[line_user_id] = int(row["id"])
        label = row["merchant"] or row["note"] or row["category"]
        time_text = f" {row['expense_time']}" if row["expense_time"] else ""
        return (
            "找到 1 筆符合的花費：\n"
            f"{row['date']}{time_text} {label} TWD {row['amount']}\n"
            "若要刪除，請輸入：確認刪除"
        )

    lines = [
        "找到以下花費，請回覆要刪除的編號：",
    ]
    for row in rows:
        label = row["merchant"] or row["note"] or row["category"]
        time_text = f" {row['expense_time']}" if row["expense_time"] else ""
        lines.append(f"{row['date']}{time_text} {label} TWD {row['amount']}（代號 {row['id']}）")
    lines.append("例如：刪除代號 54")
    return "\n".join(lines)


def normalize_delete_text(text: str) -> str:
    cleaned = text
    for keyword in ("確認", "確定", "刪除", "删除", "取消", "移除", "delete", "今天的", "今天", "這筆", "那筆", "的", "代號"):
        cleaned = cleaned.replace(keyword, "")
    return cleaned.strip()


def row_to_expense(row: sqlite3.Row) -> ExpenseEntry:
    return ExpenseEntry(
        date=row["date"],
        time=row["expense_time"] if "expense_time" in row.keys() else None,
        amount=row["amount"],
        currency=row["currency"],
        category=row["category"],
        merchant=row["merchant"],
        note=row["note"],
        confidence=row["confidence"],
    )


def count_similar_expenses(expense: ExpenseEntry, line_user_id: str | None) -> int:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM expenses
            WHERE date = ?
              AND amount = ?
              AND currency = ?
              AND category = ?
              AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
            """,
            (
                expense.date,
                expense.amount,
                expense.currency,
                expense.category,
                line_user_id,
                line_user_id,
            ),
        ).fetchone()
    return int(row["count"])


def delete_expense(raw_text: str, line_user_id: str | None) -> DeleteResult:
    if is_delete_all_intent(raw_text) and not is_confirm_delete_all_intent(raw_text):
        return DeleteResult(
            deleted=False,
            reason="這會刪除所有記帳資料。若確定要刪除，請輸入「確認刪除所有花費」。",
        )

    if is_confirm_delete_all_intent(raw_text):
        with get_db() as conn:
            cursor = conn.execute(
                """
                DELETE FROM expenses
                WHERE ? IS NULL OR line_user_id = ? OR line_user_id IS NULL
                """,
                (line_user_id, line_user_id),
            )
            conn.commit()
            return DeleteResult(
                deleted=False,
                reason=f"已刪除所有記帳資料，共 {cursor.rowcount} 筆。",
            )

    id_match = re.search(r"(?:#|編號\s*|代號\s*)?(\d+)", raw_text)
    normalized_text = normalize_delete_text(raw_text)

    with get_db() as conn:
        if is_confirm_delete_intent(raw_text) and not id_match:
            pending_id = pending_delete_by_user.get(line_user_id or "")
            if not pending_id:
                return DeleteResult(deleted=False, reason="目前沒有待確認刪除的花費。請先輸入例如「刪除昨天的消夜」。")

            row = conn.execute(
                """
                SELECT * FROM expenses
                WHERE id = ?
                  AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
                """,
                (pending_id, line_user_id, line_user_id),
            ).fetchone()
            if not row:
                pending_delete_by_user.pop(line_user_id or "", None)
                return DeleteResult(deleted=False, reason="找不到剛才那筆待刪花費，可能已經刪除。")

            conn.execute("DELETE FROM expenses WHERE id = ?", (pending_id,))
            conn.commit()
            pending_delete_by_user.pop(line_user_id or "", None)
            expense = row_to_expense(row)
            return DeleteResult(
                deleted=True,
                expense=expense,
                expense_id=pending_id,
                similar_count=count_similar_expenses(expense, line_user_id),
            )

        if id_match and len(normalized_text) <= len(id_match.group(1)) + 3:
            expense_id = int(id_match.group(1))
            row = conn.execute(
                """
                SELECT * FROM expenses
                WHERE id = ?
                  AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
                """,
                (expense_id, line_user_id, line_user_id),
            ).fetchone()
            if not row:
                return DeleteResult(deleted=False, reason=f"找不到編號 {expense_id} 的記帳。")

            conn.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
            conn.commit()
            pending_delete_by_user.pop(line_user_id or "", None)
            expense = row_to_expense(row)
            return DeleteResult(
                deleted=True,
                expense=expense,
                expense_id=expense_id,
                similar_count=count_similar_expenses(expense, line_user_id),
            )

        if not id_match:
            return DeleteResult(deleted=False, reason=build_delete_candidates_reply(raw_text, line_user_id))

        try:
            target = parse_expense_text_fallback(normalized_text)
        except ValueError:
            return DeleteResult(deleted=False, reason="刪除指令需要金額，例如「刪除今天的午餐 120」。")

        row = conn.execute(
            """
            SELECT * FROM expenses
            WHERE date = ?
              AND amount = ?
              AND currency = ?
              AND category = ?
              AND raw_text NOT LIKE '刪除%'
              AND raw_text NOT LIKE '删除%'
              AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
            ORDER BY id DESC
            LIMIT 1
            """,
            (target.date, target.amount, target.currency, target.category, line_user_id, line_user_id),
        ).fetchone()

        if not row:
            return DeleteResult(deleted=False, reason=f"找不到 {target.date} {target.category} {target.amount} 元的記帳。")

        expense_id = int(row["id"])
        conn.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
        conn.commit()
        pending_delete_by_user.pop(line_user_id or "", None)
        expense = row_to_expense(row)
        return DeleteResult(
            deleted=True,
            expense=expense,
            expense_id=expense_id,
            similar_count=count_similar_expenses(expense, line_user_id),
        )


def build_delete_reply(result: DeleteResult) -> str:
    if not result.deleted or not result.expense or not result.expense_id:
        return result.reason or "找不到可刪除的記帳。"

    expense = result.expense
    note = f"\n備註：{expense.note}" if expense.note else ""
    time = f"時間：{expense.time}\n" if expense.time else ""
    similar = (
        f"\n提醒：還有 {result.similar_count} 筆同日同分類同金額的相似花費。"
        if result.similar_count
        else ""
    )
    return (
        "已刪除記帳\n"
        f"日期：{expense.date}\n"
        f"{time}"
        f"金額：{expense.currency} {expense.amount}\n"
        f"分類：{expense.category}"
        f"{note}"
        f"{similar}"
    )


def build_update_reply(_: str) -> str:
    return "目前第一版還沒實作修改功能。可以先刪掉那筆，再重新新增正確資料。"


def chat_with_openai(text: str) -> str:
    log_usage("openai", "chat_completion", detail=get_current_model())
    response = get_openai_client().chat.completions.create(
        model=get_current_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "你是 LINE 記帳 Bot 的助理。"
                    "用繁體中文簡短回答。"
                    "不要自行新增、修改、刪除或查詢資料庫；只做一般說明或聊天。"
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content or "我在，請輸入像「午餐 120」這樣的記帳內容。"


def build_chat_reply(text: str) -> str:
    help_keywords = ("help", "說明", "怎麼用", "如何使用")
    if any(keyword in text.lower() for keyword in help_keywords):
        return (
            "可以這樣使用：\n"
            "新增：午餐 120\n"
            "查詢：今天總共花多少錢？\n"
            "明細：列出這兩天的花費\n"
            "刪除：刪除昨天的消夜\n"
            "待繳：信用卡 12000\n"
            "已繳：已繳 信用卡\n"
            "收入：薪資收入 50000\n"
            "修改：目前請先刪除再新增"
        )

    try:
        return chat_with_openai(text)
    except Exception:
        logger.exception("OpenAI chat failed for text: %s", text)
        return "我收到訊息了。若要記帳，請輸入像「午餐 120」；若要查詢，請輸入「今天總共花多少錢？」。"


def save_expense(
    expense: ExpenseEntry,
    raw_text: str,
    line_user_id: str | None,
    message_id: str | None,
) -> int:
    with get_db() as conn:
        if message_id:
            existing = conn.execute(
                "SELECT id FROM expenses WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if existing:
                return int(existing["id"])

        existing = conn.execute(
            """
            SELECT id FROM expenses
            WHERE date = ?
              AND COALESCE(expense_time, '') = COALESCE(?, '')
              AND amount = ?
              AND currency = ?
              AND category = ?
              AND COALESCE(note, '') = COALESCE(?, '')
              AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                expense.date,
                expense.time,
                expense.amount,
                expense.currency,
                expense.category,
                expense.note,
                line_user_id,
                line_user_id,
            ),
        ).fetchone()
        if existing:
            return int(existing["id"])

        cursor = conn.execute(
            """
            INSERT INTO expenses (
                line_user_id,
                message_id,
                raw_text,
                date,
                expense_time,
                amount,
                currency,
                category,
                merchant,
                note,
                confidence,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                line_user_id,
                message_id,
                raw_text,
                expense.date,
                expense.time,
                expense.amount,
                expense.currency,
                expense.category,
                expense.merchant,
                expense.note,
                expense.confidence,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def build_reply_text(expense: ExpenseEntry, expense_id: int) -> str:
    merchant = expense.merchant or "未指定"
    note = f"\n備註：{expense.note}" if expense.note else ""
    time = f"時間：{expense.time}\n" if expense.time else ""
    return (
        "已記帳\n"
        f"日期：{expense.date}\n"
        f"{time}"
        f"金額：{expense.currency} {expense.amount}\n"
        f"分類：{expense.category}\n"
        f"店家/對象：{merchant}\n"
        f"信心：{expense.confidence:.2f}"
        f"{note}"
    )


def get_pending_expense_key(line_user_id: str | None) -> str:
    return line_user_id or "_global"


def has_explicit_accounting_verb(raw_text: str) -> bool:
    verbs = (
        "\u8a18\u5e33",
        "\u5e6b\u6211\u8a18",
        "\u82b1\u4e86",
        "\u4ed8\u6b3e",
        "\u652f\u4ed8",
        "\u5237\u5361",
        "\u8f49\u5e33",
        "\u8cb7\u4e86",
    )
    return any(verb in raw_text for verb in verbs)


def has_planning_language(raw_text: str) -> bool:
    keywords = (
        "\u60f3\u8cb7",
        "\u6253\u7b97",
        "\u9810\u8a08",
        "\u5982\u679c",
        "\u5047\u8a2d",
        "\u8cb7\u8eca",
        "\u8cb7\u623f",
    )
    return any(keyword in raw_text for keyword in keywords)


def is_simple_small_expense(expense: ExpenseEntry, raw_text: str) -> bool:
    if expense.amount >= 10000 or has_planning_language(raw_text):
        return False
    if str(expense.category) == "\u5176\u4ed6":
        return False
    return parse_amount(raw_text) is not None and len(raw_text.strip()) <= 30


def should_confirm_expense(expense: ExpenseEntry, raw_text: str, route: ActionRoute) -> bool:
    if expense.amount >= 10000:
        return True
    if str(expense.category) == "\u5176\u4ed6":
        return True
    if route.confidence < 0.85:
        return True
    if has_planning_language(raw_text):
        return True
    if not has_explicit_accounting_verb(raw_text) and not is_simple_small_expense(expense, raw_text):
        return True
    return False


def build_pending_expense_reply(expense: ExpenseEntry) -> str:
    note = expense.note or "\u672a\u586b"
    return (
        "\u9019\u7b46\u770b\u8d77\u4f86\u50cf\u5927\u984d\u6216\u898f\u5283\u6027\u652f\u51fa\uff0c\u6211\u5148\u4e0d\u76f4\u63a5\u5165\u5e33\u3002\n"
        f"\u65e5\u671f\uff1a{expense.date}\n"
        f"\u91d1\u984d\uff1a{expense.currency} {expense.amount}\n"
        f"\u5206\u985e\uff1a{expense.category}\n"
        f"\u5099\u8a3b\uff1a{note}\n"
        "\u5982\u679c\u8981\u8a18\u5e33\uff0c\u8acb\u56de\u8986\uff1a\u78ba\u8a8d\u8a18\u5e33\n"
        "\u5982\u679c\u4e0d\u8981\uff0c\u8acb\u56de\u8986\uff1a\u53d6\u6d88"
    )


async def reply_line_messages(reply_token: str, messages: list[dict]) -> None:
    start = datetime.now(timezone.utc)
    load_dotenv(override=True)
    channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
    if not channel_access_token:
        raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN is not configured.")

    headers = {
        "Authorization": f"Bearer {channel_access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "replyToken": reply_token,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=10, verify=get_ssl_verify()) as client:
            response = await client.post(LINE_REPLY_URL, headers=headers, json=payload)
            response.raise_for_status()
        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        log_usage("line", "reply", success=True, latency_ms=latency_ms)
        if any(message.get("type") == "image" for message in messages):
            log_usage("line", "image", success=True, latency_ms=latency_ms)
    except Exception:
        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        log_usage("line", "reply", success=False, latency_ms=latency_ms)
        raise


async def reply_line_message(reply_token: str, text: str) -> None:
    await reply_line_messages(reply_token, [{"type": "text", "text": text}])


def reply_payload_to_messages(reply_payload: object) -> list[dict]:
    if isinstance(reply_payload, dict):
        text = str(reply_payload.get("text") or "")
        image_url = reply_payload.get("image_url")
        messages = [{"type": "text", "text": text}]
        if isinstance(image_url, str) and image_url:
            messages.append(
                {
                    "type": "image",
                    "originalContentUrl": image_url,
                    "previewImageUrl": image_url,
                }
            )
        return messages
    return [{"type": "text", "text": str(reply_payload)}]


async def reply_payload_with_token(reply_token: str, reply_payload: object) -> None:
    await reply_line_messages(reply_token, reply_payload_to_messages(reply_payload))


async def push_reply_payload(line_user_id: str, reply_payload: object) -> None:
    await push_line_messages(line_user_id, reply_payload_to_messages(reply_payload))


def build_interim_reply(status: str) -> str:
    if status == "\u7b49\u5f85 ChatGPT \u56de\u61c9\u4e2d":
        return "\u62b1\u6b49\uff0c\u9019\u6b21\u6bd4\u8f03\u4e45\uff0c\u6211\u6b63\u5728\u7b49\u5f85 ChatGPT \u56de\u61c9\uff0c\u7b49\u6211\u4e00\u4e0b\u3002"
    if status == "\u7522\u751f\u5716\u8868\u4e2d":
        return "\u6211\u6b63\u5728\u6574\u7406\u8cc7\u6599\u4e26\u88fd\u4f5c\u5716\u8868\uff0c\u7b49\u6211\u4e00\u4e0b\u3002"
    if status == "\u67e5\u8a62\u8cc7\u6599\u5eab\u4e2d":
        return "\u6211\u6b63\u5728\u67e5\u8cc7\u6599\u5eab\uff0c\u7b49\u6211\u4e00\u4e0b\u3002"
    if status == "\u8a08\u7b97\u6536\u652f\u4e2d":
        return "\u6211\u6b63\u5728\u8a08\u7b97\u9019\u500b\u6708\u7684\u6536\u652f\uff0c\u7b49\u6211\u4e00\u4e0b\u3002"
    if status == "\u5efa\u7acb\u63d0\u9192\u4e2d":
        return "\u6211\u6b63\u5728\u5efa\u7acb\u5f85\u7e73\u63d0\u9192\uff0c\u7b49\u6211\u4e00\u4e0b\u3002"
    return f"\u62b1\u6b49\uff0c\u9019\u6b21\u6bd4\u8f03\u4e45\uff0c\u6211\u6b63\u5728{status}\uff0c\u7b49\u6211\u4e00\u4e0b\u3002"


def setting_user_key(line_user_id: str | None) -> str:
    return line_user_id or "_global"


def setting_external_to_internal(name: str) -> str:
    normalized = name.strip().upper()
    if normalized == "REMINDER":
        return "REMINDER_INTERVAL_HOURS"
    if normalized == "TIMEOUT":
        return "PROCESSING_TIMEOUT_SECONDS"
    return normalized


def setting_display_name(key: str) -> str:
    if key == "REMINDER_INTERVAL_HOURS":
        return "REMINDER"
    if key == "PROCESSING_TIMEOUT_SECONDS":
        return "TIMEOUT"
    return key


def read_settings_reply() -> str:
    lines = [
        "\u76ee\u524d\u8a2d\u5b9a\uff1a",
        f"OPENAI_MODEL = {get_current_model()}",
        f"DATABASE_PATH = {get_db_path()}",
        f"SSL_VERIFY = {get_ssl_verify_setting()}",
        f"REMINDER = {get_reminder_interval_hours()} \u5c0f\u6642",
        f"TIMEOUT = {get_processing_timeout_seconds()} \u79d2",
        f"PUBLIC_BASE_URL = {get_public_base_url() or ''}",
        f"APP_ENV = {get_app_env()}",
    ]
    return "\n".join(lines)


def write_setting_help_reply() -> str:
    return (
        "\u4f60\u60f3\u4fee\u6539\u54ea\u500b\u53c3\u6578\uff1f\n\n"
        "\u53ef\u4fee\u6539\uff1a\n"
        "1. OPENAI_MODEL\n"
        "2. SSL_VERIFY\n"
        "3. REMINDER\n"
        "4. TIMEOUT\n"
        "5. PUBLIC_BASE_URL\n\n"
        "\u8acb\u56de\u8986\uff0c\u4f8b\u5982\uff1a\n"
        "\u8a2d\u5b9a OPENAI_MODEL gpt-4.1-mini\n"
        "\u8a2d\u5b9a SSL_VERIFY false\n"
        "\u8a2d\u5b9a REMINDER 12\n"
        "\u8a2d\u5b9a TIMEOUT 3\n"
        "\u8a2d\u5b9a PUBLIC_BASE_URL https://xxxx.ngrok-free.app"
    )


def validate_setting_value(key: str, value: str) -> tuple[bool, str, str]:
    if key == "DATABASE_PATH":
        return False, value, "\u0054DATABASE_PATH \u70ba\u9ad8\u98a8\u96aa\u8a2d\u5b9a\uff0c\u7b2c\u4e00\u7248\u4e0d\u5141\u8a31\u5f9e LINE \u4fee\u6539\uff0c\u8acb\u624b\u52d5\u4fee\u6539 .env \u5f8c\u91cd\u555f\u670d\u52d9\u3002".lstrip("T")
    if key == "OPENAI_MODEL":
        allowed = {"gpt-4.1-mini", "gpt-4.1", "gpt-5.4-nano", "gpt-5.4-mini"}
        return (value in allowed), value, "\u4e0d\u652f\u63f4\u7684 OPENAI_MODEL\u3002"
    if key == "SSL_VERIFY":
        normalized = value.lower()
        return (normalized in {"true", "false"}), normalized, "SSL_VERIFY \u53ea\u80fd\u662f true \u6216 false\u3002"
    if key in {"REMINDER_INTERVAL_HOURS", "PROCESSING_TIMEOUT_SECONDS"}:
        try:
            number = int(value)
        except ValueError:
            return False, value, f"{setting_display_name(key)} \u5fc5\u9808\u662f\u6578\u5b57\u3002"
        min_value, max_value = (1, 168) if key == "REMINDER_INTERVAL_HOURS" else (1, 30)
        if not min_value <= number <= max_value:
            return False, value, f"{setting_display_name(key)} \u5fc5\u9808\u5728 {min_value} \u5230 {max_value} \u4e4b\u9593\u3002"
        return True, str(number), ""
    if key == "PUBLIC_BASE_URL":
        normalized = value.rstrip("/")
        if not normalized.startswith("https://"):
            return False, normalized, "PUBLIC_BASE_URL \u5fc5\u9808\u4ee5 https:// \u958b\u982d\u3002"
        if "/line/webhook" in normalized:
            return False, normalized, "PUBLIC_BASE_URL \u4e0d\u53ef\u4ee5\u5305\u542b /line/webhook\u3002"
        return True, normalized, ""
    return False, value, "\u9019\u500b\u53c3\u6578\u4e0d\u5728\u53ef\u4fee\u6539\u767d\u540d\u55ae\u5167\u3002"


def build_setting_confirmation_reply(key: str, value: str, line_user_id: str | None) -> str:
    ok, normalized_value, error = validate_setting_value(key, value)
    if not ok:
        return error
    current = get_effective_setting(key, key, "")
    pending_setting_changes_by_user[setting_user_key(line_user_id)] = {"key": key, "value": normalized_value}
    display = setting_display_name(key)
    return (
        "\u5373\u5c07\u4fee\u6539\uff1a\n"
        f"{display}\n"
        f"\u76ee\u524d\u503c\uff1a{current}\n"
        f"\u65b0\u503c\uff1a{normalized_value}\n\n"
        "\u82e5\u78ba\u5b9a\uff0c\u8acb\u56de\u8986\uff1a\n"
        f"\u78ba\u8a8d\u8a2d\u5b9a {display} {normalized_value}"
    )


def parse_setting_command(text: str) -> tuple[str, str] | None:
    match = re.match(r"^\s*\u8a2d\u5b9a\s+([A-Za-z_]+)\s+(.+?)\s*$", text)
    if not match:
        return None
    return setting_external_to_internal(match.group(1)), match.group(2).strip()


def parse_confirm_setting_command(text: str) -> tuple[str, str] | None:
    match = re.match(r"^\s*\u78ba\u8a8d\u8a2d\u5b9a\s+([A-Za-z_]+)\s+(.+?)\s*$", text)
    if not match:
        return None
    return setting_external_to_internal(match.group(1)), match.group(2).strip()


def usage_reply(raw_text: str) -> str:
    normalized = raw_text.lower().strip()
    today = date.today()
    if "7d" in normalized:
        start_date = today - timedelta(days=6)
    elif "month" in normalized or "\u6708" in raw_text:
        start_date = today.replace(day=1)
    else:
        start_date = today
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT provider, event_type, success, latency_ms, COUNT(*) AS count
            FROM usage_logs
            WHERE date(created_at) BETWEEN ? AND ?
            GROUP BY provider, event_type, success, latency_ms IS NULL
            """,
            (start_date.isoformat(), today.isoformat()),
        ).fetchall()
        aggregate_rows = conn.execute(
            """
            SELECT provider, event_type,
                   COUNT(*) AS count,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS success_count,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failure_count,
                   AVG(latency_ms) AS avg_latency_ms
            FROM usage_logs
            WHERE date(created_at) BETWEEN ? AND ?
            GROUP BY provider, event_type
            """,
            (start_date.isoformat(), today.isoformat()),
        ).fetchall()
    stats = {(row["provider"], row["event_type"]): row for row in aggregate_rows}

    def stat(provider: str, event_type: str) -> tuple[int, int, int, int]:
        row = stats.get((provider, event_type))
        if not row:
            return 0, 0, 0, 0
        return (
            int(row["count"]),
            int(row["success_count"] or 0),
            int(row["failure_count"] or 0),
            int(row["avg_latency_ms"] or 0),
        )

    openai_count, openai_success, openai_failure, openai_avg = stat("openai", "chat_completion")
    webhook_count, _, _, _ = stat("line", "webhook")
    reply_count, _, _, _ = stat("line", "reply")
    push_count, _, _, _ = stat("line", "push")
    image_count, _, _, _ = stat("line", "image")
    chart_stats = chart_storage_stats()
    reminder_push_count = 0
    with get_db() as conn:
        reminder_push_count = int(
            conn.execute(
                "SELECT COUNT(*) AS count FROM payable_reminders WHERE reminded_on BETWEEN ? AND ?",
                (start_date.isoformat(), today.isoformat()),
            ).fetchone()["count"]
        )
    return (
        "\u4eca\u65e5\u7528\u91cf\uff1a\n"
        "ChatGPT\uff1a\n"
        f"- \u547c\u53eb\u6b21\u6578\uff1a{openai_count}\n"
        f"- \u6210\u529f\uff1a{openai_success}\n"
        f"- \u5931\u6557\uff1a{openai_failure}\n"
        f"- \u5e73\u5747\u8017\u6642\uff1a{openai_avg} ms\n\n"
        "LINE\uff1a\n"
        f"- webhook \u6b21\u6578\uff1a{webhook_count}\n"
        f"- reply \u6b21\u6578\uff1a{reply_count}\n"
        f"- push \u6b21\u6578\uff1a{push_count}\n"
        f"- image \u6b21\u6578\uff1a{image_count}\n\n"
        "ngrok\uff1a\n"
        f"- PUBLIC_BASE_URL\uff1a{get_public_base_url() or ''}\n"
        "- \u672c\u6a5f\u7121\u6cd5\u76f4\u63a5\u67e5\u5b98\u65b9\u6d41\u91cf\uff0c\u9664\u975e\u672a\u4f86\u4e32 ngrok API\n\n"
        "\u5716\u8868\u5feb\u53d6\uff1a\n"
        f"- \u5716\u7247\u6578\u91cf\uff1a{int(chart_stats['count'])}\n"
        f"- \u5360\u7528\u7a7a\u9593\uff1a{float(chart_stats['size_mb']):.2f} MB\n"
        f"- \u4fdd\u7559\u5929\u6578\uff1a{get_chart_retention_days()} \u5929\n\n"
        "\u63d0\u9192\uff1a\n"
        f"- \u4eca\u65e5\u63d0\u9192 push \u6b21\u6578\uff1a{reminder_push_count}"
    )


def clean_charts_reply(raw_text: str, line_user_id: str | None) -> str:
    normalized = re.sub(r"\s+", "", raw_text).lower()
    user_key = setting_user_key(line_user_id)
    if normalized in {"cleanchartsall", "\u6e05\u7406\u5716\u8868all", "\u6e05\u7a7a\u5716\u8868"}:
        pending_chart_cleanup_by_user.add(user_key)
        return "\u9019\u6703\u522a\u9664\u6240\u6709\u5716\u8868\u5feb\u53d6\u3002\u82e5\u78ba\u5b9a\uff0c\u8acb\u56de\u8986\uff1a\u78ba\u8a8d\u6e05\u7a7a\u5716\u8868"
    if normalized == "\u78ba\u8a8d\u6e05\u7a7a\u5716\u8868":
        if user_key not in pending_chart_cleanup_by_user:
            return "\u76ee\u524d\u6c92\u6709\u5f85\u78ba\u8a8d\u7684\u5716\u8868\u6e05\u7a7a\u6307\u4ee4\u3002"
        pending_chart_cleanup_by_user.discard(user_key)
        deleted_count = cleanup_old_charts(days=0)
        return f"\u5df2\u6e05\u7a7a\u5716\u8868\u5feb\u53d6\uff0c\u522a\u9664 {deleted_count} \u5f35\u5716\u7247\u3002"
    deleted_count = cleanup_old_charts(get_chart_retention_days())
    return f"\u5df2\u6e05\u7406\u8d85\u904e {get_chart_retention_days()} \u5929\u7684\u5716\u8868\u5feb\u53d6\uff0c\u522a\u9664 {deleted_count} \u5f35\u5716\u7247\u3002"


def handle_special_command(raw_text: str, line_user_id: str | None) -> str | None:
    normalized = re.sub(r"\s+", "", raw_text)
    if raw_text.strip().lower() in {"readsetting"} or normalized == "\u8b80\u8a2d\u5b9a":
        return read_settings_reply()
    if raw_text.strip().lower() in {"writesetting"} or normalized == "\u6539\u8a2d\u5b9a":
        return write_setting_help_reply()
    if raw_text.strip().lower().startswith("usage") or normalized == "\u7528\u91cf":
        return usage_reply(raw_text)
    if raw_text.strip().lower().startswith("cleancharts") or normalized in {"\u6e05\u7406\u5716\u8868", "\u6e05\u7a7a\u5716\u8868", "\u78ba\u8a8d\u6e05\u7a7a\u5716\u8868"}:
        return clean_charts_reply(raw_text, line_user_id)
    confirm = parse_confirm_setting_command(raw_text)
    if confirm:
        key, value = confirm
        pending = pending_setting_changes_by_user.get(setting_user_key(line_user_id))
        ok, normalized_value, error = validate_setting_value(key, value)
        if not ok:
            return error
        if not pending or pending.get("key") != key or pending.get("value") != normalized_value:
            return "\u76ee\u524d\u6c92\u6709\u5c0d\u61c9\u7684\u5f85\u78ba\u8a8d\u8a2d\u5b9a\uff0c\u8acb\u5148\u8f38\u5165\uff1a\u8a2d\u5b9a \u53c3\u6578 \u65b0\u503c"
        set_setting(key, normalized_value)
        pending_setting_changes_by_user.pop(setting_user_key(line_user_id), None)
        return f"\u5df2\u66f4\u65b0\u8a2d\u5b9a\uff1a{setting_display_name(key)} = {normalized_value}"
    command = parse_setting_command(raw_text)
    if command:
        key, value = command
        return build_setting_confirmation_reply(key, value, line_user_id)
    return None


def create_expense_reply(raw_text: str, line_user_id: str | None, message_id: str | None) -> str:
    try:
        expense = parse_expense_text(raw_text)
    except OpenAIError:
        logger.exception("OpenAI failed; using fallback parser for LINE text: %s", raw_text)
        expense = parse_expense_text_fallback(raw_text)
    expense_id = save_expense(expense, raw_text, line_user_id, message_id)
    return build_reply_text(expense, expense_id)


def create_expense_with_confirmation_reply(
    raw_text: str,
    line_user_id: str | None,
    message_id: str | None,
    route: ActionRoute,
) -> str:
    try:
        expense = parse_expense_text(raw_text)
    except OpenAIError:
        logger.exception("OpenAI failed; using fallback parser for LINE text: %s", raw_text)
        expense = parse_expense_text_fallback(raw_text)

    parsed_amount = parse_amount(raw_text)
    if parsed_amount is not None and parsed_amount > expense.amount:
        expense.amount = parsed_amount

    if should_confirm_expense(expense, raw_text, route):
        pending_expense_by_user[get_pending_expense_key(line_user_id)] = {
            "expense": expense,
            "raw_text": raw_text,
            "line_user_id": line_user_id,
            "message_id": message_id,
        }
        return build_pending_expense_reply(expense)

    expense_id = save_expense(expense, raw_text, line_user_id, message_id)
    return build_reply_text(expense, expense_id)


def create_multiline_expense_reply(raw_text: str, line_user_id: str | None, message_id: str | None) -> str | None:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    is_multiline_create = (
        len(lines) > 1
        and all(re.search(r"\d+", line) for line in lines)
        and not any(is_question_text(line) for line in lines)
    )
    if not is_multiline_create:
        return None

    created_entries = []
    for index, line in enumerate(lines, start=1):
        try:
            expense = parse_expense_text(line)
        except OpenAIError:
            logger.exception("OpenAI failed; using fallback parser for LINE text: %s", line)
            expense = parse_expense_text_fallback(line)
        save_expense(expense, line, line_user_id, f"{message_id}:{index}" if message_id else None)
        time_text = f" {expense.time}" if expense.time else ""
        created_entries.append(f"{expense.date}{time_text} {expense.note or expense.category} TWD {expense.amount}")
    return "已記帳多筆\n" + "\n".join(created_entries)




def income_type_aliases(income_type: str | None) -> list[str]:
    if not income_type:
        return []
    normalized = get_income_type(income_type) or income_type
    if normalized == "\u85aa\u8cc7\u6536\u5165":
        return ["\u85aa\u8cc7\u6536\u5165", "\u85aa\u6c34", "\u85aa\u8cc7"]
    if normalized == "\u653f\u5e9c\u88dc\u52a9":
        return ["\u653f\u5e9c\u88dc\u52a9", "\u88dc\u52a9"]
    return [normalized]


def build_income_where_clause(
    text: str,
    line_user_id: str | None,
    income_type: str | None,
) -> tuple[str, list[object], date, date, str | None]:
    start_date, end_date = get_month_range(text)
    aliases = income_type_aliases(income_type or get_income_type(text))
    where_parts = [
        "income_date BETWEEN ? AND ?",
        "amount > 0",
        "(? IS NULL OR line_user_id = ? OR line_user_id IS NULL)",
    ]
    params: list[object] = [start_date.isoformat(), end_date.isoformat(), line_user_id, line_user_id]
    if aliases:
        placeholders = ", ".join("?" for _ in aliases)
        where_parts.append(
            f"(income_type IN ({placeholders}) OR item_name IN ({placeholders}) OR category IN ({placeholders}) OR raw_text LIKE ?)"
        )
        params.extend(aliases)
        params.extend(aliases)
        params.extend(aliases)
        params.append(f"%{aliases[0]}%")
    resolved_type = aliases[0] if aliases else None
    return " AND ".join(where_parts), params, start_date, end_date, resolved_type


def build_income_query_reply(text: str, line_user_id: str | None, income_type: str | None = None) -> str:
    where_sql, params, start_date, _end_date, resolved_type = build_income_where_clause(text, line_user_id, income_type)
    with get_db() as conn:
        row = conn.execute(
            f"""
            SELECT COALESCE(SUM(amount), 0) AS total, COUNT(*) AS count
            FROM incomes
            WHERE {where_sql}
            """,
            params,
        ).fetchone()
    month_label = f"{start_date.month}\u6708"
    title_type = resolved_type or "\u6536\u5165"
    return (
        f"{month_label}{title_type}\n"
        f"\u7e3d\u6536\u5165\uff1aTWD {int(row['total'] if row else 0)}\n"
        f"\u7b46\u6578\uff1a{int(row['count'] if row else 0)}"
    )


def build_income_list_reply(text: str, line_user_id: str | None, income_type: str | None = None) -> str:
    where_sql, params, start_date, _end_date, resolved_type = build_income_where_clause(text, line_user_id, income_type)
    with get_db() as conn:
        rows = conn.execute(
            f"""
            SELECT income_date, amount, currency, income_type, item_name, owner, note
            FROM incomes
            WHERE {where_sql}
            ORDER BY income_date ASC, id ASC
            """,
            params,
        ).fetchall()

    total = sum(int(row["amount"]) for row in rows)
    month_label = f"{start_date.month}\u6708"
    title_type = resolved_type or "\u6536\u5165"
    lines = [f"{month_label}{title_type}\u6e05\u55ae"]
    if rows:
        for row in rows:
            label = row["item_name"] or row["income_type"]
            owner_text = f" {row['owner']}" if row["owner"] else ""
            note_text = f" {row['note']}" if row["note"] and row["note"] != label else ""
            lines.append(f"{row['income_date']} {label}{owner_text} {row['currency']} {int(row['amount'])}{note_text}")
    else:
        lines.append("\u76ee\u524d\u6c92\u6709\u6536\u5165\u8a18\u9304\u3002")
    lines.append(f"\u5c0f\u8a08\uff1aTWD {total}\uff0c{len(rows)} \u7b46")
    return "\n".join(lines)


def get_month_finance(raw_text: str, line_user_id: str | None) -> dict[str, object]:
    start_date, end_date = get_month_range(raw_text)
    with get_db() as conn:
        income_rows = conn.execute(
            """
            SELECT id, raw_text, income_date, amount, income_type, item_name, owner, category
            FROM incomes
            WHERE income_date BETWEEN ? AND ?
              AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
            ORDER BY income_date ASC, id ASC
            """,
            (start_date.isoformat(), end_date.isoformat(), line_user_id, line_user_id),
        ).fetchall()
        payable_row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS total
            FROM payables
            WHERE due_date BETWEEN ? AND ?
              AND status = 'unpaid'
              AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
            """,
            (start_date.isoformat(), end_date.isoformat(), line_user_id, line_user_id),
        ).fetchone()
        payable_rows = conn.execute(
            """
            SELECT due_date, item_type, amount, owner, bank
            FROM payables
            WHERE due_date BETWEEN ? AND ?
              AND status = 'unpaid'
              AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
            ORDER BY due_date ASC, id ASC
            LIMIT 12
            """,
            (start_date.isoformat(), end_date.isoformat(), line_user_id, line_user_id),
        ).fetchall()

    actual_rows = get_actual_spending_rows(start_date.isoformat(), end_date.isoformat(), line_user_id)
    expense_rows = sort_actual_spending_rows(
        actual_rows,
        ExpenseQuery(
            date_range_type="custom",
            date_ranges=[DateRange(start_date=start_date.isoformat(), end_date=end_date.isoformat())],
            sort_by="date",
            sort_direction="asc",
            confidence=1,
        ),
    )[:12]
    unique_incomes: dict[tuple[str, str, int, str, str, str], dict[str, object]] = {}
    duplicate_counts: dict[tuple[str, str, int, str, str, str], int] = {}
    raw_income_total = 0
    for row in income_rows:
        effective_amount = parse_amount(str(row["raw_text"] or "")) or int(row["amount"])
        item_name = str(row["item_name"] or resolve_income_item_name(str(row["raw_text"] or ""), str(row["income_type"])))
        owner = str(row["owner"] or resolve_income_owner(str(row["raw_text"] or "")) or "")
        category = str(row["category"] or row["income_type"])
        key = (str(row["income_date"]), item_name, effective_amount, owner, category, str(row["income_type"]))
        duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
        raw_income_total += int(row["amount"])
        if key not in unique_incomes:
            unique_incomes[key] = {"amount": effective_amount, "row": row, "item_name": item_name, "owner": owner, "category": category}

    income_total = sum(int(value["amount"]) for value in unique_incomes.values())
    expense_total = sum(int(row["amount"]) for row in actual_rows)
    unpaid_total = int(payable_row["total"])
    return {
        "start_date": start_date,
        "end_date": end_date,
        "income_total": income_total,
        "raw_income_total": raw_income_total,
        "expense_total": expense_total,
        "unpaid_total": unpaid_total,
        "available_cash": income_total - expense_total - unpaid_total,
        "unique_incomes": unique_incomes,
        "duplicate_counts": duplicate_counts,
        "expense_rows": expense_rows,
        "payable_rows": payable_rows,
    }


def has_duplicate_income_data(finance: dict[str, object]) -> bool:
    duplicate_counts = finance["duplicate_counts"]
    raw_income_total = int(finance["raw_income_total"])
    income_total = int(finance["income_total"])
    return raw_income_total != income_total or any(count > 1 for count in duplicate_counts.values())  # type: ignore[union-attr]


def build_available_investment_cash_reply(raw_text: str, line_user_id: str | None) -> str:
    finance = get_month_finance(raw_text, line_user_id)
    total_income = int(finance["income_total"])
    total_expense = int(finance["expense_total"])
    total_unpaid = int(finance["unpaid_total"])
    available_cash = int(finance["available_cash"])

    lines = [
        f"\u4f60\u9019\u500b\u6708\u76ee\u524d\u5927\u7d04\u9084\u6709 {available_cash} \u5143\u53ef\u4ee5\u52d5\u7528\u3002",
        f"\u7b97\u6cd5\u662f\uff1a\u6536\u5165 {total_income} \u5143\uff0c\u6263\u6389\u5df2\u8a18\u9304\u652f\u51fa {total_expense} \u5143\uff0c\u518d\u6263\u6389\u672a\u7e73\u6b3e\u9805 {total_unpaid} \u5143\uff0c\u6240\u4ee5\u5269\u4e0b {available_cash} \u5143\u3002",
    ]
    if has_duplicate_income_data(finance):
        lines.append("\u6211\u6709\u770b\u5230\u85aa\u8cc7\u8cc7\u6599\u91cd\u8907\uff0c\u9019\u6b21\u5df2\u5148\u7528\u53bb\u91cd\u5f8c\u91d1\u984d\u8a08\u7b97\u3002")
    lines.append("\u5982\u679c\u4f60\u662f\u5728\u554f\u6295\u8cc7\u5efa\u8b70\uff0c\u6211\u6703\u5efa\u8b70\u5148\u4fdd\u7559\u751f\u6d3b\u7de9\u885d\uff0c\u518d\u8003\u616e\u53ef\u627f\u64d4\u7684\u6295\u5165\u91d1\u984d\u3002")

    if wants_balance_details(raw_text):
        unique_incomes = finance["unique_incomes"]
        duplicate_counts = finance["duplicate_counts"]
        expense_rows = finance["expense_rows"]
        payable_rows = finance["payable_rows"]
        lines.append("\n\u6536\u5165\u660e\u7d30\uff1a")
        if unique_incomes:
            for key, value in unique_incomes.items():  # type: ignore[union-attr]
                income_date, item_name, amount, owner, category, income_type = key
                owner_text = f" {owner}" if owner else ""
                count = duplicate_counts.get(key, 1)  # type: ignore[union-attr]
                duplicate_text = f"\uff08\u91cd\u8907 {count} \u7b46\uff0c\u53ea\u7b97\u4e00\u6b21\uff09" if count > 1 else ""
                lines.append(f"{income_date} {item_name}{owner_text} {amount} \u5143{duplicate_text}")
        else:
            lines.append("\u76ee\u524d\u6c92\u6709\u6536\u5165\u8a18\u9304\u3002")
        lines.append("\n\u652f\u51fa\u660e\u7d30\uff1a")
        if expense_rows:
            for row in expense_rows:  # type: ignore[union-attr]
                label = row["merchant"] or row["note"] or row["category"]
                time_text = f" {row['expense_time']}" if row["expense_time"] else ""
                lines.append(f"{row['date']}{time_text} {label} {int(row['amount'])} \u5143")
        else:
            lines.append("\u76ee\u524d\u6c92\u6709\u652f\u51fa\u8a18\u9304\u3002")
        lines.append("\n\u672a\u7e73\u6b3e\u9805\uff1a")
        if payable_rows:
            for row in payable_rows:  # type: ignore[union-attr]
                meta = " ".join(value for value in (row["owner"], row["bank"]) if value)
                label = f"{meta} {row['item_type']}".strip()
                lines.append(f"{format_month_day(row['due_date'])} {label} {int(row['amount'])} \u5143")
        else:
            lines.append("\u76ee\u524d\u6c92\u6709\u672a\u7e73\u6b3e\u9805\u3002")
    return "\n".join(lines)


def build_available_cash_reply(raw_text: str, line_user_id: str | None, route: ActionRoute | None = None) -> str:
    finance = get_month_finance(raw_text, line_user_id)
    total_income = int(finance["income_total"])
    total_expense = int(finance["expense_total"])
    total_unpaid = int(finance["unpaid_total"])
    available_cash = int(finance["available_cash"])
    lines = [
        f"\u4f60\u9019\u500b\u6708\u76ee\u524d\u5927\u7d04\u9084\u6709 {available_cash} \u5143\u53ef\u4ee5\u52d5\u7528\u3002",
        f"\u7b97\u6cd5\u662f\uff1a\u6536\u5165 {total_income} \u5143\uff0c\u6263\u6389\u5df2\u8a18\u9304\u652f\u51fa {total_expense} \u5143\uff0c\u518d\u6263\u6389\u672a\u7e73\u6b3e\u9805 {total_unpaid} \u5143\uff0c\u6240\u4ee5\u5269\u4e0b {available_cash} \u5143\u3002",
    ]
    if has_duplicate_income_data(finance):
        lines.append("\u6211\u6709\u770b\u5230\u85aa\u8cc7\u8cc7\u6599\u91cd\u8907\uff0c\u9019\u6b21\u5df2\u5148\u7528\u53bb\u91cd\u5f8c\u91d1\u984d\u8a08\u7b97\u3002")
    return "\n".join(lines)


def build_balance_reply(raw_text: str, line_user_id: str | None) -> str:
    start_date, end_date = get_month_range(raw_text)
    show_details = wants_balance_details(raw_text)
    with get_db() as conn:
        income_rows = conn.execute(
            """
            SELECT id, raw_text, income_date, amount, income_type
            FROM incomes
            WHERE income_date BETWEEN ? AND ?
              AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
            ORDER BY income_date ASC, id ASC
            """,
            (start_date.isoformat(), end_date.isoformat(), line_user_id, line_user_id),
        ).fetchall()
        payable_row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS total
            FROM payables
            WHERE due_date BETWEEN ? AND ?
              AND status = 'unpaid'
              AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
            """,
            (start_date.isoformat(), end_date.isoformat(), line_user_id, line_user_id),
        ).fetchone()
        payable_rows = conn.execute(
            """
            SELECT due_date, item_type, amount, owner, bank
            FROM payables
            WHERE due_date BETWEEN ? AND ?
              AND status = 'unpaid'
              AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
            ORDER BY due_date ASC, id ASC
            LIMIT 12
            """,
            (start_date.isoformat(), end_date.isoformat(), line_user_id, line_user_id),
        ).fetchall()

    actual_rows = get_actual_spending_rows(start_date.isoformat(), end_date.isoformat(), line_user_id)
    expense_rows = sort_actual_spending_rows(
        actual_rows,
        ExpenseQuery(
            date_range_type="custom",
            date_ranges=[DateRange(start_date=start_date.isoformat(), end_date=end_date.isoformat())],
            sort_by="date",
            sort_direction="asc",
            confidence=1,
        ),
    )[:12]
    unique_incomes: dict[tuple[str, str, str], dict[str, object]] = {}
    duplicate_counts: dict[tuple[str, str, str], int] = {}
    for row in income_rows:
        key = (str(row["income_date"]), str(row["income_type"]), str(row["raw_text"] or ""))
        duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
        effective_amount = parse_amount(str(row["raw_text"] or "")) or int(row["amount"])
        if key not in unique_incomes:
            unique_incomes[key] = {"amount": effective_amount, "row": row}
        else:
            unique_incomes[key]["amount"] = max(int(unique_incomes[key]["amount"]), effective_amount)

    income_total = sum(int(value["amount"]) for value in unique_incomes.values())
    raw_income_total = sum(int(row["amount"]) for row in income_rows)
    expense_total = sum(int(row["amount"]) for row in actual_rows)
    unpaid_total = int(payable_row["total"])
    needed_total = expense_total + unpaid_total
    balance = income_total - needed_total
    month_label = f"{start_date.month}\u6708"

    lines = [
        f"{month_label}\u9810\u4f30\u6536\u652f",
        f"\u6536\u5165\uff1a{income_total} \u5143",
        f"\u5df2\u8a18\u9304\u652f\u51fa\uff1a{expense_total} \u5143",
        f"\u672a\u7e73\u6b3e\u9805\uff1a{unpaid_total} \u5143",
    ]
    if raw_income_total != income_total or any(count > 1 for count in duplicate_counts.values()):
        lines.append(f"\u6211\u6709\u770b\u5230\u6536\u5165\u91cd\u8907\u6216\u820a\u91d1\u984d\u8cc7\u6599\uff0c\u6240\u4ee5\u9019\u6b21\u5148\u7528\u53bb\u91cd\u5f8c\u7684 {income_total} \u5143\u8a08\u7b97\u3002")
    if balance >= 0:
        lines.append(f"\u76ee\u524d\u6c92\u6709\u4e0d\u8db3\uff0c\u9810\u4f30\u9084\u6709 {balance} \u5143\u76c8\u9918\u3002")
    else:
        lines.append(f"\u76ee\u524d\u9810\u4f30\u9084\u4e0d\u8db3 {abs(balance)} \u5143\u3002")

    if show_details:
        lines.append("\n\u6536\u5165\u660e\u7d30\uff1a")
        if unique_incomes:
            for key, value in unique_incomes.items():
                income_date, income_type, source_text = key
                count = duplicate_counts.get(key, 1)
                duplicate_text = f"\uff08\u91cd\u8907 {count} \u7b46\uff0c\u8a08\u7b97\u6642\u53ea\u7b97\u4e00\u6b21\uff09" if count > 1 else ""
                lines.append(f"{income_date} {income_type} {int(value['amount'])} \u5143 {source_text}{duplicate_text}")
        else:
            lines.append("\u76ee\u524d\u6c92\u6709\u6536\u5165\u8a18\u9304\u3002")

        lines.append("\n\u652f\u51fa\u660e\u7d30\uff1a")
        if expense_rows:
            for row in expense_rows:
                label = row["merchant"] or row["note"] or row["category"]
                time_text = f" {row['expense_time']}" if row["expense_time"] else ""
                lines.append(f"{row['date']}{time_text} {label} {int(row['amount'])} \u5143")
        else:
            lines.append("\u76ee\u524d\u6c92\u6709\u652f\u51fa\u8a18\u9304\u3002")

        lines.append("\n\u672a\u7e73\u6b3e\u9805\uff1a")
        if payable_rows:
            for row in payable_rows:
                meta = " ".join(value for value in (row["owner"], row["bank"]) if value)
                label = f"{meta} {row['item_type']}".strip()
                lines.append(f"{format_month_day(row['due_date'])} {label} {int(row['amount'])} \u5143")
        else:
            lines.append("\u76ee\u524d\u6c92\u6709\u672a\u7e73\u6b3e\u9805\u3002")
    return "\n".join(lines)


def normalize_duplicate_label(*values: object) -> str:
    text = " ".join(str(value or "") for value in values)
    return re.sub(r"\s+", "", text).lower()


def find_duplicate_data_groups(line_user_id: str | None) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    with get_db() as conn:
        expense_rows = conn.execute(
            """
            SELECT id, date, expense_time, amount, currency, category, merchant, note, raw_text
            FROM expenses
            WHERE amount > 0
              AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
            ORDER BY date ASC, id ASC
            """,
            (line_user_id, line_user_id),
        ).fetchall()
        income_rows = conn.execute(
            """
            SELECT id, income_date, amount, income_type, item_name, owner, category, raw_text
            FROM incomes
            WHERE amount > 0
              AND (? IS NULL OR line_user_id = ? OR line_user_id IS NULL)
            ORDER BY income_date ASC, id ASC
            """,
            (line_user_id, line_user_id),
        ).fetchall()
        payable_rows = conn.execute(
            """
            SELECT id, due_date, item_type, amount, owner, bank, status
            FROM payables
            WHERE amount > 0
              AND (? IS NULL OR line_user_id = ?)
            ORDER BY due_date ASC, id ASC
            """,
            (line_user_id, line_user_id),
        ).fetchall()

    expense_map: dict[tuple[object, ...], list[sqlite3.Row]] = {}
    for row in expense_rows:
        label = normalize_duplicate_label(row["merchant"], row["note"], row["category"], row["raw_text"])
        key = (row["date"], row["expense_time"] or "", row["amount"], row["currency"], row["category"], label)
        expense_map.setdefault(key, []).append(row)
    for rows in expense_map.values():
        if len(rows) > 1:
            label = rows[0]["merchant"] or rows[0]["note"] or rows[0]["category"]
            groups.append({"table": "expenses", "label": label, "rows": rows, "delete_ids": [int(row["id"]) for row in rows[1:]]})

    income_map: dict[tuple[object, ...], list[sqlite3.Row]] = {}
    for row in income_rows:
        item_name = row["item_name"] or resolve_income_item_name(str(row["raw_text"] or ""), str(row["income_type"]))
        owner = row["owner"] or resolve_income_owner(str(row["raw_text"] or "")) or ""
        category = row["category"] or row["income_type"]
        amount = parse_amount(str(row["raw_text"] or "")) or int(row["amount"])
        key = (row["income_date"], item_name, amount, owner, category)
        income_map.setdefault(key, []).append(row)
    for rows in income_map.values():
        if len(rows) > 1:
            item_name = rows[0]["item_name"] or resolve_income_item_name(str(rows[0]["raw_text"] or ""), str(rows[0]["income_type"]))
            groups.append({"table": "incomes", "label": item_name, "rows": rows, "delete_ids": [int(row["id"]) for row in rows[1:]]})

    payable_map: dict[tuple[object, ...], list[sqlite3.Row]] = {}
    for row in payable_rows:
        key = (row["due_date"], row["item_type"], row["amount"], row["owner"] or "", row["bank"] or "", row["status"])
        payable_map.setdefault(key, []).append(row)
    for rows in payable_map.values():
        if len(rows) > 1:
            label = " ".join(str(value) for value in (rows[0]["owner"], rows[0]["bank"], rows[0]["item_type"]) if value)
            groups.append({"table": "payables", "label": label or rows[0]["item_type"], "rows": rows, "delete_ids": [int(row["id"]) for row in rows[1:]]})

    return groups


def format_duplicate_group(group: dict[str, object]) -> str:
    rows = group["rows"]
    if not isinstance(rows, list) or not rows:
        return ""
    first = rows[0]
    table = str(group["table"])
    label = str(group["label"])
    count = len(rows)
    if table == "expenses":
        return f"支出：{first['date']} {label} {int(first['amount'])} 元，重複 {count} 筆"
    if table == "incomes":
        return f"收入：{first['income_date']} {label} {int(parse_amount(str(first['raw_text'] or '')) or first['amount'])} 元，重複 {count} 筆"
    return f"繳費：{first['due_date']} {label} {int(first['amount'])} 元，重複 {count} 筆"


def build_duplicate_list_reply(line_user_id: str | None, store_pending: bool = False) -> str:
    groups = find_duplicate_data_groups(line_user_id)
    key = line_user_id or "_global"
    if not groups:
        pending_duplicate_delete_by_user.pop(key, None)
        return "目前沒有看到重複資料。"

    delete_targets: list[tuple[str, int]] = []
    lines = [f"我找到 {len(groups)} 組可能重複的資料："]
    for group in groups[:12]:
        line = format_duplicate_group(group)
        if line:
            lines.append(line)
        for expense_id in group["delete_ids"]:  # type: ignore[index]
            delete_targets.append((str(group["table"]), int(expense_id)))

    if store_pending and delete_targets:
        pending_duplicate_delete_by_user[key] = delete_targets
        lines.append(f"若要刪除重複資料，我會保留每組最早一筆，刪除其餘 {len(delete_targets)} 筆。")
        lines.append("請回覆：確認刪除重複資料")
    return "\n".join(lines)


def confirm_delete_duplicates(line_user_id: str | None) -> str:
    key = line_user_id or "_global"
    targets = pending_duplicate_delete_by_user.pop(key, None)
    if not targets:
        return "目前沒有等待確認刪除的重複資料。請先輸入「列出重複的資料」。"

    deleted = 0
    with get_db() as conn:
        for table, row_id in targets:
            if table not in {"expenses", "incomes", "payables"}:
                continue
            cursor = conn.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
            deleted += cursor.rowcount
        conn.commit()
    return f"已刪除重複資料 {deleted} 筆。我保留了每組最早記錄的一筆。"


def execute_action_route(
    route: ActionRoute,
    raw_text: str,
    line_user_id: str | None,
    message_id: str | None,
) -> str:
    logger.info(
        "Action routed as %s mutate=%s confidence=%.2f reason=%s text=%s",
        route.action,
        route.should_mutate_db,
        route.confidence,
        route.reason,
        raw_text,
    )

    normalized_text = re.sub(r"\s+", "", raw_text)
    pending_key = get_pending_expense_key(line_user_id)
    if is_confirm_delete_intent(raw_text) and pending_delete_by_user.get(line_user_id or ""):
        delete_result = delete_expense(raw_text, line_user_id)
        return build_delete_reply(delete_result)

    if is_confirm_delete_duplicates_text(raw_text):
        return confirm_delete_duplicates(line_user_id)

    if normalized_text == "\u78ba\u8a8d\u8a18\u5e33":
        pending = pending_expense_by_user.pop(pending_key, None)
        if not pending:
            return "\u76ee\u524d\u6c92\u6709\u7b49\u5f85\u78ba\u8a8d\u7684\u8a18\u5e33\u9805\u76ee\u3002"
        expense = pending["expense"]
        if not isinstance(expense, ExpenseEntry):
            return "\u9019\u7b46\u5f85\u78ba\u8a8d\u8cc7\u6599\u6709\u9ede\u554f\u984c\uff0c\u8acb\u91cd\u65b0\u8f38\u5165\u4e00\u6b21\u3002"
        expense_id = save_expense(
            expense,
            str(pending["raw_text"]),
            pending.get("line_user_id") if isinstance(pending.get("line_user_id"), str) else line_user_id,
            pending.get("message_id") if isinstance(pending.get("message_id"), str) else None,
        )
        return build_reply_text(expense, expense_id)

    if normalized_text == "\u53d6\u6d88":
        canceled_duplicate = pending_duplicate_delete_by_user.pop(pending_key, None)
        if pending_expense_by_user.pop(pending_key, None):
            return "\u597d\uff0c\u5df2\u53d6\u6d88\u9019\u7b46\u5f85\u78ba\u8a8d\u8a18\u5e33\u3002"
        if canceled_duplicate:
            return "好，已取消刪除重複資料。"
        return "\u76ee\u524d\u6c92\u6709\u7b49\u5f85\u78ba\u8a8d\u7684\u8a18\u5e33\u9805\u76ee\u3002"

    if route.should_mutate_db and is_question_text(raw_text):
        logger.warning("Blocked mutating action for question text: action=%s text=%s", route.action, raw_text)
        if route.action in {"mark_payable_paid", "create_payable"} or get_payable_type(raw_text):
            return build_payable_query_reply(raw_text, line_user_id, route)
        return build_chat_reply(raw_text)

    if route.action == "query_payables":
        return build_payable_query_reply(raw_text, line_user_id, route)

    if route.action == "list_duplicates":
        return build_duplicate_list_reply(line_user_id, store_pending=False)

    if route.action == "delete_duplicates":
        return build_duplicate_list_reply(line_user_id, store_pending=True)

    if route.action == "ask_available_investment_cash":
        return build_available_investment_cash_reply(raw_text, line_user_id)

    if route.action == "query_available_cash":
        return build_available_cash_reply(raw_text, line_user_id, route)

    if route.action == "query_incomes":
        return build_income_query_reply(raw_text, line_user_id, route.income_type)

    if route.action == "list_incomes":
        return build_income_list_reply(raw_text, line_user_id, route.income_type)

    if route.action == "query_balance":
        return build_balance_reply(raw_text, line_user_id)

    if route.action == "mark_payable_paid":
        if route.confidence < 0.65:
            return "我不太確定你要標記哪一筆已繳，請回覆例如：已繳 玉山信用卡。"
        reply = mark_payable_paid(raw_text, line_user_id)
        return reply or "找不到符合的未繳項目，請說明項目名稱，例如：已繳 玉山信用卡。"

    if route.action == "create_payable":
        reply = handle_payable_setup(raw_text, line_user_id, message_id)
        if reply:
            return reply
        if route.item_type and route.amount is not None and route.due_date and line_user_id:
            return create_payable(line_user_id, route.item_type, route.amount, route.due_date, raw_text, message_id)
        item_type = route.item_type or get_payable_type(raw_text) or "這筆繳費"
        amount_text = f" TWD {route.amount}" if route.amount is not None else ""
        if get_payable_type(raw_text) or route.item_type:
            return "\u6211\u60f3\u4f60\u662f\u8981\u5efa\u7acb\u5f85\u7e73\u63d0\u9192\uff0c\u4f46\u6211\u6c92\u89e3\u6790\u6210\u529f\u3002\u53ef\u4ee5\u8a66\u8a66\uff1a\u623f\u8cb8 62218 7/8"
        return (
            f"{item_type}{amount_text} 的繳費期限是哪一天？\n"
            "可以回覆例如：7/15、7月15日、15號、明天。"
        )

    if route.action == "create_income":
        reply = save_income(raw_text, line_user_id)
        return reply or build_chat_reply(raw_text)

    if route.action == "delete_expense":
        delete_result = delete_expense(raw_text, line_user_id)
        return build_delete_reply(delete_result)

    if route.action == "top_expense":
        return build_top_expense_reply(raw_text, line_user_id, route)

    if route.action == "list_expenses":
        return build_list_reply(raw_text, line_user_id, route)

    if route.action == "query_expenses":
        return build_expense_reply(raw_text, line_user_id, route)

    if route.action == "create_expense":
        multiline_reply = create_multiline_expense_reply(raw_text, line_user_id, message_id)
        if multiline_reply:
            return multiline_reply
        return create_expense_with_confirmation_reply(raw_text, line_user_id, message_id, route)

    return build_chat_reply(raw_text)


def normalize_in_flight_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def get_line_chat_id(source: dict) -> str | None:
    source_type = source.get("type")
    if source_type == "group":
        return source.get("groupId")
    if source_type == "room":
        return source.get("roomId")
    return source.get("userId")


def in_flight_key(chat_id: str | None, actor_user_id: str | None, raw_text: str) -> str:
    return f"{chat_id or '_chat'}:{actor_user_id or '_actor'}:{normalize_in_flight_text(raw_text)}"


def process_message_sync(ctx: ProcessingContext, event: dict) -> object:
    message = event.get("message", {})
    raw_text = message.get("text", "").strip()
    source = event.get("source", {})
    line_user_id = source.get("userId")
    message_id = message.get("id")
    ctx.status = "\u5206\u6790\u8a0a\u606f\u4e2d"
    special_reply = handle_special_command(raw_text, line_user_id)
    if special_reply is not None:
        ctx.status = "\u5b8c\u6210"
        return special_reply
    ctx.status = "\u7b49\u5f85 ChatGPT \u56de\u61c9\u4e2d"
    route = route_action(raw_text, line_user_id)
    if route.action in {"query_expenses", "list_expenses", "top_expense", "query_payables", "query_incomes", "list_incomes"}:
        ctx.status = "\u67e5\u8a62\u8cc7\u6599\u5eab\u4e2d"
    elif route.action in {"query_balance", "query_available_cash", "ask_available_investment_cash"}:
        ctx.status = "\u8a08\u7b97\u6536\u652f\u4e2d"
    elif route.action == "create_payable":
        ctx.status = "\u5efa\u7acb\u63d0\u9192\u4e2d"
    elif route.action in {"create_expense", "create_income", "mark_payable_paid", "delete_expense", "delete_duplicates"}:
        ctx.status = "\u5beb\u5165\u8cc7\u6599\u5eab\u4e2d"
    if route.action == "query_expenses" and any(keyword in raw_text for keyword in ("\u6298\u7dda\u5716", "\u9577\u689d\u5716", "\u5713\u9905\u5716", "\u5716\u8868")):
        ctx.status = "\u7522\u751f\u5716\u8868\u4e2d"
    reply_payload = execute_action_route(route, raw_text, line_user_id, message_id)
    ctx.status = "\u5b8c\u6210"
    return reply_payload


async def process_message_with_context(ctx: ProcessingContext, event: dict) -> object:
    return await asyncio.to_thread(process_message_sync, ctx, event)


async def finalize_background_message(ctx: ProcessingContext, task: asyncio.Task, key: str) -> None:
    try:
        reply_payload = await task
        ctx.status = "\u5b8c\u6210"
        ctx.final_replied = True
        if ctx.interim_replied and ctx.chat_id:
            await push_reply_payload(ctx.chat_id, reply_payload)
    except Exception:
        logger.exception("Background LINE message processing failed.")
        error_text = "\u525b\u525b\u8655\u7406\u5931\u6557\u4e86\uff0c\u6211\u6709\u8a18\u9304\u932f\u8aa4\uff0c\u8acb\u7a0d\u5f8c\u518d\u8a66\u3002"
        if ctx.interim_replied and ctx.chat_id:
            await push_line_message(ctx.chat_id, error_text)
    finally:
        current = in_flight_tasks.get(key)
        if current and current.get("task") is task:
            in_flight_tasks.pop(key, None)


async def handle_text_event(event: dict) -> None:
    reply_token = event.get("replyToken")
    message = event.get("message", {})
    raw_text = message.get("text", "").strip()

    if not reply_token or not raw_text:
        return

    source = event.get("source", {})
    actor_user_id = source.get("userId")
    chat_id = get_line_chat_id(source)
    source_type = source.get("type", "")
    key = in_flight_key(chat_id, actor_user_id, raw_text)
    existing = in_flight_tasks.get(key)
    now = datetime.now(timezone.utc)
    if existing:
        task = existing.get("task")
        ctx = existing.get("ctx")
        started_at = existing.get("started_at")
        if (
            isinstance(task, asyncio.Task)
            and not task.done()
            and isinstance(ctx, ProcessingContext)
            and isinstance(started_at, datetime)
            and (now - started_at).total_seconds() <= 10
        ):
            await reply_line_message(reply_token, f"\u6211\u9084\u5728\u8655\u7406\u4e0a\u4e00\u500b\u67e5\u8a62\uff0c\u76ee\u524d\u6b63\u5728{ctx.status}\uff0c\u7b49\u6211\u4e00\u4e0b\u3002")
            return

    ctx = ProcessingContext(
        actor_user_id=actor_user_id or "",
        chat_id=chat_id or "",
        reply_token=reply_token,
        source_type=source_type,
        raw_text=raw_text,
        status="\u5206\u6790\u8a0a\u606f\u4e2d",
        started_at=now,
    )
    task = asyncio.create_task(process_message_with_context(ctx, event))
    in_flight_tasks[key] = {"task": task, "ctx": ctx, "started_at": now}
    try:
        reply_payload = await asyncio.wait_for(asyncio.shield(task), timeout=get_processing_timeout_seconds())
        ctx.final_replied = True
        await reply_payload_with_token(reply_token, reply_payload)
        in_flight_tasks.pop(key, None)
    except asyncio.TimeoutError:
        ctx.interim_replied = True
        await reply_line_message(reply_token, build_interim_reply(ctx.status))
        task.add_done_callback(lambda done_task: asyncio.create_task(finalize_background_message(ctx, done_task, key)))
    except Exception:
        logger.exception("Failed to handle LINE text: %s", raw_text)
        in_flight_tasks.pop(key, None)
        await reply_line_message(reply_token, "\u525b\u525b\u8655\u7406\u5931\u6557\u4e86\uff0c\u6211\u6709\u8a18\u9304\u932f\u8aa4\uff0c\u8acb\u7a0d\u5f8c\u518d\u8a66\u3002")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/debug/send-due-reminders")
async def debug_send_due_reminders() -> dict[str, bool]:
    await send_due_reminders()
    return {"ok": True}


@app.post("/debug/dedupe-payables")
async def debug_dedupe_payables() -> dict[str, int]:
    deleted_count = dedupe_payables()
    return {"deleted_count": deleted_count}


@app.get("/debug/pending-payables")
async def debug_pending_payables() -> dict[str, object]:
    today = date.today()
    targets = {
        (today + timedelta(days=3)).isoformat(): 3,
        (today + timedelta(days=2)).isoformat(): 2,
        (today + timedelta(days=1)).isoformat(): 1,
    }
    with get_db() as conn:
        unpaid_rows = conn.execute(
            """
            SELECT id, line_user_id, item_type, amount, currency, due_date, owner, bank, status
            FROM payables
            WHERE status = 'unpaid'
            ORDER BY due_date ASC, id ASC
            """
        ).fetchall()
        reminder_rows = conn.execute(
            f"""
            SELECT id, line_user_id, item_type, amount, currency, due_date, owner, bank, status
            FROM payables
            WHERE status = 'unpaid'
              AND due_date IN ({','.join('?' for _ in targets)})
            ORDER BY due_date ASC, id ASC
            """,
            list(targets.keys()),
        ).fetchall()

    def row_dict(row: sqlite3.Row) -> dict[str, object]:
        return {
            "id": row["id"],
            "line_user_id": row["line_user_id"],
            "item_type": row["item_type"],
            "amount": row["amount"],
            "currency": row["currency"],
            "due_date": row["due_date"],
            "owner": row["owner"],
            "bank": row["bank"],
            "status": row["status"],
            "remind_days_before": targets.get(row["due_date"]),
        }

    return {
        "today": today.isoformat(),
        "target_dates": targets,
        "unpaid_payables": [row_dict(row) for row in unpaid_rows],
        "next_reminders": [row_dict(row) for row in reminder_rows],
    }


@app.post("/line/webhook")
async def line_webhook(
    request: Request,
    x_line_signature: str | None = Header(default=None, alias="X-Line-Signature"),
) -> dict[str, str]:
    body = await request.body()

    if not verify_line_signature(body, x_line_signature):
        raise HTTPException(status_code=400, detail="Invalid LINE signature.")

    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    events = payload.get("events", [])
    for event in events:
        log_usage("line", "webhook", detail=event.get("type"), success=True)
        if event.get("type") == "message" and event.get("message", {}).get("type") == "text":
            await handle_text_event(event)

    return {"status": "ok"}

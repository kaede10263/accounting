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
from zoneinfo import ZoneInfo

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
TAIPEI_TZ = ZoneInfo("Asia/Taipei")
HOME_TASK_DEFAULT_REMINDER_HOUR = 9


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
recent_line_event_times: dict[str, datetime] = {}
LINE_EVENT_DEDUP_WINDOW_SECONDS = 600

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
    "create_home_task",
    "query_home_tasks",
    "complete_home_task",
    "cancel_home_task",
    "query_home_task_history",
    "chat",
]
PayableAction = Literal["create_payable", "query_payables", "mark_payable_paid"]
PayableStatus = Literal["unpaid", "paid", "all"]
PlanOperation = Literal["create", "query", "update", "delete", "chat"]
PlanTarget = Literal["expenses", "incomes", "payables", "balance", "duplicates", "tasks", "chat"]
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


class ParsedExpenseCandidate(BaseModel):
    normalized_text: str
    expense: ExpenseEntry
    raw_model_output: str | None = None
    source: str = "openai"
    error: str | None = None


class MultiExpenseSemanticResult(BaseModel):
    original_text: str
    shared_context: str | None = None
    entries: list[ParsedExpenseCandidate] = Field(default_factory=list)


class ExpenseDbRowDraft(BaseModel):
    raw_text: str
    message_id: str | None = None
    expense: ExpenseEntry


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
    task_title: str | None = None
    task_item_key: str | None = None
    task_category: str | None = None
    scheduled_date: str | None = None
    scheduled_time: str | None = None
    task_requires_confirmation: bool = False


class HomeTaskRecord(BaseModel):
    id: int
    actor_user_id: str | None = None
    chat_id: str
    message_id: str | None = None
    title: str
    item_key: str | None = None
    category: str = "家庭事項"
    scheduled_date: str | None = None
    scheduled_time: str | None = None
    status: str = "pending"
    completed_at: str | None = None
    completion_text: str | None = None
    last_reminded_at: str | None = None
    raw_text: str
    created_at: str
    updated_at: str


class HomeTaskDraft(BaseModel):
    chat_id: str
    actor_user_id: str | None = None
    title: str
    item_key: str | None = None
    category: str = "家庭事項"
    scheduled_date: str | None = None
    scheduled_time: str | None = None
    original_text: str
    created_at: str


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
last_openai_route_plan: dict[str, str] | None = None
last_route_debug_by_user: dict[str, dict[str, object]] = {}
last_openai_expense_parse: dict[str, str] | None = None
last_expense_parse_debug_by_user: dict[str, dict[str, object]] = {}


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


def summarize_httpx_response(response: httpx.Response | None) -> str:
    if response is None:
        return ""
    try:
        text = response.text.strip()
    except Exception:
        return ""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text)[:500]


def log_line_api_http_error(api_name: str, exc: httpx.HTTPStatusError) -> None:
    response = exc.response
    request_id = response.headers.get("x-line-request-id", "")
    response_text = summarize_httpx_response(response)
    logger.error(
        "LINE %s failed: status=%s request_id=%s body=%s",
        api_name,
        response.status_code,
        request_id or "-",
        response_text or "-",
    )


def is_invalid_line_reply_error(exc: httpx.HTTPStatusError) -> bool:
    response_text = summarize_httpx_response(exc.response).lower()
    return exc.response.status_code == 400 and (
        "reply token" in response_text
        or "replytoken" in response_text
        or "invalid reply" in response_text
    )


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.row_factory = sqlite3.Row
    return conn


def taipei_now() -> datetime:
    if getattr(date, "__module__", "") != "datetime":
        fake_today = date.today()
        return datetime(fake_today.year, fake_today.month, fake_today.day, tzinfo=TAIPEI_TZ)
    return datetime.now(TAIPEI_TZ)


def taipei_today() -> date:
    return taipei_now().date()


def get_household_read_scope_line_user_id(_: str | None) -> str | None:
    # This bot is used as a household ledger, so read queries should aggregate
    # entries across family members instead of filtering to the current sender.
    return None


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


def log_sql_change(
    entity_type: str,
    action: str,
    row_id: int | None,
    summary: str,
    line_user_id: str | None = None,
) -> None:
    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO sql_change_logs (entity_type, action, row_id, summary, line_user_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_type,
                    action,
                    row_id,
                    summary,
                    line_user_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
    except sqlite3.Error:
        logger.exception("Failed to write sql change log entity=%s action=%s row_id=%s", entity_type, action, row_id)


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
            CREATE TABLE IF NOT EXISTS sql_change_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                action TEXT NOT NULL,
                row_id INTEGER,
                summary TEXT NOT NULL,
                line_user_id TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sql_change_logs_created_at ON sql_change_logs(created_at)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS utterance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT,
                scope_type TEXT,
                scope_id TEXT,
                actor_user_id TEXT,
                raw_text TEXT NOT NULL,
                predicted_domain TEXT,
                predicted_intent TEXT,
                predicted_tool_calls_json TEXT,
                predicted_confidence REAL,
                final_domain TEXT,
                final_intent TEXT,
                final_tool_calls_json TEXT,
                user_feedback TEXT,
                is_correct INTEGER,
                source TEXT NOT NULL DEFAULT 'line',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS line_event_receipts (
                dedup_key TEXT PRIMARY KEY,
                webhook_event_id TEXT,
                message_id TEXT,
                event_type TEXT,
                chat_id TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_line_event_receipts_created_at ON line_event_receipts(created_at)")
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incomes_date ON incomes(income_date)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS home_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_user_id TEXT,
                chat_id TEXT NOT NULL,
                message_id TEXT,
                title TEXT NOT NULL,
                item_key TEXT,
                category TEXT NOT NULL DEFAULT '家庭事項',
                scheduled_date TEXT,
                scheduled_time TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                completed_at TEXT,
                completion_text TEXT,
                last_reminded_at TEXT,
                raw_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_home_tasks_status_date
            ON home_tasks(status, scheduled_date)
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_home_tasks_message_id
            ON home_tasks(message_id)
            WHERE message_id IS NOT NULL
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS home_task_drafts (
                chat_id TEXT PRIMARY KEY,
                actor_user_id TEXT,
                title TEXT NOT NULL,
                item_key TEXT,
                category TEXT NOT NULL DEFAULT '家庭事項',
                scheduled_date TEXT,
                scheduled_time TEXT,
                original_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


@app.on_event("startup")
async def startup_event() -> None:
    init_db()
    asyncio.create_task(reminder_loop())


def verify_line_signature(body: bytes, signature: str | None) -> bool:
    secret = os.getenv("LINE_CHANNEL_SECRET")
    if not secret:
        return False
    if not signature:
        return False

    expected = base64.b64encode(hmac.new(secret.encode("utf-8"), body, "sha256").digest()).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def get_line_headers() -> dict[str, str]:
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN is not set.")

    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def normalize_line_messages(messages: list[dict] | dict) -> list[dict]:
    if isinstance(messages, dict):
        return [messages]
    return list(messages)


def build_line_push_payload(target_id: str, messages: list[dict] | dict) -> dict:
    normalized = normalize_line_messages(messages)
    return {
        "to": target_id,
        "messages": normalized,
    }


async def push_line_messages(target_id: str, messages: list[dict] | dict) -> None:
    payload = build_line_push_payload(target_id, messages)
    headers = get_line_headers()
    if os.getenv("APP_ENV", "dev").lower() == "test":
        logger.info("[TEST] LINE push to=%s payload=%s", target_id, payload)
        log_usage("line", "push", success=True, latency_ms=0)
        if any(message.get("type") == "image" for message in payload["messages"]):
            log_usage("line", "image", success=True, latency_ms=0)
        return
    request_started_at = datetime.now(timezone.utc)
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=get_ssl_verify()) as client:
            response = await client.post(
                LINE_PUSH_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        latency_ms = int((datetime.now(timezone.utc) - request_started_at).total_seconds() * 1000)
        log_usage("line", "push", success=True, latency_ms=latency_ms)
        if any(message.get("type") == "image" for message in payload["messages"]):
            log_usage("line", "image", success=True, latency_ms=latency_ms)
    except httpx.HTTPStatusError as exc:
        latency_ms = int((datetime.now(timezone.utc) - request_started_at).total_seconds() * 1000)
        log_line_api_http_error("push", exc)
        log_usage("line", "push", success=False, latency_ms=latency_ms)
        raise
    except Exception:
        latency_ms = int((datetime.now(timezone.utc) - request_started_at).total_seconds() * 1000)
        logger.exception("LINE push failed target=%s", target_id)
        log_usage("line", "push", success=False, latency_ms=latency_ms)
        raise


async def push_line_message(target_id: str, text: str) -> None:
    await push_line_messages(target_id, [{"type": "text", "text": text}])


async def send_due_reminders() -> None:
    today = date.today()
    reminder_offsets = (3, 2, 1)
    targets = {(today + timedelta(days=offset)).isoformat(): offset for offset in reminder_offsets}

    if not targets:
        return

    with get_db() as conn:
        placeholders = ",".join("?" for _ in targets)
        rows = conn.execute(
            f"""
            SELECT p.id, p.line_user_id, p.item_type, p.amount, p.currency, p.due_date,
                   p.owner, p.bank,
                   r.id AS reminder_id
            FROM payables p
            LEFT JOIN payable_reminders r
              ON r.payable_id = p.id
             AND r.reminded_on = ?
             AND r.remind_days_before = CAST(julianday(p.due_date) - julianday(?) AS INTEGER)
            WHERE p.status = 'unpaid'
              AND p.due_date IN ({placeholders})
            ORDER BY p.due_date ASC, p.id ASC
            """,
            [today.isoformat(), today.isoformat(), *targets.keys()],
        ).fetchall()

    for row in rows:
        if row["reminder_id"] is not None:
            continue
        due_date = row["due_date"]
        offset = targets.get(due_date)
        if offset is None:
            continue
        target_name = format_payable_target_name(row["item_type"], row["owner"], row["bank"])
        amount = int(row["amount"])
        text = (
            f"待繳款提醒\n"
            f"{target_name} TWD {amount} 將在 {offset} 天後到期。\n"
            f"到期日：{due_date}\n"
            f"繳完可以回覆：{target_name}已繳"
        )
        try:
            await push_line_message(str(row["line_user_id"]), text)
        except Exception:
            logger.exception("Failed to push reminder for payable_id=%s", row["id"])
            continue

        with get_db() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO payable_reminders (payable_id, remind_days_before, reminded_on)
                VALUES (?, ?, ?)
                """,
                (int(row["id"]), offset, today.isoformat()),
            )
            conn.commit()


async def reminder_loop() -> None:
    await asyncio.sleep(5)
    interval_hours = get_reminder_interval_hours()
    while True:
        try:
            await send_due_reminders()
            await send_home_task_reminders()
        except Exception:
            logger.exception("Reminder loop failed.")
        await asyncio.sleep(interval_hours * 60 * 60)


def parse_amount(text: str) -> int | None:
    multiplier = 1
    normalized = text.replace(",", "")
    normalized = normalized.replace("塊錢", "")
    normalized = normalized.replace("塊", "")
    normalized = normalized.replace("元", "")

    if "萬" in normalized:
        multiplier = 10000
        normalized = normalized.replace("萬", "")

    matches = re.findall(r"\d+(?:\.\d+)?", normalized)
    if not matches:
        return None

    value = float(matches[-1]) * multiplier
    return int(value)


def parse_payable_amount(text: str) -> int | None:
    cleaned = text.replace(",", "")
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


def parse_weekday_number(text: str) -> int | None:
    mapping = {
        "一": 0,
        "二": 1,
        "三": 2,
        "四": 3,
        "五": 4,
        "六": 5,
        "日": 6,
        "天": 6,
    }
    return mapping.get(text)


def resolve_weekday_date(prefix: str, weekday: int, today: date) -> date:
    if prefix.startswith("下"):
        start = today - timedelta(days=today.weekday()) + timedelta(days=7)
        return start + timedelta(days=weekday)
    if prefix.startswith(("這", "本")):
        start = today - timedelta(days=today.weekday())
        return start + timedelta(days=weekday)
    days_ahead = weekday - today.weekday()
    if days_ahead < 0:
        days_ahead += 7
    return today + timedelta(days=days_ahead)


def parse_due_date(text: str) -> str | None:
    today = taipei_today()

    iso_match = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", text)
    if iso_match:
        return date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))).isoformat()

    month_day_match = re.search(r"(\d{1,2})\s*(?:/|月)\s*(\d{1,2})\s*(?:日|號)?", text)
    if month_day_match:
        due = date(today.year, int(month_day_match.group(1)), int(month_day_match.group(2)))
        if due < today:
            due = date(today.year + 1, due.month, due.day)
        return due.isoformat()

    day_match = re.search(r"(?<!月)(\d{1,2})\s*(?:日|號)", text)
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

    weekday_match = re.search(r"(下個星期|下星期|下禮拜|下週|下周|這星期|這禮拜|這週|這周|本星期|本禮拜|本週|本周|星期|禮拜|週)([一二三四五六日天])", text)
    if weekday_match:
        weekday = parse_weekday_number(weekday_match.group(2))
        if weekday is not None:
            return resolve_weekday_date(weekday_match.group(1), weekday, today).isoformat()

    if "今天" in text:
        return today.isoformat()
    if "明天" in text:
        return (today + timedelta(days=1)).isoformat()
    if "後天" in text:
        return (today + timedelta(days=2)).isoformat()
    return None


def home_task_confirmation_yes(text: str) -> bool:
    return re.sub(r"\s+", "", text) in {"要", "好", "確認", "確定", "可以", "是"}


def home_task_confirmation_no(text: str) -> bool:
    return re.sub(r"\s+", "", text) in {"不要", "取消", "不用", "否"}


def strip_home_task_date_phrases(text: str) -> str:
    cleaned = text
    patterns = (
        r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}",
        r"\d{1,2}\s*(?:/|月)\s*\d{1,2}\s*(?:日|號)?",
        r"(?:下個星期|下星期|下禮拜|下週|下周|這星期|這禮拜|這週|這周|本星期|本禮拜|本週|本周|星期|禮拜|週)[一二三四五六日天]",
        r"今天|明天|後天",
    )
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_home_task_title(title: str) -> str:
    cleaned = strip_home_task_date_phrases(title)
    cleaned = re.sub(r"^(?:提醒我(?:要)?|提醒(?:我)?|記得|要做|要|我要|完成|已完成)\s*", "", cleaned)
    cleaned = re.sub(r"^(?:可能要|可能會)\s*", "", cleaned)
    cleaned = re.sub(r"[。．，,！!？?]+$", "", cleaned)
    return cleaned.strip()


def extract_home_task_item_key(title: str) -> str | None:
    normalized = normalize_home_task_title(title)
    for verb in ("清洗", "清理", "更換", "整理", "保養", "檢查", "維修", "修理", "換", "洗", "修"):
        if normalized.startswith(verb):
            item_key = normalized[len(verb) :].strip()
            return item_key or normalized
    return normalized or None


def infer_home_task_category(title: str) -> str:
    normalized = normalize_home_task_title(title)
    if any(keyword in normalized for keyword in ("地漏", "濾芯", "水龍頭", "維修", "修理", "修", "換")):
        return "居家修繕"
    if any(keyword in normalized for keyword in ("洗", "清洗", "清理", "整理", "冷氣", "洗衣機", "陽台")):
        return "家庭事項"
    if any(keyword in normalized for keyword in ("回診", "看醫生", "吃藥")):
        return "健康"
    return "家庭事項"


def parse_home_task_time(text: str) -> str | None:
    return parse_time_fallback(text)


def is_home_task_create_text(text: str) -> bool:
    if parse_due_date(text) is None:
        return False
    if is_question_text(text):
        return False
    return any(keyword in text for keyword in ("提醒我", "記得", "要", "要做"))


def is_ambiguous_home_task_create_text(text: str) -> bool:
    return parse_due_date(text) is not None and any(keyword in text for keyword in ("可能要", "可能會"))


def is_home_task_query_text(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return any(keyword in normalized for keyword in ("家庭事項", "居家事項", "事情要做", "有哪些提醒", "沒完成", "完成了嗎"))


def is_home_task_history_text(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return any(keyword in normalized for keyword in ("上次", "最近一次", "紀錄", "換過幾次", "最近三次"))


def is_home_task_cancel_text(text: str) -> bool:
    return any(keyword in text for keyword in ("取消", "刪除", "删除", "移除")) and (
        parse_due_date(text) is not None
        or any(keyword in text for keyword in ("換", "洗", "修", "整理", "清洗", "清理"))
    )


def is_generic_home_task_completion_text(text: str) -> bool:
    return re.sub(r"\s+", "", text) in {"已換", "完成了", "好了", "已完成", "弄好了"}


def is_home_task_complete_text(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    if is_generic_home_task_completion_text(text):
        return True
    patterns = (
        r"^已(?:換|洗|修|整理|清洗|清理).+",
        r"^完成.+",
        r"^.+(?:換|洗|修|整理|清洗|清理)好了$",
        r"^.+已經(?:換|洗|修|整理|清洗|清理)好了$",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def extract_home_task_title_from_create(text: str) -> str | None:
    title = normalize_home_task_title(text)
    return title or None


def extract_home_task_lookup_term(text: str) -> str | None:
    normalized = re.sub(r"\s+", "", text)
    if is_generic_home_task_completion_text(normalized):
        return None
    if any(keyword in normalized for keyword in ("家庭事項", "居家事項")) and any(keyword in normalized for keyword in ("有哪些", "還有", "沒完成", "提醒")):
        return None
    cleaned = strip_home_task_date_phrases(text)
    cleaned = re.sub(r"^(?:上次|最近一次|列出|最近三次|還有哪些|有哪些|取消|刪除|删除|移除|已完成|完成|已)\s*", "", cleaned)
    cleaned = re.sub(r"(?:是什麼時候|是哪一天|的紀錄|完成了嗎|還沒完成|沒完成)[？?]?\s*$", "", cleaned)
    cleaned = re.sub(r"(?:好了|已經好了)\s*$", "", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return None

    object_first = re.match(r"^(.+?)(換|洗|修|整理|清洗|清理)(?:好了)?$", cleaned)
    if object_first:
        return object_first.group(1).strip()

    title = normalize_home_task_title(cleaned)
    item_key = extract_home_task_item_key(title)
    return item_key or title or None


def format_task_date_with_weekday(date_text: str | None) -> str:
    if not date_text:
        return ""
    target = date.fromisoformat(date_text)
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"{date_text}（{weekdays[target.weekday()]}）"


def resolve_home_task_chat_id(chat_id: str | None, actor_user_id: str | None) -> str:
    return chat_id or actor_user_id or "_global"


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
        "\u591a\u5c11",
        "\u53ef\u4ee5",
        "\u80fd\u4e0d\u80fd",
        "\u5efa\u8b70",
        "\u53ef\u52d5\u7528",
        "\u9084\u5269",
        "\u6709\u591a\u5c11\u9322",
    )
    return any(term in normalized for term in advice_terms)


def is_available_purchase_cash_query(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    spending_keywords = ("\u8cb7", "\u63db", "\u5165\u624b", "\u6dfb\u8cfc", "\u5165\u8cb7")
    if not any(keyword in normalized for keyword in spending_keywords):
        return False
    if any(term in normalized for term in ("\u80a1\u7968", "\u8cb7\u80a1", "\u6295\u8cc7", "ETF", "etf", "\u57fa\u91d1")):
        return False
    advice_terms = ("\u591a\u5c11", "\u53ef\u4ee5", "\u80fd\u4e0d\u80fd", "\u5920\u4e0d\u5920", "\u80fd\u5426", "\u9084\u6709\u591a\u5c11\u9322")
    return any(term in normalized for term in advice_terms)


def extract_purchase_purpose(text: str) -> str | None:
    normalized = text.strip().replace("？", "?")
    match = re.search(r"(?:\u8cb7|\u63db|\u5165\u624b|\u6dfb\u8cfc)(.+?)(?:\?|\u55ce|\u591a\u5c11|\u53ef\u4ee5|\u80fd\u4e0d\u80fd|\u5920\u4e0d\u5920|\u80fd\u5426|$)", normalized)
    if match:
        purpose = match.group(1).strip()
        return purpose or None
    return None


def parse_hour_text(text: str) -> int | None:
    if not text:
        return None
    chinese_numbers = {
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
    if text.isdigit():
        value = int(text)
        return value if 0 <= value <= 23 else None
    return chinese_numbers.get(text)


def parse_time_fallback(text: str) -> str | None:
    normalized = text.replace("：", ":")
    match = re.search(r"(?:\u4e0a\u5348|\u4e0b\u5348|\u665a\u4e0a|\u4e2d\u5348)?\s*(\d{1,2})[:：](\d{2})", normalized)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"
    match = re.search(r"(\u4e0a\u5348|\u4e0b\u5348|\u665a\u4e0a|\u4e2d\u5348)?\s*(\d{1,2}|[一二兩三四五六七八九十]{1,3})\s*(?:\u9ede|\u9ede\u9418|\u9ede\u534a)", normalized)
    if match:
        period = match.group(1) or ""
        hour = parse_hour_text(match.group(2))
        if hour is None:
            return None
        if "\u534a" in match.group(0):
            minute = 30
        else:
            minute = 0
        if period in {"\u4e0b\u5348", "\u665a\u4e0a"} and 1 <= hour <= 11:
            hour += 12
        if period == "\u4e2d\u5348" and hour == 12:
            hour = 12
        return f"{hour:02d}:{minute:02d}"
    return None


def get_income_type(text: str) -> str | None:
    if any(keyword in text for keyword in ("\u85aa\u6c34", "\u85aa\u8cc7")):
        return "薪資收入"
    if "\u88dc\u52a9" in text:
        return "補助收入"
    if any(keyword in text for keyword in ("\u4e2d\u734e", "\u734e\u91d1")):
        return "獎金收入"
    if any(keyword in text for keyword in ("\u79df\u91d1", "\u623f\u79df")):
        return "租金收入"
    if "\u80a1\u5229" in text:
        return "股利收入"
    if any(keyword in text for keyword in ("\u63a5\u6848", "\u5916\u5305", "\u4efb\u52d9\u8cbb")):
        return "接案收入"
    if any(keyword in text for keyword in ("\u696d\u5916", "\u526f\u696d")):
        return "業外收入"
    return None


def route_domain(route: ActionRoute) -> str:
    if route.action in {"create_payable", "query_payables", "mark_payable_paid"}:
        return "payable"
    if route.action in {"create_income", "query_incomes", "list_incomes"}:
        return "income"
    if route.action in {"create_home_task", "query_home_tasks", "complete_home_task", "cancel_home_task", "query_home_task_history"}:
        return "home"
    if route.action in {"query_balance", "query_available_cash", "ask_available_investment_cash"}:
        return "balance"
    if route.action in {"list_duplicates", "delete_duplicates"}:
        return "duplicates"
    if route.action == "chat":
        return "chat"
    return "finance"


def route_tool_calls_json(route: ActionRoute) -> str:
    payload = {
        "action": route.action,
        "item_type": route.item_type,
        "owner": route.owner,
        "bank": route.bank,
        "amount": route.amount,
        "due_date": route.due_date,
        "date_text": route.date_text,
        "category": route.category,
        "income_type": route.income_type,
        "status": route.status,
        "purchase_purpose": route.purchase_purpose,
        "task_title": route.task_title,
        "task_item_key": route.task_item_key,
        "task_category": route.task_category,
        "scheduled_date": route.scheduled_date,
        "scheduled_time": route.scheduled_time,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def route_intent(route: ActionRoute) -> str:
    if route.action in {"create_expense", "create_payable", "create_income", "create_home_task"}:
        return "create"
    if route.action in {"mark_payable_paid", "complete_home_task", "cancel_home_task"}:
        return "update"
    if route.action in {"delete_expense", "delete_duplicates"}:
        return "delete"
    if route.action in {"list_expenses", "list_incomes", "query_home_task_history"}:
        return "list"
    if route.action in {"query_expenses", "query_payables", "query_incomes", "query_balance", "query_available_cash", "ask_available_investment_cash", "top_expense", "query_home_tasks"}:
        return "query"
    return "chat"


def log_utterance(
    raw_text: str,
    message_id: str | None,
    actor_user_id: str | None,
    scope_type: str | None,
    scope_id: str | None,
    route: ActionRoute,
    source: str = "line",
) -> None:
    tool_calls_json = route_tool_calls_json(route)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO utterance_logs (
                message_id, scope_type, scope_id, actor_user_id, raw_text,
                predicted_domain, predicted_intent, predicted_tool_calls_json, predicted_confidence,
                final_domain, final_intent, final_tool_calls_json,
                source, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                scope_type,
                scope_id,
                actor_user_id,
                raw_text,
                route_domain(route),
                route_intent(route),
                tool_calls_json,
                route.confidence,
                route_domain(route),
                route_intent(route),
                tool_calls_json,
                source,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def get_last_utterance(scope_id: str | None) -> sqlite3.Row | None:
    if not scope_id:
        return None
    with get_db() as conn:
        return conn.execute(
            """
            SELECT *
            FROM utterance_logs
            WHERE scope_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (scope_id,),
        ).fetchone()


def feedback_to_domain_intent(feedback_text: str) -> tuple[str, str]:
    normalized = feedback_text.lower()
    if any(keyword in feedback_text for keyword in ("\u5c45\u5bb6", "\u63d0\u9192", "\u4fee\u7e55", "\u7dad\u8b77")):
        return "reminder", "create"
    if "\u652f\u51fa" in feedback_text or "expense" in normalized:
        return "finance", "create"
    if "\u6536\u5165" in feedback_text or "income" in normalized:
        return "income", "create"
    if any(keyword in feedback_text for keyword in ("\u7e73\u8cbb", "\u5f85\u7e73", "payable")):
        return "payable", "create"
    return "chat", "chat"


def apply_last_utterance_feedback(raw_text: str, actor_user_id: str | None, scope_id: str | None) -> str | None:
    if not raw_text.startswith("\u4e0a\u4e00\u53e5"):
        return None
    last = get_last_utterance(scope_id)
    if last is None:
        return "我還找不到上一句可以修正的紀錄。"
    final_domain, final_intent = feedback_to_domain_intent(raw_text)
    with get_db() as conn:
        conn.execute(
            """
            UPDATE utterance_logs
            SET user_feedback = ?,
                is_correct = 0,
                final_domain = ?,
                final_intent = ?
            WHERE id = ?
            """,
            (raw_text, final_domain, final_intent, int(last["id"])),
        )
        conn.commit()
    return "已記錄修正，之後我會用來校正判斷。"


def export_training_data(output_path: str) -> tuple[int, str]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT raw_text, final_domain, final_intent, user_feedback, is_correct
            FROM utterance_logs
            WHERE is_correct = 0
              AND user_feedback IS NOT NULL
            ORDER BY id ASC
            """
        ).fetchall()
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            label = f"{row['final_domain']}.{label_suffix_from_intent(str(row['final_intent']))}"
            payload = {
                "text": row["raw_text"],
                "label": label,
                "feedback": row["user_feedback"],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count, output_path


def label_suffix_from_intent(intent: str) -> str:
    mapping = {
        "create": "create_maintenance_record",
        "update": "update_record",
        "delete": "delete_record",
        "query": "query_record",
        "list": "list_record",
        "chat": "chat",
    }
    return mapping.get(intent, intent)


def local_intent_fallback(text: str) -> IntentResult:
    if parse_amount(text) is not None:
        return IntentResult(intent="create", confidence=0.70, reason="local fallback amount")

    if any(keyword in text.lower() for keyword in ("刪除", "删除", "移除", "delete")):
        return IntentResult(intent="delete", confidence=0.80, reason="local fallback delete keyword")

    if any(keyword in text for keyword in ("明細", "列出", "清單", "有哪些")):
        return IntentResult(intent="list", confidence=0.70, reason="local fallback list keyword")

    if any(keyword in text for keyword in ("總共", "多少", "統計", "查詢", "小計", "平均", "最高", "最低", "占比", "比例", "圖表")):
        return IntentResult(intent="summary", confidence=0.70, reason="local fallback summary keyword")

    if normalized := text.strip().lower():
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
                "enum": ["expenses", "incomes", "payables", "balance", "duplicates", "tasks", "chat"],
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
""".strip() + AGENT_TASKS_PROMPT_SUFFIX


AGENT_TASKS_PROMPT_SUFFIX = """

tasks：家庭事項、居家修繕、生活待辦及一般非帳務提醒。
- 「下禮拜三提醒我要換地漏」=> create tasks
- 「明天要洗冷氣」=> create tasks
- 「有哪些家庭事項還沒完成？」=> query tasks
- 「已換地漏」=> update tasks
- 「地漏換好了」=> update tasks
- 「取消下週洗冷氣」=> delete tasks
- 「上次換地漏是什麼時候？」=> query tasks
- 「最近一次洗冷氣是哪一天？」=> query tasks
- tasks 只輸出結構化 AgentPlan，不要輸出 SQL。
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
        elif plan.target == "tasks":
            action = "cancel_home_task"
        else:
            action = "chat"
        return ActionRoute(
            action=action,
            should_mutate_db=should_mutate_db if action in {"delete_expense", "cancel_home_task"} else False,
            confidence=plan.confidence,
            reason=reason,
            category=plan.category,
            amount=plan.amount,
            purchase_purpose=plan.note,
            task_title=plan.note,
            task_item_key=plan.keywords[0] if plan.keywords else None,
            task_category=plan.category,
            scheduled_date=plan.start_date,
        )

    if plan.operation == "update":
        if plan.target == "payables":
            action = "mark_payable_paid"
        elif plan.target == "tasks":
            action = "complete_home_task"
        else:
            action = "chat"
        return ActionRoute(
            action=action,
            should_mutate_db=should_mutate_db,
            confidence=plan.confidence,
            reason=reason,
            item_type=plan_item_type(plan),
            amount=plan.amount,
            due_date=plan.end_date or plan.start_date,
            category=plan.category,
            task_title=plan.note,
            task_item_key=plan.keywords[0] if plan.keywords else None,
            task_category=plan.category,
            scheduled_date=plan.start_date,
        )

    if plan.operation == "create":
        if plan.target == "incomes":
            action = "create_income"
        elif plan.target == "payables":
            action = "create_payable"
        elif plan.target == "expenses":
            action = "create_expense"
        elif plan.target == "tasks":
            action = "create_home_task"
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
            task_title=plan.note,
            task_item_key=plan.keywords[0] if plan.keywords else None,
            task_category=plan.category,
            scheduled_date=plan.start_date,
        )

    if plan.target == "duplicates":
        action = "list_duplicates"
    elif plan.target == "payables":
        action = "query_payables"
    elif plan.target == "tasks":
        action = "query_home_task_history" if any(keyword in raw_text for keyword in ("上次", "最近一次", "紀錄", "幾次")) else "query_home_tasks"
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
        income_type=plan_item_type(plan) if plan.target == "incomes" else None,
        purchase_purpose=plan.note,
        task_title=plan.note,
        task_item_key=plan.keywords[0] if plan.keywords else None,
        task_category=plan.category,
        scheduled_date=plan.start_date,
    )


def route_action_with_openai(text: str) -> ActionRoute:
    global last_openai_route_plan
    model = get_current_model()
    logger.info("Requesting OpenAI route plan model=%s text=%s", model, text)
    started_at = datetime.now(timezone.utc)
    prompt = (
        "判斷以下記帳機器人使用者訊息最適合的 action。\n"
        "請只輸出 JSON，符合給定 schema。\n"
        f"user_text: {text}"
    )
    messages = [
        {"role": "system", "content": build_agent_planner_prompt(date.today().isoformat())},
        {"role": "user", "content": prompt},
    ]

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "agent_plan",
                "schema": build_agent_plan_schema(),
                "strict": True,
            },
        },
    )
    latency_ms = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
    log_usage("openai", "chat_completion", detail=get_current_model(), success=True, latency_ms=latency_ms)
    content = str(response.output_text).strip()
    last_openai_route_plan = {"input": text, "output": content}
    plan = AgentPlan.model_validate_json(content)
    return agent_plan_to_action_route(plan, text)


def fallback_action_route(text: str) -> ActionRoute:
    has_amount = parse_amount(text) is not None
    payable_amount = parse_payable_amount(text)
    home_task_route = local_home_task_route(text)
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
        if "最貴" in text:
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
    if home_task_route is not None:
        return home_task_route.model_copy(update={"reason": "local fallback home task"})
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
    return any(keyword in normalized for keyword in ("還有哪些沒繳", "哪些沒繳", "目前未繳", "未繳費用", "待繳"))


def is_payable_create_text(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    if any(keyword in normalized for keyword in ("提醒我", "要繳", "待繳", "繳費")):
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
    return any(keyword in normalized for keyword in ("統計", "總共", "總花費", "平均", "占比", "比例"))


def local_home_task_route(text: str) -> ActionRoute | None:
    scheduled_date = parse_due_date(text)
    title = extract_home_task_title_from_create(text) if scheduled_date else None
    task_item_key = extract_home_task_item_key(title or "") if title else None
    task_category = infer_home_task_category(title or "") if title else None
    if is_ambiguous_home_task_create_text(text) and title and scheduled_date:
        return ActionRoute(
            action="create_home_task",
            should_mutate_db=True,
            confidence=0.80,
            reason="deterministic ambiguous home task create",
            task_title=title,
            task_item_key=task_item_key,
            task_category=task_category,
            scheduled_date=scheduled_date,
            scheduled_time=parse_home_task_time(text),
            task_requires_confirmation=True,
        )
    if is_home_task_create_text(text) and title and scheduled_date:
        return ActionRoute(
            action="create_home_task",
            should_mutate_db=True,
            confidence=0.96,
            reason="deterministic home task create",
            task_title=title,
            task_item_key=task_item_key,
            task_category=task_category,
            scheduled_date=scheduled_date,
            scheduled_time=parse_home_task_time(text),
        )
    if is_home_task_cancel_text(text):
        return ActionRoute(
            action="cancel_home_task",
            should_mutate_db=True,
            confidence=0.92,
            reason="deterministic home task cancel",
            task_item_key=extract_home_task_lookup_term(text),
            scheduled_date=scheduled_date,
        )
    if is_home_task_complete_text(text):
        return ActionRoute(
            action="complete_home_task",
            should_mutate_db=True,
            confidence=0.93,
            reason="deterministic home task complete",
            task_item_key=extract_home_task_lookup_term(text),
        )
    if is_home_task_history_text(text):
        return ActionRoute(
            action="query_home_task_history",
            should_mutate_db=False,
            confidence=0.93,
            reason="deterministic home task history query",
            task_item_key=extract_home_task_lookup_term(text),
        )
    if is_home_task_query_text(text):
        return ActionRoute(
            action="query_home_tasks",
            should_mutate_db=False,
            confidence=0.92,
            reason="deterministic home task pending query",
            task_item_key=extract_home_task_lookup_term(text),
        )
    return None


def save_last_route_debug(
    line_user_id: str | None,
    original_text: str,
    route_source: str,
    route: ActionRoute,
    *,
    raw_model_output: str | None = None,
    error: str | None = None,
) -> None:
    user_key = expense_parse_debug_user_key(line_user_id)
    last_route_debug_by_user[user_key] = {
        "original_text": original_text,
        "route_source": route_source,
        "raw_route_model_output": raw_model_output,
        "final_route": route.model_dump(),
        "error": error,
    }


def route_action(text: str, line_user_id: str | None) -> ActionRoute:
    pending = get_payable_draft(line_user_id) if line_user_id else None
    due_date = parse_due_date(text)
    payable_type = get_payable_type(text)
    payable_amount = parse_payable_amount(text)

    def finish(
        route: ActionRoute,
        route_source: str,
        *,
        raw_model_output: str | None = None,
        error: str | None = None,
    ) -> ActionRoute:
        save_last_route_debug(
            line_user_id,
            text,
            route_source,
            route,
            raw_model_output=raw_model_output,
            error=error,
        )
        return route

    if is_paid_text(text) and payable_type and not is_question_text(text):
        return finish(
            ActionRoute(
            action="mark_payable_paid",
            should_mutate_db=True,
            confidence=0.98,
            reason="deterministic payable paid update",
            item_type=payable_type,
            owner=get_owner(text),
            bank=get_bank(text),
            status="paid",
            ),
            "deterministic",
        )

    if payable_type and payable_amount is not None and is_payable_create_text(text):
        return finish(
            ActionRoute(
            action="create_payable",
            should_mutate_db=True,
            confidence=0.95,
            reason="deterministic payable reminder create",
            item_type=payable_type,
            amount=payable_amount,
            due_date=due_date,
            owner=get_owner(text),
            bank=get_bank(text),
            ),
            "deterministic",
        )

    if is_question_text(text) and (payable_type or is_general_unpaid_payable_query(text)):
        return finish(
            ActionRoute(
            action="query_payables",
            should_mutate_db=False,
            confidence=0.95,
            reason="deterministic payable question",
            item_type=payable_type,
            status="all" if payable_type else "unpaid",
            ),
            "deterministic",
        )

    if is_income_list_query(text):
        return finish(
            ActionRoute(
            action="list_incomes",
            should_mutate_db=False,
            confidence=0.95,
            reason="deterministic income list query",
            income_type=get_income_type(text),
            ),
            "deterministic",
        )

    if is_income_query(text):
        return finish(
            ActionRoute(
            action="query_incomes",
            should_mutate_db=False,
            confidence=0.95,
            reason="deterministic income aggregate query",
            income_type=get_income_type(text),
            ),
            "deterministic",
        )

    home_task_route = local_home_task_route(text)
    if home_task_route is not None:
        return finish(home_task_route, "deterministic")

    raw_route_model_output: str | None = None
    route_error: str | None = None
    base_route_source = "openai"
    try:
        route = route_action_with_openai(text)
        raw_route_model_output = (last_openai_route_plan or {}).get("output")
    except Exception:
        logger.exception("OpenAI action routing failed; using local fallback for text: %s", text)
        route = fallback_action_route(text)
        base_route_source = "fallback"
        route_error = "OpenAI route planning failed; used local fallback."

    if is_delete_duplicate_text(text):
        return finish(
            ActionRoute(
            action="delete_duplicates",
            should_mutate_db=False,
            confidence=max(route.confidence, 0.90),
            reason="local correction: duplicate delete",
            ),
            f"{base_route_source}+local_correction",
            raw_model_output=raw_route_model_output,
            error=route_error,
        )

    if is_duplicate_data_text(text):
        return finish(
            ActionRoute(
            action="list_duplicates",
            should_mutate_db=False,
            confidence=max(route.confidence, 0.90),
            reason="local correction: duplicate list",
            ),
            f"{base_route_source}+local_correction",
            raw_model_output=raw_route_model_output,
            error=route_error,
        )

    if is_statistical_expense_query(text):
        return finish(
            ActionRoute(
            action="query_expenses",
            should_mutate_db=False,
            confidence=max(route.confidence, 0.90),
            reason="local correction: statistical expense query",
            category=route.category,
            purchase_purpose=route.purchase_purpose,
            ),
            f"{base_route_source}+local_correction",
            raw_model_output=raw_route_model_output,
            error=route_error,
        )

    if route.action == "query_balance" and is_available_purchase_cash_query(text):
        corrected_route = route.model_copy(
            update={
                "action": "query_available_cash",
                "should_mutate_db": False,
                "confidence": max(route.confidence, 0.90),
                "reason": "local correction: available cash purchase query",
                "purchase_purpose": route.purchase_purpose or extract_purchase_purpose(text),
            }
        )
        return finish(
            corrected_route,
            f"{base_route_source}+local_correction",
            raw_model_output=raw_route_model_output,
            error=route_error,
        )

    if route.action == "query_expenses" and is_balance_query(text):
        corrected_route = route.model_copy(
            update={
                "action": "query_balance",
                "should_mutate_db": False,
                "confidence": max(route.confidence, 0.90),
                "reason": "local correction: balance query",
                "purchase_purpose": None,
            }
        )
        if is_available_purchase_cash_query(text):
            corrected_route = corrected_route.model_copy(
                update={
                    "action": "query_available_cash",
                    "reason": "local correction: available cash purchase query",
                    "purchase_purpose": extract_purchase_purpose(text),
                }
            )
        return finish(
            corrected_route,
            f"{base_route_source}+local_correction",
            raw_model_output=raw_route_model_output,
            error=route_error,
        )

    return finish(route, base_route_source, raw_model_output=raw_route_model_output, error=route_error)


def summarize_income_type(income_type: str | None) -> str:
    return income_type or "收入"


def normalize_payable_item_type(item_type: str | None) -> str | None:
    if not item_type:
        return None
    raw = str(item_type).strip()
    normalized = re.sub(r"\s+", "", raw)
    synonyms = {
        "房貸": {"房貸", "房屋貸款", "房屋貸", "房屋房貸", "mortgage"},
        "信貸": {"信貸", "信用貸款", "個人信貸", "信用貸", "信貸款"},
        "貸款": {"貸款", "借款", "loan"},
        "信用卡": {"信用卡", "卡費", "刷卡", "卡款", "creditcard", "credit"},
        "保母費": {"保母費", "保母", "托育費", "保母托育"},
        "幼稚園": {"幼稚園", "托兒所", "學費", "幼兒園"},
        "管理費": {"管理費", "社區管理費", "大樓管理費"},
        "水費": {"水費"},
        "電費": {"電費"},
        "瓦斯費": {"瓦斯費", "天然氣費"},
        "電話費": {"電話費", "手機費", "手機帳單"},
        "網路費": {"網路費", "網路帳單", "網費", "寬頻費"},
        "保險": {"保險", "保費", "保險費"},
    }
    for canonical, values in synonyms.items():
        if normalized in values:
            return canonical
    if normalized.endswith("卡") and "信用" in normalized:
        return "信用卡"
    if normalized.endswith("費") or normalized.endswith("款"):
        return raw
    return raw


def get_payable_type(text: str) -> str | None:
    normalized_text = re.sub(r"\s+", "", text)
    keyword_order = [
        ("房貸", ("房貸", "房屋貸款", "房屋貸")),
        ("信貸", ("信貸", "信用貸款", "個人信貸", "信用貸")),
        ("貸款", ("貸款", "借款")),
        ("信用卡", ("信用卡", "卡費", "刷卡", "卡款")),
        ("保母費", ("保母費", "保母", "托育費")),
        ("幼稚園", ("幼稚園", "幼兒園", "托兒所", "學費")),
        ("管理費", ("管理費", "社區管理費", "大樓管理費")),
        ("水費", ("水費",)),
        ("電費", ("電費",)),
        ("瓦斯費", ("瓦斯費", "天然氣費")),
        ("電話費", ("電話費", "手機費", "手機帳單")),
        ("網路費", ("網路費", "網路帳單", "網費", "寬頻費")),
        ("保險", ("保險", "保費", "保險費")),
    ]
    for canonical, keywords in keyword_order:
        if any(keyword in normalized_text for keyword in keywords):
            return canonical
    return None


def get_owner(text: str) -> str | None:
    owner_patterns = (
        (r"老婆", "老婆"),
        (r"老公", "老公"),
        (r"先生", "先生"),
        (r"太太", "太太"),
    )
    for pattern, owner in owner_patterns:
        if re.search(pattern, text):
            return owner
    return None


def get_bank(text: str) -> str | None:
    bank_keywords = (
        "玉山",
        "國泰",
        "富邦",
        "台新",
        "永豐",
        "中信",
        "中國信託",
        "兆豐",
        "第一",
        "合庫",
        "華南",
        "彰銀",
        "台銀",
        "元大",
        "新光",
        "聯邦",
        "渣打",
        "花旗",
    )
    for keyword in bank_keywords:
        if keyword in text:
            return "中信" if keyword == "中國信託" else keyword
    return None


def format_payable_target_name(item_type: str | None, owner: str | None = None, bank: str | None = None) -> str:
    parts = [part for part in (owner, bank, item_type) if part]
    return " ".join(parts) if parts else (item_type or "這筆費用")


def is_paid_text(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    patterns = [
        r"^已繳(?:了)?$",
        r"^繳了$",
        r"^已經繳了$",
        r"^繳完了$",
        r"^已繳.+",
        r"^.+已繳$",
        r"^.+繳了$",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def parse_payable_query_with_openai(raw_text: str) -> PayableQuery:
    system_prompt = (
        "你是台灣家計 Bot 的繳費查詢解析器。\n"
        "任務：把使用者的自然語言查詢轉成固定 JSON。\n"
        "只輸出 JSON，不要回答。\n"
        "status 只能是 unpaid、paid、all。\n"
        "date_range_type 只能是 this_month、custom、unspecified。\n"
        "如果使用者有問『繳了嗎』『還沒繳嗎』通常是查詢某個項目目前是否 unpaid 或 paid。\n"
        "房貸、信貸、貸款、信用卡都要保留原意，不可亂合併。"
    )
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
    model = get_current_model()
    started_at = datetime.now(timezone.utc)
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "payable_query",
                "schema": schema,
                "strict": True,
            },
        },
    )
    latency_ms = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
    log_usage("openai", "chat_completion", detail=get_current_model(), success=True, latency_ms=latency_ms)
    content = str(response.output_text).strip()
    return PayableQuery.model_validate_json(content)


def parse_time_fragment(text: str) -> str | None:
    normalized = text.replace("：", ":")
    match = re.search(r"(\d{1,2}:\d{2})", normalized)
    if not match:
        return None
    try:
        hour_text, minute_text = match.group(1).split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError:
        return None
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return f"{hour:02d}:{minute:02d}"
    return None


def infer_category(text: str) -> Category:
    normalized = text.lower().replace(" ", "")

    if any(keyword in normalized for keyword in ("早餐", "午餐", "晚餐", "飲料", "咖啡", "麵包", "餐", "吃飯", "宵夜", "點心", "停車費", "飲料25")):
        if any(keyword in normalized for keyword in ("停車", "油", "加油", "捷運", "公車", "高鐵", "台鐵", "uber", "計程車", "火車")):
            return "交通"
        return "餐飲"

    if any(keyword in normalized for keyword in ("加油", "停車", "捷運", "公車", "高鐵", "台鐵", "uber", "計程車", "交通", "火車")):
        return "交通"

    if any(keyword in normalized for keyword in ("全聯", "家樂福", "超市", "日用品", "衛生紙", "牛奶", "香蕉", "芭樂", "蓮霧", "水果")):
        return "生活用品"

    if any(keyword in normalized for keyword in ("衣服", "鞋", "外套", "褲", "包包", "購物", "食材")):
        return "購物"

    if any(keyword in normalized for keyword in ("牙醫", "醫院", "藥", "掛號", "看醫生")):
        return "醫療"

    if "房租" in normalized:
        return "房租"
    if any(keyword in normalized for keyword in ("水費", "自來水")):
        return "水費"
    if any(keyword in normalized for keyword in ("電費", "台電")):
        return "電費"
    if any(keyword in normalized for keyword in ("瓦斯", "天然氣")):
        return "瓦斯費"
    if any(keyword in normalized for keyword in ("電話費", "手機費")):
        return "電話費"
    if any(keyword in normalized for keyword in ("網路費", "寬頻")):
        return "網路費"
    if any(keyword in normalized for keyword in ("保險", "保費")):
        return "保險"
    if any(keyword in normalized for keyword in ("信用卡", "卡費")):
        return "信用卡"
    if any(keyword in normalized for keyword in ("房貸", "信貸", "貸款")):
        return "貸款"
    if "保母" in normalized:
        return "保母費"

    return "其他"


def parse_expense_text(text: str) -> ExpenseEntry:
    global last_openai_expense_parse
    model = get_current_model()
    logger.info("Requesting OpenAI expense parse model=%s text=%s", model, text)
    started_at = datetime.now(timezone.utc)
    client = OpenAI()
    prompt = (
        "從以下使用者輸入提取一筆支出，僅輸出 JSON。\n"
        "date 若沒寫請根據今天自動補今天日期，格式 YYYY-MM-DD。\n"
        "time 若沒寫填 null。\n"
        "amount 是整數。\n"
        "currency 優先使用 TWD。\n"
        "category 必須是這些分類之一：餐飲、交通、購物、生活用品、醫療、娛樂、教育、旅遊、房租、水費、電費、瓦斯費、電話費、網路費、保險、信用卡、貸款、保母費、其他。\n"
        "merchant 可為 null。\n"
        "note 為摘要。\n"
        "confidence 介於 0 和 1。"
    )

    schema = {
        "type": "object",
        "properties": {
            "date": {"type": "string"},
            "time": {"type": ["string", "null"]},
            "amount": {"type": "integer"},
            "currency": {"type": "string", "enum": ["TWD", "USD", "JPY", "CNY", "EUR", "OTHER"]},
            "category": {"type": "string", "enum": EXPENSE_CATEGORY_VALUES},
            "merchant": {"type": ["string", "null"]},
            "note": {"type": ["string", "null"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["date", "time", "amount", "currency", "category", "merchant", "note", "confidence"],
        "additionalProperties": False,
    }

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "expense_entry",
                "schema": schema,
                "strict": True,
            },
        },
    )
    latency_ms = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
    log_usage("openai", "chat_completion", detail=str(model), success=True, latency_ms=latency_ms)
    content = str(response.output_text).strip()
    last_openai_expense_parse = {"input": text, "output": content}
    logger.info("OpenAI expense parse raw=%s", content)
    try:
        return ExpenseEntry.model_validate_json(content)
    except Exception as exc:
        logger.error("Failed to validate OpenAI expense parse: %s", content)
        raise OpenAIError(f"Invalid structured expense output: {exc}") from exc


def normalize_category(value: str) -> str:
    mapping = {
        "水電瓦斯": "水費",
        "生活": "生活用品",
        "日用品": "生活用品",
        "食材": "購物",
    }
    return mapping.get(value, value)


def row_to_expense_entry(row: sqlite3.Row) -> ExpenseEntry:
    return ExpenseEntry(
        date=str(row["date"]),
        time=str(row["expense_time"]) if row["expense_time"] else None,
        amount=int(row["amount"]),
        currency=str(row["currency"]),
        category=normalize_category(str(row["category"])),
        merchant=str(row["merchant"]) if row["merchant"] else None,
        note=str(row["note"]) if row["note"] else None,
        confidence=float(row["confidence"]),
    )


def parse_expense_text_fallback(text: str) -> ExpenseEntry:
    today = date.today().isoformat()
    amount = parse_amount(text)
    if amount is None:
        raise ValueError("Could not parse amount from text.")

    time_text = parse_time_fallback(text)
    return ExpenseEntry(
        date=parse_due_date(text) or today,
        time=time_text,
        amount=amount,
        currency="TWD",
        category=infer_category(text),
        merchant=None,
        note=text,
        confidence=0.90,
    )


def parse_intent_with_openai(text: str) -> IntentResult:
    prompt = (
        "判斷以下訊息意圖，僅輸出 JSON。\n"
        "intent 必須是 create、update、delete、summary、list、chat 其中之一。\n"
        "confidence 為 0 到 1，reason 為簡短中文說明。"
    )
    schema = {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": ["create", "update", "delete", "summary", "list", "chat"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": ["string", "null"]},
        },
        "required": ["intent", "confidence", "reason"],
        "additionalProperties": False,
    }
    model = get_current_model()
    started_at = datetime.now(timezone.utc)
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "intent_result",
                "schema": schema,
                "strict": True,
            },
        },
    )
    latency_ms = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
    log_usage("openai", "chat_completion", detail=get_current_model(), success=True, latency_ms=latency_ms)
    content = str(response.output_text).strip()
    return IntentResult.model_validate_json(content)


def classify_intent(text: str) -> IntentResult:
    try:
        return parse_intent_with_openai(text)
    except Exception:
        logger.exception("OpenAI intent classification failed; using local fallback for text: %s", text)
        return local_intent_fallback(text)


def build_reply_text(expense: ExpenseEntry, expense_id: int | None = None) -> str:
    lines = ["已記帳"]
    lines.append(f"日期：{expense.date}")
    if expense.time:
        lines.append(f"時間：{expense.time}")
    lines.append(f"金額：{expense.currency} {expense.amount}")
    lines.append(f"分類：{expense.category}")
    lines.append(f"店家/對象：{expense.merchant or '未指定'}")
    lines.append(f"信心：{expense.confidence:.2f}")
    lines.append(f"備註：{expense.note or '無'}")
    if expense_id is not None:
        lines.append(f"代號：{expense_id}")
    return "\n".join(lines)


def normalize_duplicate_label(merchant: str | None, note: str | None, category: str, raw_text: str) -> str:
    for value in (merchant, note):
        if value:
            return str(value)
    return str(raw_text).strip() or category


def count_similar_expenses(expense: ExpenseEntry, line_user_id: str | None) -> int:
    note_label = normalize_duplicate_label(expense.merchant, expense.note, expense.category, expense.note or expense.category)
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM expenses
            WHERE date = ?
              AND COALESCE(expense_time, '') = COALESCE(?, '')
              AND amount = ?
              AND currency = ?
              AND category = ?
              AND COALESCE(merchant, '') = COALESCE(?, '')
              AND COALESCE(note, '') = COALESCE(?, '')
              AND (? IS NULL OR line_user_id = ?)
            """,
            (
                expense.date,
                expense.time,
                expense.amount,
                expense.currency,
                expense.category,
                expense.merchant,
                expense.note,
                line_user_id,
                line_user_id,
            ),
        ).fetchone()
    return int(row["count"] if row else 0)


def get_pending_expense_key(line_user_id: str | None) -> str:
    return line_user_id or "_global"


def is_simple_small_expense(expense: ExpenseEntry, raw_text: str) -> bool:
    if expense.amount > 1000:
        return False
    if expense.category == "其他":
        return False
    if expense.confidence < 0.85:
        return False
    if is_question_text(raw_text):
        return False
    return True


def should_confirm_expense(expense: ExpenseEntry, raw_text: str, route: ActionRoute) -> bool:
    if is_question_text(raw_text):
        return False
    if route.confidence < 0.85:
        return True
    if expense.category == "其他":
        return True
    if expense.amount >= 100000:
        return True
    if expense.confidence < 0.85:
        return True
    return not is_simple_small_expense(expense, raw_text)


def build_pending_expense_reply(expense: ExpenseEntry) -> str:
    return (
        "我先幫你整理成這筆記帳，請確認：\n\n"
        f"日期：{expense.date}\n"
        f"金額：{expense.currency} {expense.amount}\n"
        f"分類：{expense.category}\n"
        f"店家/對象：{expense.merchant or '未指定'}\n"
        f"備註：{expense.note or '無'}\n\n"
        "如果沒問題，請回覆：確認記帳\n"
        "若不要記，請回覆：取消"
    )


def save_home_task_draft(
    chat_id: str,
    actor_user_id: str | None,
    title: str,
    item_key: str | None,
    category: str,
    scheduled_date: str | None,
    scheduled_time: str | None,
    original_text: str,
) -> None:
    now_iso = taipei_now().isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO home_task_drafts (
                chat_id, actor_user_id, title, item_key, category,
                scheduled_date, scheduled_time, original_text, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                actor_user_id = excluded.actor_user_id,
                title = excluded.title,
                item_key = excluded.item_key,
                category = excluded.category,
                scheduled_date = excluded.scheduled_date,
                scheduled_time = excluded.scheduled_time,
                original_text = excluded.original_text,
                created_at = excluded.created_at
            """,
            (chat_id, actor_user_id, title, item_key, category, scheduled_date, scheduled_time, original_text, now_iso),
        )
        conn.commit()


def get_home_task_draft(chat_id: str) -> HomeTaskDraft | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM home_task_drafts WHERE chat_id = ?", (chat_id,)).fetchone()
    if row is None:
        return None
    return HomeTaskDraft(
        chat_id=str(row["chat_id"]),
        actor_user_id=str(row["actor_user_id"]) if row["actor_user_id"] else None,
        title=str(row["title"]),
        item_key=str(row["item_key"]) if row["item_key"] else None,
        category=str(row["category"]),
        scheduled_date=str(row["scheduled_date"]) if row["scheduled_date"] else None,
        scheduled_time=str(row["scheduled_time"]) if row["scheduled_time"] else None,
        original_text=str(row["original_text"]),
        created_at=str(row["created_at"]),
    )


def delete_home_task_draft(chat_id: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM home_task_drafts WHERE chat_id = ?", (chat_id,))
        conn.commit()


def build_home_task_created_reply(task: HomeTaskRecord) -> str:
    completion_example = f"已完成{task.title}"
    return (
        "已建立家庭事項\n\n"
        f"事項：{task.title}\n"
        f"日期：{format_task_date_with_weekday(task.scheduled_date)}\n\n"
        f"完成後可以回覆「已完成」或「{completion_example}」。"
    )


def row_to_home_task(row: sqlite3.Row) -> HomeTaskRecord:
    return HomeTaskRecord(
        id=int(row["id"]),
        actor_user_id=str(row["actor_user_id"]) if row["actor_user_id"] else None,
        chat_id=str(row["chat_id"]),
        message_id=str(row["message_id"]) if row["message_id"] else None,
        title=str(row["title"]),
        item_key=str(row["item_key"]) if row["item_key"] else None,
        category=str(row["category"]),
        scheduled_date=str(row["scheduled_date"]) if row["scheduled_date"] else None,
        scheduled_time=str(row["scheduled_time"]) if row["scheduled_time"] else None,
        status=str(row["status"]),
        completed_at=str(row["completed_at"]) if row["completed_at"] else None,
        completion_text=str(row["completion_text"]) if row["completion_text"] else None,
        last_reminded_at=str(row["last_reminded_at"]) if row["last_reminded_at"] else None,
        raw_text=str(row["raw_text"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def create_home_task_record(
    actor_user_id: str | None,
    chat_id: str,
    raw_text: str,
    message_id: str | None,
    title: str,
    item_key: str | None,
    category: str,
    scheduled_date: str | None,
    scheduled_time: str | None,
) -> HomeTaskRecord:
    now_iso = taipei_now().isoformat()
    with get_db() as conn:
        if message_id:
            existing = conn.execute(
                "SELECT * FROM home_tasks WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if existing:
                return row_to_home_task(existing)
        cursor = conn.execute(
            """
            INSERT INTO home_tasks (
                actor_user_id, chat_id, message_id, title, item_key, category,
                scheduled_date, scheduled_time, status, completed_at, completion_text,
                last_reminded_at, raw_text, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL, NULL, NULL, ?, ?, ?)
            """,
            (
                actor_user_id,
                chat_id,
                message_id,
                title,
                item_key,
                category,
                scheduled_date,
                scheduled_time,
                raw_text,
                now_iso,
                now_iso,
            ),
        )
        task_id = int(cursor.lastrowid)
        conn.commit()
        row = conn.execute("SELECT * FROM home_tasks WHERE id = ?", (task_id,)).fetchone()
    log_sql_change("home_task", "insert", task_id, title, actor_user_id)
    log_usage("app", "home_task_create", detail=title, success=True)
    if row is None:
        raise RuntimeError("Failed to load created home task.")
    return row_to_home_task(row)


def create_home_task_from_route(
    raw_text: str,
    actor_user_id: str | None,
    chat_id: str,
    message_id: str | None,
    route: ActionRoute,
) -> str:
    title = route.task_title or extract_home_task_title_from_create(raw_text)
    scheduled_date = route.scheduled_date or parse_due_date(raw_text)
    scheduled_time = route.scheduled_time or parse_home_task_time(raw_text)
    if not title or not scheduled_date:
        return "我有猜到你可能想建立家庭事項，但日期或事項內容還不夠明確。"
    item_key = route.task_item_key or extract_home_task_item_key(title)
    category = route.task_category or infer_home_task_category(title)
    if route.task_requires_confirmation:
        save_home_task_draft(chat_id, actor_user_id, title, item_key, category, scheduled_date, scheduled_time, raw_text)
        return f"要幫你建立 {scheduled_date} 的「{title}」家庭事項嗎？\n請回覆「要」或「不要」。"
    task = create_home_task_record(actor_user_id, chat_id, raw_text, message_id, title, item_key, category, scheduled_date, scheduled_time)
    return build_home_task_created_reply(task)


def match_home_task(row: sqlite3.Row, lookup_term: str | None) -> bool:
    if not lookup_term:
        return True
    available_keys = set(row.keys())
    searchable = " ".join(
        str(row[key] or "")
        for key in ("title", "item_key", "raw_text")
        if key in available_keys
    )
    return lookup_term in searchable


def build_home_task_choice_reply(rows: list[sqlite3.Row]) -> str:
    lines = [
        "目前有多個尚未完成的家庭事項，請問完成的是哪一項？",
        "",
    ]
    for index, row in enumerate(rows[:5], start=1):
        lines.append(f"{index}. {row['title']}")
    lines.append("")
    lines.append("請回覆事項名稱，例如「已完成換地漏」。")
    return "\n".join(lines)


def find_home_task_candidates(
    chat_id: str,
    lookup_term: str | None,
    *,
    status: str = "pending",
    scheduled_date: str | None = None,
) -> list[sqlite3.Row]:
    with get_db() as conn:
        sql = """
            SELECT *
            FROM home_tasks
            WHERE chat_id = ?
              AND status = ?
        """
        params: list[object] = [chat_id, status]
        if scheduled_date:
            sql += " AND scheduled_date = ?"
            params.append(scheduled_date)
        sql += " ORDER BY COALESCE(scheduled_date, '9999-12-31') ASC, id DESC"
        rows = conn.execute(
            sql,
            tuple(params),
        ).fetchall()
    return [row for row in rows if match_home_task(row, lookup_term)]


def complete_home_task(raw_text: str, actor_user_id: str | None, chat_id: str) -> str:
    lookup_term = extract_home_task_lookup_term(raw_text)
    candidates = find_home_task_candidates(chat_id, lookup_term, status="pending")
    if not lookup_term:
        with get_db() as conn:
            reminded = conn.execute(
                """
                SELECT *
                FROM home_tasks
                WHERE chat_id = ?
                  AND status = 'pending'
                  AND last_reminded_at IS NOT NULL
                ORDER BY last_reminded_at DESC, id DESC
                LIMIT 5
                """,
                (chat_id,),
            ).fetchall()
            if len(reminded) == 1:
                candidates = list(reminded)
            elif len(reminded) > 1:
                return build_home_task_choice_reply(list(reminded))
            else:
                due_rows = conn.execute(
                    """
                    SELECT *
                    FROM home_tasks
                    WHERE chat_id = ?
                      AND status = 'pending'
                    ORDER BY COALESCE(scheduled_date, '9999-12-31') ASC, id DESC
                    LIMIT 5
                    """,
                    (chat_id,),
                ).fetchall()
                if len(due_rows) == 1:
                    candidates = list(due_rows)
                elif len(due_rows) > 1:
                    return build_home_task_choiceReply(list(due_rows))
    if not candidates:
        return "目前找不到可完成的家庭事項。"
    if len(candidates) > 1:
        return build_home_task_choice_reply(candidates)

    row = candidates[0]
    now_iso = taipei_now().isoformat()
    completed_date = taipei_now().date().isoformat()
    with get_db() as conn:
        conn.execute(
            """
            UPDATE home_tasks
            SET status = 'completed',
                completed_at = ?,
                completion_text = ?,
                updated_at = ?,
                actor_user_id = COALESCE(?, actor_user_id)
            WHERE id = ?
            """,
            (now_iso, raw_text, now_iso, actor_user_id, int(row["id"])),
        )
        conn.commit()
    log_sql_change("home_task", "update", int(row["id"]), str(row["title"]), actor_user_id)
    log_usage("app", "home_task_complete", detail=str(row["title"]), success=True)
    if row["scheduled_date"] and str(row["scheduled_date"]) != completed_date:
        return (
            "已完成家庭事項\n\n"
            f"事項：{row['title']}\n"
            f"實際完成日期：{completed_date}\n"
            f"原定日期：{row['scheduled_date']}"
        )
    return (
        "已完成家庭事項\n\n"
        f"事項：{row['title']}\n"
        f"完成日期：{completed_date}"
    )


def cancel_home_task(raw_text: str, actor_user_id: str | None, chat_id: str) -> str:
    lookup_term = extract_home_task_lookup_term(raw_text)
    scheduled_date = parse_due_date(raw_text)
    candidates = find_home_task_candidates(chat_id, lookup_term, status="pending", scheduled_date=scheduled_date)
    if not candidates:
        return "目前找不到可取消的家庭事項。"
    if len(candidates) > 1:
        return build_home_task_choice_reply(candidates)
    row = candidates[0]
    now_iso = taipei_now().isoformat()
    with get_db() as conn:
        conn.execute(
            """
            UPDATE home_tasks
            SET status = 'cancelled',
                updated_at = ?,
                actor_user_id = COALESCE(?, actor_user_id)
            WHERE id = ?
            """,
            (now_iso, actor_user_id, int(row["id"])),
        )
        conn.commit()
    log_sql_change("home_task", "update", int(row["id"]), str(row["title"]), actor_user_id)
    log_usage("app", "home_task_cancel", detail=str(row["title"]), success=True)
    return f"已取消家庭事項\n\n事項：{row['title']}"


def get_home_task_date_range(raw_text: str) -> tuple[str | None, str | None]:
    today = taipei_today()
    normalized = re.sub(r"\s+", "", raw_text)
    if any(keyword in normalized for keyword in ("這週", "這周", "這禮拜", "這星期", "本週", "本周", "本禮拜", "本星期")):
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        return start.isoformat(), end.isoformat()
    if any(keyword in normalized for keyword in ("下週", "下周", "下禮拜", "下星期", "下個星期")):
        start = today - timedelta(days=today.weekday()) + timedelta(days=7)
        end = start + timedelta(days=6)
        return start.isoformat(), end.isoformat()
    target_date = parse_due_date(raw_text)
    if target_date:
        return target_date, target_date
    return None, None


def query_home_tasks(raw_text: str, chat_id: str) -> str:
    lookup_term = extract_home_task_lookup_term(raw_text)
    if "完成了嗎" in raw_text and lookup_term:
        pending = find_home_task_candidates(chat_id, lookup_term, status="pending")
        if pending:
            row = pending[0]
            return f"「{row['title']}」還沒完成，預定日期是 {row['scheduled_date']}。"
        with get_db() as conn:
            completed = conn.execute(
                """
                SELECT *
                FROM home_tasks
                WHERE chat_id = ?
                  AND status = 'completed'
                ORDER BY completed_at DESC, id DESC
                """,
                (chat_id,),
            ).fetchall()
        for row in completed:
            if match_home_task(row, lookup_term):
                return f"「{row['title']}」已完成，完成日期是 {str(row['completed_at'])[:10]}。"
        return f"目前查不到 {lookup_term} 的家庭事項。"

    start_date, end_date = get_home_task_date_range(raw_text)
    sql = """
        SELECT *
        FROM home_tasks
        WHERE chat_id = ?
          AND status = 'pending'
    """
    params: list[object] = [chat_id]
    if start_date and end_date:
        sql += " AND scheduled_date BETWEEN ? AND ?"
        params.extend([start_date, end_date])
    sql += " ORDER BY COALESCE(scheduled_date, '9999-12-31') ASC, id ASC"
    with get_db() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    if lookup_term:
        rows = [row for row in rows if match_home_task(row, lookup_term)]
    if not rows:
        lines = ["查無家庭事項"]
        if start_date:
            lines.append(f"日期：{start_date}" if start_date == end_date else f"日期：{start_date}~{end_date}")
        return "\n".join(lines)
    lines = ["尚未完成的家庭事項", ""]
    for row in rows[:20]:
        lines.append(f"{row['scheduled_date'] or '未排日期'}　{row['title']}")
    return "\n".join(lines)


def parse_home_task_history_limit(raw_text: str) -> int:
    if "最近三次" in raw_text or "三次" in raw_text:
        return 3
    return 1 if any(keyword in raw_text for keyword in ("上次", "最近一次")) else 10


def query_home_task_history(raw_text: str, chat_id: str) -> str:
    lookup_term = extract_home_task_lookup_term(raw_text)
    if not lookup_term:
        return "請告訴我要查哪一件家庭事項，例如「上次換地漏是什麼時候？」。"
    limit = parse_home_task_history_limit(raw_text)
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT title, item_key, scheduled_date, completed_at
            FROM home_tasks
            WHERE chat_id = ?
              AND status = 'completed'
            ORDER BY completed_at DESC, id DESC
            """,
            (chat_id,),
        ).fetchall()
    rows = [row for row in rows if match_home_task(row, lookup_term)]
    if not rows:
        return f"目前查不到已完成的{lookup_term}記錄。"
    if "幾次" in raw_text:
        return f"{lookup_term} 目前查到已完成 {len(rows)} 次。"
    if limit == 1 and not any(keyword in raw_text for keyword in ("紀錄", "列出")):
        row = rows[0]
        completed_date = str(row["completed_at"])[:10]
        if row["scheduled_date"] and str(row["scheduled_date"]) != completed_date:
            return f"上次{lookup_term}是在 {completed_date}。\n當時原定日期是 {row['scheduled_date']}。"
        return f"上次{lookup_term}是在 {completed_date}。"
    lines = [f"{lookup_term} 的完成紀錄", ""]
    for row in rows[:limit]:
        completed_date = str(row["completed_at"])[:10]
        if row["scheduled_date"] and str(row["scheduled_date"]) != completed_date:
            lines.append(f"{completed_date}（原定 {row['scheduled_date']}） {row['title']}")
        else:
            lines.append(f"{completed_date} {row['title']}")
    return "\n".join(lines)


async def send_home_task_reminders() -> None:
    now_local = taipei_now()
    today_text = now_local.date().isoformat()
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM home_tasks
            WHERE status = 'pending'
              AND scheduled_date = ?
            ORDER BY COALESCE(scheduled_time, '23:59') ASC, id ASC
            """,
            (today_text,),
        ).fetchall()
    for row in rows:
        last_reminded_at = str(row["last_reminded_at"] or "")
        if last_reminded_at[:10] == today_text:
            continue
        scheduled_time = row["scheduled_time"]
        if scheduled_time:
            scheduled_dt = datetime.fromisoformat(f"{today_text}T{scheduled_time}:00").replace(tzinfo=TAIPEI_TZ)
            if now_local < scheduled_dt:
                continue
        elif now_local.hour < HOME_TASK_DEFAULT_REMINDER_HOUR:
            continue
        message = (
            "家庭事項提醒\n\n"
            f"今天要{row['title']}。\n\n"
            "完成後可以回覆：\n"
            "已完成\n"
            "或\n"
            f"已完成{row['title']}"
        )
        try:
            await push_line_message(str(row["chat_id"]), message)
        except Exception:
            logger.exception("send_home_task_reminders push failed task_id=%s", row["id"])
            continue
        now_iso = taipei_now().isoformat()
        with get_db() as conn:
            conn.execute(
                """
                UPDATE home_tasks
                SET last_reminded_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (now_iso, now_iso, int(row["id"])),
            )
            conn.commit()


def is_delete_intent(text: str) -> bool:
    return any(keyword in text.lower() for keyword in ("刪除", "删除", "取消", "移除", "delete"))


def is_delete_all_intent(text: str) -> bool:
    return is_delete_intent(text) and any(keyword in text for keyword in ("所有", "全部", "全刪", "清空"))


def is_confirm_delete_all_intent(text: str) -> bool:
    return any(keyword in text for keyword in ("確認", "確定")) and is_delete_all_intent(text)


def is_confirm_delete_intent(text: str) -> bool:
    return re.sub(r"\s+", "", text) == "88"


def is_update_intent(text: str) -> bool:
    return any(keyword in text.lower() for keyword in ("修改", "改成", "更正", "更新", "edit", "update"))


def is_summary_intent(text: str) -> bool:
    return any(
        keyword in text
        for keyword in ("總共", "總花費", "花多少", "多少錢", "平均", "最高", "最低", "統計", "比例", "占比")
    )


def is_list_intent(text: str) -> bool:
    return any(keyword in text for keyword in ("明細", "清單", "列出", "有哪些", "列表"))


def get_query_dates(text: str) -> list[str]:
    today = date.today()
    dates: list[date] = []

    if any(keyword in text for keyword in ("最近兩天", "這兩天", "昨天跟今天", "昨天和今天")):
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


def get_query_category(text: str) -> str | None:
    normalized = text.lower().replace(" ", "")

    if any(keyword in normalized for keyword in ("早餐", "午餐", "晚餐", "飲料", "咖啡", "宵夜", "餐飲")):
        return "餐飲"
    if any(keyword in normalized for keyword in ("交通", "加油", "停車", "捷運", "高鐵", "台鐵", "公車", "uber", "計程車", "火車")):
        return "交通"
    if any(keyword in normalized for keyword in ("購物", "衣服", "鞋子", "治裝", "食材")):
        return "購物"
    if any(keyword in normalized for keyword in ("生活用品", "日用品", "全聯", "家樂福", "超市", "香蕉", "芭樂", "蓮霧", "水果")):
        return "生活用品"
    if any(keyword in normalized for keyword in ("醫療", "看醫生", "牙醫", "掛號", "藥")):
        return "醫療"

    return None


def parse_chart_request(raw_text: str) -> tuple[bool, str | None]:
    wants_chart = "圖" in raw_text
    if not wants_chart:
        return False, None
    normalized = raw_text.lower()
    if "折線圖" in raw_text or "line" in normalized:
        return True, "line"
    if "長條圖" in raw_text or "bar" in normalized:
        return True, "bar"
    if any(keyword in raw_text for keyword in ("圓餅圖", "pie", "餅圖")):
        return True, "pie"
    return True, None


def parse_due_specific_day(text: str) -> str | None:
    return parse_due_date(text)


def parse_expense_query_with_openai(text: str) -> ExpenseQuery:
    global last_openai_expense_parse
    system_prompt = (
        "你是台灣家庭記帳 Bot 的支出查詢解析器。\n"
        "任務：把使用者的自然語言查詢轉成固定 JSON。\n"
        "只輸出 JSON，不要回答。\n"
        "date_range_type 可用：today、yesterday、last_2_days、this_week、this_month、last_6_months、specific_months、custom、unspecified。\n"
        "mode 可用：list_detail、aggregate、grouped_aggregate。\n"
        "metric 可用：sum、count、avg、max、min。\n"
        "group_by 可用：day、week、month、category、merchant。\n"
        "chart_type 可用：none、line、bar、pie。\n"
        "若使用者要圖表，wants_chart=true，chart_type 依句意選擇。\n"
        "比例/占比查詢時 include_ratio=true。\n"
        "分類請用：餐飲、交通、購物、生活用品、醫療、娛樂、教育、旅遊、房租、水費、電費、瓦斯費、電話費、網路費、保險、信用卡、貸款、保母費、其他。"
    )
    schema = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["list_detail", "aggregate", "grouped_aggregate"]},
            "metric": {"type": "string", "enum": ["sum", "count", "avg", "max", "min"]},
            "group_by": {"type": "array", "items": {"type": "string", "enum": ["day", "week", "month", "category", "merchant"]}},
            "date_range_type": {"type": "string", "enum": ["today", "yesterday", "last_2_days", "this_week", "this_month", "last_6_months", "specific_months", "custom", "unspecified"]},
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
            "category": {"type": ["string", "null"], "enum": [None, *EXPENSE_CATEGORY_VALUES]},
            "include_categories": {"type": "array", "items": {"type": "string", "enum": EXPENSE_CATEGORY_VALUES}},
            "exclude_categories": {"type": "array", "items": {"type": "string", "enum": EXPENSE_CATEGORY_VALUES}},
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
            "ratio_denominator": {"type": "string", "enum": ["all_expenses", "filtered_expenses", "included_expenses", "none"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            "sort_by": {"type": "string", "enum": ["date", "time", "amount", "created_at", "group", "total", "count"]},
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
    model = get_current_model()
    started_at = datetime.now(timezone.utc)
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "expense_query",
                "schema": schema,
                "strict": True,
            },
        },
    )
    latency_ms = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
    log_usage("openai", "chat_completion", detail=get_current_model(), success=True, latency_ms=latency_ms)
    content = str(response.output_text).strip()
    last_openai_expense_parse = {"input": text, "output": content}
    return ExpenseQuery.model_validate_json(content)


def extract_expense_chart_request(
    raw_text: str,
    precomputed_group_by: list[ExpenseGroupBy] | None = None,
    include_ratio: bool | None = None,
) -> tuple[str, bool, ChartType]:
    wants_chart, explicit_chart_type = parse_chart_request(raw_text)
    clean_text = raw_text
    if not wants_chart:
        return clean_text, False, "none"

    patterns = [
        r"\+\s*折線圖",
        r"\+\s*長條圖",
        r"\+\s*圓餅圖",
        r"\+\s*圖表",
        r"，\s*請用折線圖",
        r"，\s*請用長條圖",
        r"，\s*請用圓餅圖",
        r"，\s*畫折線圖",
        r"，\s*畫長條圖",
        r"，\s*畫圓餅圖",
        r"折線圖",
        r"長條圖",
        r"圓餅圖",
        r"圖表",
    ]
    for pattern in patterns:
        clean_text = re.sub(pattern, "", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"\s+", " ", clean_text).strip(" +，,。")

    group_by = precomputed_group_by if precomputed_group_by is not None else parse_expense_group_by(clean_text)
    ratio_flag = include_ratio if include_ratio is not None else is_ratio_query(clean_text)
    chart_type = explicit_chart_type or infer_expense_chart_type(group_by, ratio_flag)
    return clean_text, True, chart_type


def parse_expense_group_by(raw_text: str) -> list[ExpenseGroupBy]:
    normalized = re.sub(r"\s+", "", raw_text)
    group_by: list[ExpenseGroupBy] = []
    if any(keyword in normalized for keyword in ("每天", "每日", "逐日", "一天一天", "每天花多少")):
        group_by.append("day")
    if any(keyword in normalized for keyword in ("每週", "逐週")):
        group_by.append("week")
    if any(keyword in normalized for keyword in ("每月", "每個月", "逐月", "各月")):
        group_by.append("month")
    if any(keyword in normalized for keyword in ("各類別", "分類別", "按分類", "依分類", "哪個分類", "分類占比", "各分類")):
        group_by.append("category")
    if any(keyword in normalized for keyword in ("各店家", "依店家", "按店家", "商家", "店家別")):
        group_by.append("merchant")
    return group_by


def parse_expense_metric(raw_text: str) -> ExpenseMetric:
    normalized = re.sub(r"\s+", "", raw_text)
    if "平均" in normalized:
        return "avg"
    if any(keyword in normalized for keyword in ("最高", "最大", "最貴")):
        return "max"
    if any(keyword in normalized for keyword in ("最低", "最小", "最便宜")):
        return "min"
    if any(keyword in normalized for keyword in ("幾筆", "筆數", "次數")):
        return "count"
    return "sum"


def infer_expense_query_mode(raw_text: str, group_by: list[ExpenseGroupBy]) -> ExpenseQueryMode:
    if group_by:
        return "grouped_aggregate"
    if is_list_intent(raw_text):
        return "list_detail"
    return "aggregate"


def extract_expense_month_sequence(raw_text: str) -> list[int]:
    matches = [int(value) for value in re.findall(r"(?<!\d)(\d{1,2})\s*月", raw_text)]
    unique: list[int] = []
    for month in matches:
        if 1 <= month <= 12 and month not in unique:
            unique.append(month)
    return unique


def default_expense_query_dates(date_range_type: ExpenseDateRangeType, raw_text: str) -> list[DateRange]:
    today = date.today()
    if date_range_type == "today":
        return [DateRange(start_date=today.isoformat(), end_date=today.isoformat())]
    if date_range_type == "yesterday":
        target = today - timedelta(days=1)
        return [DateRange(start_date=target.isoformat(), end_date=target.isoformat())]
    if date_range_type == "last_2_days":
        start = today - timedelta(days=1)
        return [DateRange(start_date=start.isoformat(), end_date=today.isoformat())]
    if date_range_type == "this_week":
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        return [DateRange(start_date=start.isoformat(), end_date=end.isoformat())]
    if date_range_type == "this_month":
        start = today.replace(day=1)
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        end = next_month - timedelta(days=1)
        return [DateRange(start_date=start.isoformat(), end_date=end.isoformat())]
    if date_range_type == "last_6_months":
        months = []
        for offset in range(5, -1, -1):
            month = today.month - offset
            year = today.year
            while month <= 0:
                month += 12
                year -= 1
            start = date(year, month, 1)
            if month == 12:
                next_start = date(year + 1, 1, 1)
            else:
                next_start = date(year, month + 1, 1)
            months.append(DateRange(start_date=start.isoformat(), end_date=(next_start - timedelta(days=1)).isoformat()))
        return months
    if date_range_type == "specific_months":
        months = extract_expense_month_sequence(raw_text)
        ranges: list[DateRange] = []
        for month in months:
            start = date(today.year, month, 1)
            if month == 12:
                next_start = date(today.year + 1, 1, 1)
            else:
                next_start = date(today.year, month + 1, 1)
            ranges.append(DateRange(start_date=start.isoformat(), end_date=(next_start - timedelta(days=1)).isoformat()))
        return ranges
    return []


def is_ratio_query(raw_text: str) -> bool:
    normalized = re.sub(r"\s+", "", raw_text)
    return any(keyword in normalized for keyword in ("占比", "比例", "百分比", "比重"))


def parse_expense_query_fallback(text: str) -> ExpenseQuery:
    preliminary_group_by = parse_expense_group_by(text)
    clean_text, wants_chart, chart_type = extract_expense_chart_request(text, preliminary_group_by)
    if wants_chart:
        preliminary_group_by = parse_expense_group_by(clean_text)
    date_range_type = infer_date_range_type(clean_text)
    group_by = parse_expense_group_by(clean_text)
    metric = parse_expense_metric(clean_text)
    include_ratio = is_ratio_query(clean_text) or group_by == ["category"]
    mode = infer_expense_query_mode(clean_text, group_by)
    category = get_query_category(clean_text)
    include_categories = [category] if category else []
    exclude_categories = []
    exclude_keywords = extract_expense_exclude_keywords(clean_text)
    include_keywords = []
    keywords = []
    merchant = None
    min_amount = None
    max_amount = None
    aggregation: ExpenseAggregation = "sum"
    if mode == "list_detail":
        aggregation = "list"
    elif metric == "count":
        aggregation = "count"
    elif metric == "max":
        aggregation = "top"
    elif include_ratio:
        aggregation = "category_breakdown" if group_by == ["category"] else "sum_and_ratio"
    ratio_denominator: RatioDenominator = "none"
    if include_ratio:
        ratio_denominator = "filtered_expenses" if exclude_keywords or exclude_categories else "all_expenses"
    sort_by, sort_direction = parse_expense_sort(clean_text)
    date_ranges = default_expense_query_dates(date_range_type, clean_text)
    if specific_day := extract_expense_specific_day(clean_text):
        date_range_type = "custom"
        date_ranges = [specific_day]
    return ExpenseQuery(
        mode=mode,
        metric=metric,
        group_by=group_by,
        date_range_type=date_range_type,
        date_ranges=date_ranges,
        category=category,
        include_categories=include_categories,
        exclude_categories=exclude_categories,
        merchant=merchant,
        include_keywords=include_keywords,
        keywords=keywords,
        exclude_keywords=exclude_keywords,
        min_amount=min_amount,
        max_amount=max_amount,
        include_ratio=include_ratio,
        wants_chart=wants_chart,
        chart_type=chart_type,
        aggregation=aggregation,
        ratio_denominator=ratio_denominator,
        limit=15,
        sort_by=sort_by,
        sort_direction=sort_direction,
        confidence=0.85,
        reason="local fallback expense query",
    )


def infer_date_range_type(raw_text: str) -> ExpenseDateRangeType:
    normalized = re.sub(r"\s+", "", raw_text)
    if any(keyword in normalized for keyword in ("今天",)):
        return "today"
    if any(keyword in normalized for keyword in ("昨天",)):
        return "yesterday"
    if any(keyword in normalized for keyword in ("最近兩天", "這兩天")):
        return "last_2_days"
    if any(keyword in normalized for keyword in ("本週", "這週", "這禮拜", "本周", "這周", "本星期", "這星期")):
        return "this_week"
    if any(keyword in normalized for keyword in ("這個月", "本月", "七月", "8月", "9月", "10月", "11月", "12月")):
        month_matches = extract_expense_month_sequence(raw_text)
        return "specific_months" if len(month_matches) > 1 else "this_month"
    if any(keyword in normalized for keyword in ("最近半年", "近半年", "六個月")):
        return "last_6_months"
    if extract_expense_month_sequence(raw_text):
        return "specific_months"
    if parse_due_specific_day(raw_text):
        return "custom"
    return "this_month"


def extract_expense_exclude_keywords(text: str) -> list[str]:
    matches = re.findall(r"(?:排除|不要計算|扣掉|不算)([^，,。\s]+)", text)
    keywords: list[str] = []
    for match in matches:
        cleaned = match.strip()
        if cleaned and cleaned not in keywords:
            keywords.append(cleaned)
    return keywords


def extract_expense_specific_day(raw_text: str) -> DateRange | None:
    if any(keyword in raw_text for keyword in ("今天", "昨天", "這週", "本週", "這個月", "本月", "最近")):
        return None
    target_date = parse_due_specific_day(raw_text)
    if not target_date:
        return None
    return DateRange(start_date=target_date, end_date=target_date)


def parse_expense_sort(raw_text: str) -> tuple[ExpenseSortBy, SortDirection]:
    normalized = re.sub(r"\s+", "", raw_text)
    if any(keyword in normalized for keyword in ("由舊到新", "升冪")):
        direction: SortDirection = "asc"
    else:
        direction = "desc"
    if any(keyword in normalized for keyword in ("金額", "多少元")):
        return "amount", direction
    if any(keyword in normalized for keyword in ("建立時間", "新增時間")):
        return "created_at", direction
    return "date", direction


def parse_expense_chart_type(raw_text: str, group_by: list[ExpenseGroupBy], include_ratio: bool) -> tuple[bool, ChartType]:
    return extract_expense_chart_request(raw_text, group_by, include_ratio)[1:]


def infer_expense_chart_type(group_by: list[ExpenseGroupBy], include_ratio: bool) -> ChartType:
    if include_ratio or group_by == ["category"]:
        return "pie"
    if group_by in (["day"], ["week"], ["month"]):
        return "line"
    return "bar"


def normalize_expense_category(value: str | None) -> Category | None:
    if value is None:
        return None
    normalized = normalize_category(str(value))
    return normalized if normalized in EXPENSE_CATEGORY_VALUES or normalized in {"餐飲", "交通", "購物", "生活用品", "醫療", "娛樂", "教育", "旅遊", "房租", "水費", "電費", "瓦斯費", "電話費", "網路費", "保險", "信用卡", "貸款", "保母費", "其他"} else None


def parse_expense_query(text: str) -> ExpenseQuery:
    preliminary_group_by = parse_expense_group_by(text)
    clean_text, wants_chart, chart_type = extract_expense_chart_request(text, preliminary_group_by)
    if wants_chart:
        preliminary_group_by = parse_expense_group_by(clean_text)
    try:
        query = parse_expense_query_with_openai(clean_text)
    except Exception:
        logger.exception("OpenAI expense query parsing failed; using local fallback for text: %s", clean_text)
        query = parse_expense_query_fallback(clean_text)
        text_group_by = parse_expense_group_by(clean_text)
        text_metric = parse_expense_metric(clean_text)
    else:
        text_group_by = parse_expense_group_by(clean_text)
        text_metric = parse_expense_metric(clean_text)
        if text_group_by:
            query.group_by = text_group_by
            query.mode = "grouped_aggregate"
        if query.mode != "grouped_aggregate":
            query.metric = text_metric
        query.include_ratio = query.include_ratio or is_ratio_query(clean_text) or query.group_by == ["category"]
        if query.include_ratio and query.aggregation in {"sum", "top"}:
            query.aggregation = "category_breakdown" if query.group_by == ["category"] else "sum_and_ratio"
        if query.include_ratio and query.ratio_denominator == "none":
            query.ratio_denominator = "filtered_expenses" if query.exclude_keywords or query.exclude_categories else "all_expenses"
        if query.category:
            normalized_category = normalize_expense_category(query.category)
            if normalized_category is not None:
                query.category = normalized_category
        query.exclude_keywords = extract_expense_exclude_keywords(clean_text) or query.exclude_keywords
        if text_metric == "count":
            query.aggregation = "count"
        elif text_metric == "max":
            query.aggregation = "top"
        elif query.mode == "list_detail":
            query.aggregation = "list"
        query.sort_by, query.sort_direction = parse_expense_sort(clean_text)
        if query.date_range_type == "unspecified":
            query.date_range_type = infer_date_range_type(clean_text)
            query.date_ranges = default_expense_query_dates(query.date_range_type, clean_text)
        if specific_day := extract_expense_specific_day(clean_text):
            query.date_range_type = "custom"
            query.date_ranges = [specific_day]
    query.wants_chart = wants_chart or query.wants_chart
    if query.wants_chart:
        query.chart_type = chart_type
    elif query.chart_type == "none":
        query.chart_type = "none"

    category = get_query_category(clean_text)
    if category is not None:
        query.category = category
    if not query.date_ranges:
        query.date_ranges = default_expense_query_dates(query.date_range_type, clean_text)
    return query


def build_expense_where_clauses(
    query: ExpenseQuery,
    line_user_id: str | None,
) -> tuple[str, list[object]]:
    clauses = ["1 = 1"]
    params: list[object] = []
    household_scope_user_id = get_household_read_scope_line_user_id(line_user_id)
    if household_scope_user_id is not None:
        clauses.append("line_user_id = ?")
        params.append(household_scope_user_id)
    if query.date_ranges:
        range_clauses = []
        for item in query.date_ranges:
            range_clauses.append("(date BETWEEN ? AND ?)")
            params.extend([item.start_date, item.end_date])
        clauses.append("(" + " OR ".join(range_clauses) + ")")
    if query.category:
        clauses.append("category = ?")
        params.append(query.category)
    if query.include_categories:
        placeholders = ",".join("?" for _ in query.include_categories)
        clauses.append(f"category IN ({placeholders})")
        params.extend(query.include_categories)
    if query.exclude_categories:
        placeholders = ",".join("?" for _ in query.exclude_categories)
        clauses.append(f"category NOT IN ({placeholders})")
        params.extend(query.exclude_categories)
    if query.merchant:
        clauses.append("COALESCE(merchant, '') LIKE ?")
        params.append(f"%{query.merchant}%")
    for keyword in query.include_keywords + query.keywords:
        clauses.append("(COALESCE(note, '') LIKE ? OR COALESCE(raw_text, '') LIKE ? OR COALESCE(merchant, '') LIKE ?)")
        params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
    for keyword in query.exclude_keywords:
        clauses.append("(COALESCE(note, '') NOT LIKE ? AND COALESCE(raw_text, '') NOT LIKE ? AND COALESCE(merchant, '') NOT LIKE ?)")
        params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
    if query.min_amount is not None:
        clauses.append("amount >= ?")
        params.append(query.min_amount)
    if query.max_amount is not None:
        clauses.append("amount <= ?")
        params.append(query.max_amount)
    return " AND ".join(clauses), params


def build_expense_query_from_structured(
    query: ExpenseQuery,
    line_user_id: str | None,
) -> tuple[str, list[object]]:
    where_sql, params = build_expense_where_clauses(query, line_user_id)
    sql = f"SELECT * FROM expenses WHERE {where_sql}"
    return sql, params


def calculate_expense_summary_with_ratio(query: ExpenseQuery, line_user_id: str | None) -> dict:
    filtered_result = execute_expense_query(query, line_user_id)
    denominator_total = filtered_result["total"]
    if query.include_ratio and query.ratio_denominator == "all_expenses":
        base_query = query.model_copy(update={"exclude_keywords": [], "exclude_categories": [], "category": None, "include_categories": [], "keywords": [], "include_keywords": [], "merchant": None})
        denominator_total = execute_expense_query(base_query, line_user_id)["total"]
    return {**filtered_result, "denominator_total": denominator_total}


def build_expense_query(query: ExpenseQuery, line_user_id: str | None) -> tuple[str, list[object]]:
    where_sql, params = build_expense_where_clauses(query, line_user_id)
    return f"SELECT * FROM expenses WHERE {where_sql}", params


def expense_query_date_label(query: ExpenseQuery) -> str:
    if not query.date_ranges:
        return "未指定"
    if len(query.date_ranges) == 1:
        item = query.date_ranges[0]
        return item.start_date if item.start_date == item.end_date else f"{item.start_date}~{item.end_date}"
    months = [expense_query_month_label(query)]
    return "、".join(filter(None, months)) or "多日期範圍"


def expense_query_month_label(query: ExpenseQuery) -> str:
    labels: list[str] = []
    for item in query.date_ranges:
        try:
            start = date.fromisoformat(item.start_date)
        except ValueError:
            continue
        labels.append(f"{start.year}-{start.month:02d}")
    return "、".join(labels)


def build_expense_order_by(query: ExpenseQuery) -> str:
    direction = "ASC" if query.sort_direction == "asc" else "DESC"
    mapping = {
        "date": "date",
        "time": "expense_time",
        "amount": "amount",
        "created_at": "created_at",
    }
    if query.sort_by in mapping:
        return f"ORDER BY {mapping[query.sort_by]} {direction}, id {direction}"
    return f"ORDER BY date {direction}, id {direction}"


def format_expense_sort_label(query: ExpenseQuery) -> str:
    mapping = {
        ("date", "desc"): "日期由新到舊",
        ("date", "asc"): "日期由舊到新",
        ("amount", "desc"): "金額由高到低",
        ("amount", "asc"): "金額由低到高",
        ("created_at", "desc"): "建立時間由新到舊",
        ("created_at", "asc"): "建立時間由舊到新",
    }
    return mapping.get((query.sort_by, query.sort_direction), "日期由新到舊")


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
    if group_by == "day":
        return "date"
    if group_by == "week":
        return "strftime('%Y-W%W', date)"
    if group_by == "month":
        return "substr(date, 1, 7)"
    if group_by == "category":
        return "category"
    return "COALESCE(merchant, '未指定')"


def iter_expense_group_keys(query: ExpenseQuery, group_by: ExpenseGroupBy) -> list[str]:
    if group_by == "day":
        keys: list[str] = []
        for item in query.date_ranges:
            start = date.fromisoformat(item.start_date)
            end = date.fromisoformat(item.end_date)
            cursor = start
            while cursor <= end:
                key = cursor.isoformat()
                if key not in keys:
                    keys.append(key)
                cursor += timedelta(days=1)
        return keys
    if group_by == "month":
        keys: list[str] = []
        for item in query.date_ranges:
            start = date.fromisoformat(item.start_date)
            end = date.fromisoformat(item.end_date)
            year = start.year
            month = start.month
            while (year, month) <= (end.year, end.month):
                key = f"{year}-{month:02d}"
                if key not in keys:
                    keys.append(key)
                month += 1
                if month == 13:
                    month = 1
                    year += 1
        return keys
    return []


def expense_row_group_key(row: dict[str, object], group_by: ExpenseGroupBy) -> str:
    if group_by == "day":
        return str(row["date"])
    if group_by == "week":
        target = date.fromisoformat(str(row["date"]))
        return target.strftime("%Y-W%W")
    if group_by == "month":
        return str(row["date"])[:7]
    if group_by == "category":
        return str(row["category"])
    merchant = row.get("merchant")
    return str(merchant) if merchant else "未指定"


def execute_expense_query(query: ExpenseQuery, line_user_id: str | None) -> dict:
    sql, params = build_expense_query(query, line_user_id)
    sql += " " + build_expense_order_by(query)
    with get_db() as conn:
        rows = conn.execute(sql, params).fetchall()
    data = [dict(row) for row in rows]

    if query.mode == "list_detail":
        return {"rows": data[: query.limit], "total": sum(int(row["amount"]) for row in data), "count": len(data)}

    if query.group_by:
        group = query.group_by[0]
        grouped: dict[str, dict[str, object]] = {}
        for row in data:
            key = expense_row_group_key(row, group)
            bucket = grouped.setdefault(key, {"group": key, "total": 0, "count": 0, "items": []})
            bucket["total"] = int(bucket["total"]) + int(row["amount"])
            bucket["count"] = int(bucket["count"]) + 1
            bucket["items"].append(row)
        ordered_rows: list[dict[str, object]] = []
        existing_keys = set(grouped)
        for key in iter_expense_group_keys(query, group):
            if key not in existing_keys:
                grouped[key] = {"group": key, "total": 0, "count": 0, "items": []}
        for key in sorted(grouped.keys()):
            ordered_rows.append(grouped[key])
        total = sum(int(row["total"]) for row in ordered_rows)
        return {"rows": ordered_rows, "total": total, "count": sum(int(row["count"]) for row in ordered_rows)}

    amounts = [int(row["amount"]) for row in data]
    total = sum(amounts)
    count = len(amounts)
    avg = int(total / count) if count else 0
    highest = max(data, key=lambda item: int(item["amount"])) if data else None
    lowest = min(data, key=lambda item: int(item["amount"])) if data else None
    return {
        "rows": data,
        "total": total,
        "count": count,
        "avg": avg,
        "highest": highest,
        "lowest": lowest,
    }


def build_expense_result_text(query: ExpenseQuery, result: dict) -> str:
    lines = ["查詢結果"]
    lines.append(f"日期：{expense_query_date_label(query)}")
    lines.append(f"分類：{query.category or '全部分類'}")

    if query.mode == "list_detail":
        lines.append(f"排序：{format_expense_sort_label(query)}")
        for row in result["rows"]:
            note = row.get("note") or row.get("merchant") or row.get("category")
            lines.append(f"{row['date']} {note} TWD {int(row['amount'])}")
        lines.append(f"小計：TWD {result['total']}")
        return "\n".join(lines)

    if query.group_by:
        group_label = {"day": "每日花費", "week": "每週花費", "month": "每月花費", "category": "各分類花費", "merchant": "各店家花費"}[query.group_by[0]]
        lines.append("")
        lines.append(f"{group_label}：")
        for row in result["rows"]:
            lines.append(f"{row['group']}：TWD {int(row['total'])}，{int(row['count'])} 筆")
        lines.append(f"合計：TWD {result['total']}")
        return "\n".join(lines)

    lines.append(f"筆數：{result['count']}")
    lines.append(f"總花費：TWD {result['total']}")
    lines.append(f"平均：TWD {result['avg']}")
    if result.get("highest"):
        highest = result["highest"]
        lines.append(f"最高：TWD {int(highest['amount'])}（{highest.get('note') or highest.get('merchant') or highest.get('category')}）")
    else:
        lines.append("最高：TWD 0")
    if result.get("lowest"):
        lowest = result["lowest"]
        lines.append(f"最低：TWD {int(lowest['amount'])}（{lowest.get('note') or lowest.get('merchant') or lowest.get('category')}）")
    else:
        lines.append("最低：TWD 0")
    return "\n".join(lines)


def generate_expense_chart(query: ExpenseQuery, result: dict, line_user_id: str | None = None) -> str | None:
    if not query.wants_chart or query.chart_type == "none":
        return None
    base_url = get_public_base_url()
    if not base_url:
        logger.warning("PUBLIC_BASE_URL is not configured; skip chart generation.")
        return None
    if result.get("total", 0) == 0:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = CHART_DIR / f"expense_{chart_cache_key({**query.model_dump(), 'rows': result['rows']})}.png"
    labels = []
    values = []
    if query.group_by:
        labels = [row["group"] for row in result["rows"]]
        values = [int(row["total"]) for row in result["rows"]]
    else:
        rows = result.get("rows", [])
        labels = [row.get("note") or row.get("merchant") or row.get("category") or row.get("date") for row in rows[:10]]
        values = [int(row["amount"]) for row in rows[:10]]
    if not labels or not values:
        return None

    plt.figure(figsize=(8, 4.5))
    if query.chart_type == "line":
        plt.plot(labels, values, marker="o")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Amount")
    elif query.chart_type == "bar":
        plt.bar(labels, values)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Amount")
    else:
        plt.pie(values, labels=labels, autopct="%1.0f%%")
        plt.axis("equal")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    return f"{base_url.rstrip('/')}/static/charts/{chart_path.name}"


def build_expense_reply(raw_text: str, line_user_id: str | None, route: ActionRoute | None = None) -> dict[str, str | None]:
    query = parse_expense_query(raw_text)
    result = execute_expense_query(query, line_user_id)
    text = build_expense_result_text(query, result)
    image_url = generate_expense_chart(query, result, line_user_id)
    if query.wants_chart and image_url is None:
        text += "\n\n圖表未產生：請確認 PUBLIC_BASE_URL 已設定，且查詢結果有可繪製的金額。"
    return {"text": text, "image_url": image_url}


def build_list_reply(raw_text: str, line_user_id: str | None, route: ActionRoute | None = None) -> str:
    reply = build_expense_reply(raw_text, line_user_id, route)
    return str(reply["text"])


def build_top_expense_reply(raw_text: str, line_user_id: str | None, route: ActionRoute | None = None) -> str:
    query = parse_expense_query(raw_text)
    result = execute_expense_query(query, line_user_id)
    highest = result.get("highest")
    if not highest:
        return "查無資料"
    return f"最高花費：{highest['date']} {highest.get('note') or highest.get('merchant') or highest.get('category')} TWD {int(highest['amount'])}"


def build_income_query_reply(raw_text: str, line_user_id: str | None, income_type: str | None = None) -> str:
    target_type = income_type or get_income_type(raw_text)
    normalized = re.sub(r"\s+", "", raw_text)
    today = date.today()
    start_date = today.replace(day=1).isoformat()
    end_date = today.isoformat()
    if any(keyword in normalized for keyword in ("昨天",)):
        start_date = end_date = (today - timedelta(days=1)).isoformat()
    elif any(keyword in normalized for keyword in ("今天",)):
        start_date = end_date = today.isoformat()
    with get_db() as conn:
        params: list[object] = [start_date, end_date]
        sql = (
            "SELECT income_date, amount, item_name, owner, category, raw_text "
            "FROM incomes WHERE income_date BETWEEN ? AND ?"
        )
        scope_line_user_id = get_household_read_scope_line_user_id(line_user_id)
        if scope_line_user_id is not None:
            sql += " AND line_user_id = ?"
            params.append(scope_line_user_id)
        if target_type:
            sql += " AND income_type = ?"
            params.append(target_type)
        rows = conn.execute(sql + " ORDER BY income_date DESC, id DESC", params).fetchall()
    if not rows:
        return f"查無{summarize_income_type(target_type)}資料\n日期：{start_date}" if start_date == end_date else f"查無{summarize_income_type(target_type)}資料\n日期：{start_date}~{end_date}"
    total = sum(int(row["amount"]) for row in rows)
    average = int(total / len(rows)) if rows else 0
    highest = max(rows, key=lambda row: int(row["amount"]))
    lowest = min(rows, key=lambda row: int(row["amount"]))
    lines = ["收入查詢結果"]
    lines.append(f"日期：{start_date}" if start_date == end_date else f"日期：{start_date}~{end_date}")
    lines.append(f"類型：{summarize_income_type(target_type)}")
    lines.append(f"筆數：{len(rows)}")
    lines.append(f"總收入：TWD {total}")
    lines.append(f"平均：TWD {average}")
    lines.append(f"最高：TWD {int(highest['amount'])}（{highest['item_name'] or highest['category'] or highest['raw_text']}）")
    lines.append(f"最低：TWD {int(lowest['amount'])}（{lowest['item_name'] or lowest['category'] or lowest['raw_text']}）")
    return "\n".join(lines)


def build_income_list_reply(raw_text: str, line_user_id: str | None, income_type: str | None = None) -> str:
    target_type = income_type or get_income_type(raw_text)
    normalized = re.sub(r"\s+", "", raw_text)
    today = date.today()
    start_date = today.replace(day=1).isoformat()
    end_date = today.isoformat()
    if any(keyword in normalized for keyword in ("昨天",)):
        start_date = end_date = (today - timedelta(days=1)).isoformat()
    elif any(keyword in normalized for keyword in ("今天",)):
        start_date = end_date = today.isoformat()
    with get_db() as conn:
        params: list[object] = [start_date, end_date]
        sql = (
            "SELECT income_date, amount, item_name, owner, category, raw_text "
            "FROM incomes WHERE income_date BETWEEN ? AND ?"
        )
        scope_line_user_id = get_household_read_scope_line_user_id(line_user_id)
        if scope_line_user_id is not None:
            sql += " AND line_user_id = ?"
            params.append(scope_line_user_id)
        if target_type:
            sql += " AND income_type = ?"
            params.append(target_type)
        rows = conn.execute(sql + " ORDER BY income_date DESC, id DESC", params).fetchall()
    if not rows:
        return f"查無{summarize_income_type(target_type)}明細\n日期：{start_date}" if start_date == end_date else f"查無{summarize_income_type(target_type)}明細\n日期：{start_date}~{end_date}"
    lines = ["收入明細"]
    lines.append(f"日期：{start_date}" if start_date == end_date else f"日期：{start_date}~{end_date}")
    lines.append(f"類型：{summarize_income_type(target_type)}")
    for row in rows[:20]:
        label = row["item_name"] or row["category"] or row["raw_text"]
        lines.append(f"{row['income_date']} {label} TWD {int(row['amount'])}")
    lines.append(f"小計：TWD {sum(int(row['amount']) for row in rows)}")
    return "\n".join(lines)


def get_period_range(raw_text: str) -> tuple[str, str]:
    today = date.today()
    normalized = re.sub(r"\s+", "", raw_text)
    if any(keyword in normalized for keyword in ("今天",)):
        target = today.isoformat()
        return target, target
    if any(keyword in normalized for keyword in ("昨天",)):
        target = (today - timedelta(days=1)).isoformat()
        return target, target
    if any(keyword in normalized for keyword in ("這週", "這周", "這禮拜", "本週", "本周", "本禮拜", "本星期", "這星期")):
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        return start.isoformat(), end.isoformat()
    start = today.replace(day=1)
    return start.isoformat(), today.isoformat()


def compute_unpaid_payables_total(line_user_id: str | None, start_date: str, end_date: str) -> int:
    scope_line_user_id = get_household_read_scope_line_user_id(line_user_id)
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS total
            FROM payables
            WHERE status = 'unpaid'
              AND due_date BETWEEN ? AND ?
              AND (? IS NULL OR line_user_id = ?)
            """,
            (start_date, end_date, scope_line_user_id, scope_line_user_id),
        ).fetchone()
    return int(row["total"] if row else 0)


def build_balance_reply(raw_text: str, line_user_id: str | None) -> str:
    start_date, end_date = get_period_range(raw_text)
    with get_db() as conn:
        scope_line_user_id = get_household_read_scope_line_user_id(line_user_id)
        expense_row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS total
            FROM expenses
            WHERE date BETWEEN ? AND ?
              AND (? IS NULL OR line_user_id = ?)
            """,
            (start_date, end_date, scope_line_user_id, scope_line_user_id),
        ).fetchone()
        income_row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS total
            FROM incomes
            WHERE income_date BETWEEN ? AND ?
              AND (? IS NULL OR line_user_id = ?)
            """,
            (start_date, end_date, scope_line_user_id, scope_line_user_id),
        ).fetchone()
    expense_total = int(expense_row["total"] if expense_row else 0)
    income_total = int(income_row["total"] if income_row else 0)
    unpaid_total = compute_unpaid_payables_total(line_user_id, start_date, end_date)
    balance = income_total - expense_total - unpaid_total
    if any(keyword in raw_text for keyword in ("透支", "超支", "赤字")):
        status = "是" if balance < 0 else "否"
        lines = ["收支檢查結果", f"日期：{start_date}" if start_date == end_date else f"日期：{start_date}~{end_date}"]
        lines.append(f"收入：TWD {income_total}")
        lines.append(f"支出：TWD {expense_total}")
        lines.append(f"待繳：TWD {unpaid_total}")
        lines.append(f"是否透支：{status}")
        lines.append(f"餘額：TWD {balance}")
        return "\n".join(lines)
    lines = ["收支查詢結果", f"日期：{start_date}" if start_date == end_date else f"日期：{start_date}~{end_date}"]
    lines.append(f"收入：TWD {income_total}")
    lines.append(f"支出：TWD {expense_total}")
    lines.append(f"待繳：TWD {unpaid_total}")
    lines.append(f"餘額：TWD {balance}")
    return "\n".join(lines)


def build_available_cash_reply(raw_text: str, line_user_id: str | None, route: ActionRoute | None = None) -> str:
    start_date, end_date = get_period_range(raw_text)
    with get_db() as conn:
        scope_line_user_id = get_household_read_scope_line_user_id(line_user_id)
        income_row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS total
            FROM incomes
            WHERE income_date BETWEEN ? AND ?
              AND (? IS NULL OR line_user_id = ?)
            """,
            (start_date, end_date, scope_line_user_id, scope_line_user_id),
        ).fetchone()
        expense_row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS total
            FROM expenses
            WHERE date BETWEEN ? AND ?
              AND (? IS NULL OR line_user_id = ?)
            """,
            (start_date, end_date, scope_line_user_id, scope_line_user_id),
        ).fetchone()
    income_total = int(income_row["total"] if income_row else 0)
    expense_total = int(expense_row["total"] if expense_row else 0)
    unpaid_total = compute_unpaid_payables_total(line_user_id, start_date, end_date)
    available = income_total - expense_total - unpaid_total
    purpose = route.purchase_purpose if route and route.purchase_purpose else extract_purchase_purpose(raw_text)
    if purpose:
        return (
            f"目前可動用金額：TWD {available}\n"
            f"用途：{purpose}\n"
            f"期間：{start_date}" if start_date == end_date else f"目前可動用金額：TWD {available}\n用途：{purpose}\n期間：{start_date}~{end_date}"
        )
    return f"目前可動用金額：TWD {available}\n期間：{start_date}" if start_date == end_date else f"目前可動用金額：TWD {available}\n期間：{start_date}~{end_date}"


def build_available_investment_cash_reply(raw_text: str, line_user_id: str | None) -> str:
    start_date, end_date = get_period_range(raw_text)
    with get_db() as conn:
        scope_line_user_id = get_household_read_scope_line_user_id(line_user_id)
        income_row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS total
            FROM incomes
            WHERE income_date BETWEEN ? AND ?
              AND (? IS NULL OR line_user_id = ?)
            """,
            (start_date, end_date, scope_line_user_id, scope_line_user_id),
        ).fetchone()
        expense_row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS total
            FROM expenses
            WHERE date BETWEEN ? AND ?
              AND (? IS NULL OR line_user_id = ?)
            """,
            (start_date, end_date, scope_line_user_id, scope_line_user_id),
        ).fetchone()
    income_total = int(income_row["total"] if income_row else 0)
    expense_total = int(expense_row["total"] if expense_row else 0)
    unpaid_total = compute_unpaid_payables_total(line_user_id, start_date, end_date)
    available = income_total - expense_total - unpaid_total
    recommended = max(0, int(available * 0.3))
    lines = ["投資可動用金額"]
    lines.append(f"期間：{start_date}" if start_date == end_date else f"期間：{start_date}~{end_date}")
    lines.append(f"收入：TWD {income_total}")
    lines.append(f"支出：TWD {expense_total}")
    lines.append(f"待繳：TWD {unpaid_total}")
    lines.append(f"目前可動用：TWD {available}")
    lines.append(f"保守建議投入：約 TWD {recommended}")
    return "\n".join(lines)


def find_expense_candidates(
    raw_text: str,
    line_user_id: str | None,
    *,
    include_all_users: bool = False,
) -> list[sqlite3.Row]:
    target_date = parse_due_specific_day(raw_text) or (date.today().isoformat() if "今天" in raw_text or not parse_due_specific_day(raw_text) and not re.search(r"\d{1,2}\s*/\s*\d{1,2}|\d{1,2}\s*(?:日|號)|20\d{2}-\d{2}-\d{2}|昨天", raw_text) else None)
    if "昨天" in raw_text:
        target_date = (date.today() - timedelta(days=1)).isoformat()
    amount = parse_amount(raw_text)
    category = get_query_category(raw_text)
    normalized = re.sub(r"\s+", "", raw_text)
    note = re.sub(r"(?:刪除|删除|移除|取消|早餐|午餐|晚餐|飲料|今天|昨天|\d+|元|塊|塊錢|[\u3000\s])", "", normalized)
    note = note.strip()

    with get_db() as conn:
        scope_line_user_id = None if include_all_users else line_user_id
        base_sql = "SELECT * FROM expenses WHERE 1 = 1"
        params: list[object] = []
        if target_date:
            base_sql += " AND date = ?"
            params.append(target_date)
        if amount is not None:
            base_sql += " AND amount = ?"
            params.append(amount)
        if category is not None:
            base_sql += " AND category = ?"
            params.append(category)
        if scope_line_user_id is not None:
            base_sql += " AND line_user_id = ?"
            params.append(scope_line_user_id)
        rows = conn.execute(base_sql + " ORDER BY id DESC", params).fetchall()

    if note:
        filtered = []
        for row in rows:
            haystack = f"{row['raw_text']} {row['note'] or ''} {row['merchant'] or ''}"
            if note in re.sub(r"\s+", "", str(haystack)):
                filtered.append(row)
        rows = filtered
    return list(rows)


def row_to_expense(row: sqlite3.Row) -> ExpenseEntry:
    return row_to_expense_entry(row)


def build_delete_candidates_reply(raw_text: str, line_user_id: str | None) -> str:
    rows = find_expense_candidates(raw_text, line_user_id)
    target_date = parse_due_specific_day(raw_text)
    category = get_query_category(raw_text)
    if not rows:
        lines = ["找不到可刪除的花費。"]
        if target_date:
            lines.append(f"日期：{target_date}")
        if category:
            lines.append(f"分類：{category}")
        return "\n".join(lines)
    if len(rows) > 1:
        lines = ["我找到多筆可能符合的花費，請再說更完整一點，例如金額或日期："]
        for row in rows[:5]:
            lines.append(f"- {row['date']} {row['note'] or row['merchant'] or row['category']} TWD {int(row['amount'])}")
        return "\n".join(lines)
    row = rows[0]
    pending_delete_by_user[line_user_id or "_global"] = int(row["id"])
    return (
        "找到一筆可刪除花費，請回覆確認碼 88 以刪除：\n"
        f"日期：{row['date']}\n"
        f"金額：TWD {int(row['amount'])}\n"
        f"分類：{row['category']}\n"
        f"備註：{row['note'] or row['merchant'] or row['raw_text']}"
    )


def delete_expense(raw_text: str, line_user_id: str | None) -> DeleteResult:
    if is_confirm_delete_intent(raw_text):
        pending_id = pending_delete_by_user.pop(line_user_id or "_global", None)
        if pending_id is None:
            return DeleteResult(deleted=False, reason="目前沒有等待確認刪除的記帳項目。")
        with get_db() as conn:
            row = conn.execute("SELECT * FROM expenses WHERE id = ?", (pending_id,)).fetchone()
            if row is None:
                return DeleteResult(deleted=False, reason="剛剛那筆記帳已經不存在了。")
            conn.execute("DELETE FROM expenses WHERE id = ?", (pending_id,))
            conn.commit()
            log_sql_change("expense", "delete", pending_id, str(row["raw_text"]), line_user_id)
            expense = row_to_expense(row)
        return DeleteResult(deleted=True, expense=expense, expense_id=pending_id)

    rows = find_expense_candidates(raw_text, line_user_id)
    if not rows:
        return DeleteResult(deleted=False, reason=build_delete_candidates_reply(raw_text, line_user_id))
    if len(rows) > 1:
        return DeleteResult(deleted=False, reason=build_delete_candidates_reply(raw_text, line_user_id), similar_count=len(rows))
    row = rows[0]
    pending_delete_by_user[line_user_id or "_global"] = int(row["id"])
    expense = row_to_expense(row)
    return DeleteResult(deleted=False, expense=expense, expense_id=int(row["id"]), reason=build_delete_candidates_reply(raw_text, line_user_id))


def build_delete_reply(result: DeleteResult) -> str:
    if result.deleted and result.expense:
        return (
            "已刪除花費\n"
            f"日期：{result.expense.date}\n"
            f"金額：{result.expense.currency} {result.expense.amount}\n"
            f"分類：{result.expense.category}\n"
            f"備註：{result.expense.note or '無'}"
        )
    if result.reason:
        return result.reason
    return "刪除失敗，請再試一次。"


def is_delete_duplicate_text(raw_text: str) -> bool:
    normalized = re.sub(r"\s+", "", raw_text)
    return any(keyword in normalized for keyword in ("刪除重複", "删除重複", "移除重複", "只留一筆其他刪除", "刪掉重複"))


def is_duplicate_data_text(raw_text: str) -> bool:
    normalized = re.sub(r"\s+", "", raw_text)
    return any(keyword in normalized for keyword in ("重複資料", "重複的資料", "列出重複", "看重複"))


def dedupe_payables() -> int:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, line_user_id, item_type, amount, due_date, COALESCE(owner, '') AS owner,
                   COALESCE(bank, '') AS bank, status
            FROM payables
            WHERE status = 'unpaid'
            ORDER BY id ASC
            """
        ).fetchall()
        groups: dict[tuple[str, str, int, str, str, str, str], list[int]] = {}
        for row in rows:
            key = (
                str(row["line_user_id"]),
                str(row["item_type"]),
                int(row["amount"]),
                str(row["due_date"]),
                str(row["owner"]),
                str(row["bank"]),
                str(row["status"]),
            )
            groups.setdefault(key, []).append(int(row["id"]))
        delete_ids: list[int] = []
        for ids in groups.values():
            if len(ids) > 1:
                delete_ids.extend(ids[1:])
        if not delete_ids:
            return 0
        conn.executemany("DELETE FROM payables WHERE id = ?", [(payable_id,) for payable_id in delete_ids])
        conn.commit()
    return len(delete_ids)


def normalize_income_type(value: str | None) -> str | None:
    if not value:
        return None
    raw = str(value).strip()
    normalized = re.sub(r"\s+", "", raw)
    mapping = {
        "薪資收入": {"薪資收入", "薪水", "薪資", "工資", "月薪"},
        "補助收入": {"補助", "補貼", "政府補助", "補助收入"},
        "獎金收入": {"獎金", "中獎", "獎金收入"},
        "租金收入": {"租金", "房租", "租金收入"},
        "股利收入": {"股利", "配息", "股利收入"},
        "接案收入": {"接案", "外包", "任務費", "接案收入"},
        "業外收入": {"業外", "副業", "兼差", "業外收入"},
        "其他收入": {"其他", "其他收入"},
    }
    for canonical, values in mapping.items():
        if normalized in values:
            return canonical
    return raw


def resolve_income_item_name(raw_text: str, income_type: str | None = None) -> str:
    normalized = re.sub(r"\s+", "", raw_text)
    category = normalize_income_type(income_type) or "收入"
    pattern = re.compile(r"^(?:今天|昨天|前天|\d{1,2}[/-]\d{1,2})?(.*?)(\d+(?:\.\d+)?)(?:元|塊|塊錢|萬)?$")
    match = pattern.match(normalized)
    if match:
        item_text = match.group(1)
        item_text = re.sub(r"^(?:有一筆|一筆|收到|入帳|進帳|新增|記一筆)", "", item_text)
        item_text = re.sub(r"(收入|薪資收入|補助收入|獎金收入|租金收入|股利收入|接案收入|業外收入)$", "", item_text)
        item_text = item_text.strip()
        if item_text:
            return item_text
    if category == "薪資收入":
        return "薪水"
    if category == "補助收入":
        return "補助"
    if category == "獎金收入":
        return "獎金"
    if category == "租金收入":
        return "租金"
    if category == "股利收入":
        return "股利"
    if category == "接案收入":
        return "接案"
    if category == "業外收入":
        return "業外收入"
    return category


def resolve_income_owner(raw_text: str) -> str | None:
    owner_patterns = (
        (r"老婆", "老婆"),
        (r"老公", "老公"),
        (r"先生", "先生"),
        (r"太太", "太太"),
    )
    for pattern, owner in owner_patterns:
        if re.search(pattern, raw_text):
            return owner
    return None


def resolve_income_category(income_type: str | None) -> str:
    normalized = normalize_income_type(income_type)
    return normalized or "其他收入"


def parse_income_date(raw_text: str) -> str:
    return parse_due_date(raw_text) or date.today().isoformat()


def save_income(raw_text: str, line_user_id: str | None) -> str | None:
    income_type = normalize_income_type(get_income_type(raw_text))
    if income_type is None:
        return None
    amount = parse_amount(raw_text)
    if amount is None:
        return None
    income_date = parse_income_date(raw_text)
    item_name = resolve_income_item_name(raw_text, income_type)
    owner = resolve_income_owner(raw_text)
    category = resolve_income_category(income_type)
    created_at = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO incomes (
                line_user_id, raw_text, income_date, amount, currency,
                income_type, item_name, owner, category, note, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                line_user_id,
                raw_text,
                income_date,
                amount,
                "TWD",
                income_type,
                item_name,
                owner,
                category,
                raw_text,
                created_at,
            ),
        )
        income_id = int(cursor.lastrowid)
        conn.commit()
    log_sql_change("income", "insert", income_id, raw_text, line_user_id)
    return (
        "已記收入\n"
        f"日期：{income_date}\n"
        f"金額：TWD {amount}\n"
        f"類型：{income_type}\n"
        f"項目：{item_name}\n"
        f"對象：{owner or '未指定'}"
    )


def handle_payable_setup(raw_text: str, line_user_id: str | None, message_id: str | None) -> str | None:
    if not line_user_id:
        return None
    user_key = line_user_id
    draft = get_payable_draft(line_user_id)
    if draft and raw_text.strip() == "取消":
        clear_payable_draft(line_user_id)
        return "好，已取消這筆待繳提醒草稿。"
    if draft and raw_text.strip() in {"確認", "好", "可以", "確定"}:
        item_type = draft["item_type"]
        amount = int(draft["amount"])
        due_date = str(draft.get("due_date") or parse_due_date(raw_text) or date.today().isoformat())
        clear_payable_draft(line_user_id)
        return create_payable(line_user_id, item_type, amount, due_date, draft["raw_text"], message_id)

    if get_payable_type(raw_text) and parse_payable_amount(raw_text) is not None and parse_due_date(raw_text) is None:
        save_payable_draft(
            line_user_id,
            get_payable_type(raw_text) or "這筆繳費",
            parse_payable_amount(raw_text) or 0,
            get_owner(raw_text),
            get_bank(raw_text),
            raw_text,
        )
        return "我先記下這筆待繳款，請再告訴我繳費日期，例如：7/15、7月15日、明天。"

    if draft and parse_due_date(raw_text):
        item_type = draft["item_type"]
        amount = int(draft["amount"])
        due_date = parse_due_date(raw_text) or date.today().isoformat()
        clear_payable_draft(line_user_id)
        return create_payable(line_user_id, item_type, amount, due_date, draft["raw_text"], message_id)

    return None


def save_payable_draft(
    line_user_id: str,
    item_type: str,
    amount: int,
    owner: str | None,
    bank: str | None,
    raw_text: str,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO payable_drafts (line_user_id, item_type, amount, currency, owner, bank, note, created_at)
            VALUES (?, ?, ?, 'TWD', ?, ?, ?, ?)
            ON CONFLICT(line_user_id) DO UPDATE SET
                item_type = excluded.item_type,
                amount = excluded.amount,
                currency = excluded.currency,
                owner = excluded.owner,
                bank = excluded.bank,
                note = excluded.note,
                created_at = excluded.created_at
            """,
            (line_user_id, item_type, amount, owner, bank, raw_text, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def get_payable_draft(line_user_id: str | None) -> dict[str, object] | None:
    if not line_user_id:
        return None
    with get_db() as conn:
        row = conn.execute("SELECT * FROM payable_drafts WHERE line_user_id = ?", (line_user_id,)).fetchone()
    if row is None:
        return None
    return dict(row)


def clear_payable_draft(line_user_id: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM payable_drafts WHERE line_user_id = ?", (line_user_id,))
        conn.commit()


def create_payable(
    line_user_id: str,
    item_type: str,
    amount: int,
    due_date: str,
    raw_text: str,
    message_id: str | None = None,
) -> str:
    item_type = normalize_payable_item_type(item_type) or item_type
    owner = get_owner(raw_text)
    bank = get_bank(raw_text)
    with get_db() as conn:
        if message_id:
            existing = conn.execute(
                "SELECT * FROM payables WHERE line_user_id = ? AND message_id = ?",
                (line_user_id, message_id),
            ).fetchone()
            if existing:
                return "這筆待繳款已經建立過。"
        duplicate = conn.execute(
            """
            SELECT id FROM payables
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
        if duplicate:
            return "這筆待繳款已經存在。"
        cursor = conn.execute(
            """
            INSERT INTO payables (
                line_user_id, item_type, amount, currency, due_date,
                owner, bank, note, message_id, status, paid_at, created_at
            )
            VALUES (?, ?, ?, 'TWD', ?, ?, ?, ?, ?, 'unpaid', NULL, ?)
            """,
            (
                line_user_id,
                item_type,
                amount,
                due_date,
                owner,
                bank,
                raw_text,
                message_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        payable_id = int(cursor.lastrowid)
        conn.commit()
    log_sql_change("payable", "insert", payable_id, f"{item_type} TWD {amount} {due_date}", line_user_id)
    remind_dates = [
        (date.fromisoformat(due_date) - timedelta(days=offset)).isoformat()
        for offset in (3, 2, 1)
    ]
    target_name = format_payable_target_name(item_type, owner, bank)
    return (
        "已建立待繳提醒\n"
        f"項目：{target_name}\n"
        f"金額：TWD {amount}\n"
        f"到期日：{due_date}\n"
        f"提醒日期：{'、'.join(remind_dates)}\n"
        "提醒方式：LINE push"
    )


def build_payable_query_reply(raw_text: str, line_user_id: str | None, route: ActionRoute | None = None) -> str:
    parsed = None
    try:
        parsed = parse_payable_query_with_openai(raw_text)
    except Exception:
        logger.exception("OpenAI payable query parsing failed; using local fallback for text: %s", raw_text)
    item_type = normalize_payable_item_type((route.item_type if route and route.item_type else None) or (parsed.item_type if parsed else None) or get_payable_type(raw_text))
    owner = (route.owner if route else None) or (parsed.owner if parsed else None) or get_owner(raw_text)
    bank = (route.bank if route else None) or (parsed.bank if parsed else None) or get_bank(raw_text)
    status = (route.status if route and route.status else None) or (parsed.status if parsed else None) or "all"

    normalized = re.sub(r"\s+", "", raw_text)
    today = date.today()
    if parsed and parsed.date_range_type == "custom" and parsed.start_date and parsed.end_date:
        start_date, end_date = parsed.start_date, parsed.end_date
    elif any(keyword in normalized for keyword in ("今天",)):
        start_date = end_date = today.isoformat()
    elif any(keyword in normalized for keyword in ("昨天",)):
        start_date = end_date = (today - timedelta(days=1)).isoformat()
    elif any(keyword in normalized for keyword in ("這週", "這周", "這禮拜", "本週", "本周", "本禮拜", "本星期")):
        week_start = today - timedelta(days=today.weekday())
        start_date = week_start.isoformat()
        end_date = (week_start + timedelta(days=6)).isoformat()
    else:
        start_date = today.replace(day=1).isoformat()
        end_date = today.isoformat()

    scope_line_user_id = get_household_read_scope_line_user_id(line_user_id)
    with get_db() as conn:
        params: list[object] = [start_date, end_date]
        sql = (
            "SELECT item_type, amount, due_date, owner, bank, status, paid_at "
            "FROM payables WHERE due_date BETWEEN ? AND ?"
        )
        if scope_line_user_id is not None:
            sql += " AND line_user_id = ?"
            params.append(scope_line_user_id)
        if item_type:
            sql += " AND item_type = ?"
            params.append(item_type)
        if owner:
            sql += " AND COALESCE(owner, '') = COALESCE(?, '')"
            params.append(owner)
        if bank:
            sql += " AND COALESCE(bank, '') = COALESCE(?, '')"
            params.append(bank)
        if status in {"paid", "unpaid"}:
            sql += " AND status = ?"
            params.append(status)
        rows = conn.execute(sql + " ORDER BY due_date ASC, item_type ASC, amount DESC", params).fetchall()

    target_name = format_payable_target_name(item_type, owner, bank) if (item_type or owner or bank) else "待繳款"

    if any(keyword in normalized for keyword in ("繳了嗎", "還沒繳嗎", "未繳", "待繳", "還沒繳")):
        if not rows:
            return f"這個月沒有{target_name}繳費紀錄"
        unpaid_rows = [row for row in rows if row["status"] == "unpaid"]
        paid_rows = [row for row in rows if row["status"] == "paid"]
        if unpaid_rows and status != "paid":
            row = unpaid_rows[0]
            return f"{format_payable_target_name(row['item_type'], row['owner'], row['bank'])} 還沒繳，到期日 {row['due_date']}，金額 TWD {int(row['amount'])}。"
        if paid_rows:
            row = paid_rows[0]
            paid_date = str(row["paid_at"])[:10] if row["paid_at"] else "未記錄"
            return f"{format_payable_target_name(row['item_type'], row['owner'], row['bank'])} 已經繳了，繳費日期 {paid_date}。"

    if not rows:
        if item_type:
            return f"查無{target_name}資料\n日期：{start_date}" if start_date == end_date else f"查無{target_name}資料\n日期：{start_date}~{end_date}"
        return f"查無待繳款資料\n日期：{start_date}" if start_date == end_date else f"查無待繳款資料\n日期：{start_date}~{end_date}"

    lines = ["待繳款查詢結果"]
    lines.append(f"日期：{start_date}" if start_date == end_date else f"日期：{start_date}~{end_date}")
    lines.append(f"項目：{target_name}")
    for row in rows[:20]:
        row_name = format_payable_target_name(row["item_type"], row["owner"], row["bank"])
        suffix = "已繳" if row["status"] == "paid" else "未繳"
        lines.append(f"{row['due_date']} {row_name} TWD {int(row['amount'])}（{suffix}）")
    return "\n".join(lines)


def find_duplicate_data_groups(line_user_id: str | None) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    scope_line_user_id = get_household_read_scope_line_user_id(line_user_id)
    with get_db() as conn:
        expense_rows = conn.execute(
            """
            SELECT id, date, expense_time, amount, currency, category, merchant, note, raw_text, line_user_id
            FROM expenses
            WHERE amount > 0
              AND (? IS NULL OR line_user_id = ?)
            ORDER BY date ASC, expense_time ASC, id ASC
            """,
            (scope_line_user_id, scope_line_user_id),
        ).fetchall()
        income_rows = conn.execute(
            """
            SELECT id, income_date, amount, income_type, item_name, owner, category, raw_text, line_user_id
            FROM incomes
            WHERE amount > 0
              AND (? IS NULL OR line_user_id = ?)
            ORDER BY income_date ASC, id ASC
            """,
            (scope_line_user_id, scope_line_user_id),
        ).fetchall()
        payable_rows = conn.execute(
            """
            SELECT id, line_user_id, due_date, item_type, amount, owner, bank, status
            FROM payables
            WHERE amount > 0
              AND (? IS NULL OR line_user_id = ?)
            ORDER BY due_date ASC, id ASC
            """,
            (scope_line_user_id, scope_line_user_id),
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


def ensure_chat_prefix(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("[chat]"):
        return stripped
    return f"[chat] {stripped}"


def execute_action_route(
    route: ActionRoute,
    raw_text: str,
    line_user_id: str | None,
    message_id: str | None,
    chat_id: str | None = None,
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
    resolved_chat_id = resolve_home_task_chat_id(chat_id, line_user_id)
    pending_key = get_pending_expense_key(line_user_id)
    if is_confirm_delete_intent(raw_text) and pending_delete_by_user.get(line_user_id or ""):
        delete_result = delete_expense(raw_text, line_user_id)
        return build_delete_reply(delete_result)

    if is_confirm_delete_duplicates_text(raw_text):
        return confirm_delete_duplicates(line_user_id)

    draft = get_home_task_draft(resolved_chat_id)
    if draft and home_task_confirmation_yes(raw_text):
        delete_home_task_draft(resolved_chat_id)
        task = create_home_task_record(
            draft.actor_user_id or line_user_id,
            resolved_chat_id,
            draft.original_text,
            message_id,
            draft.title,
            draft.item_key,
            draft.category,
            draft.scheduled_date,
            draft.scheduled_time,
        )
        return build_home_task_created_reply(task)
    if draft and home_task_confirmation_no(raw_text):
        delete_home_task_draft(resolved_chat_id)
        return "好，已取消這筆家庭事項草稿。"

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

    if route.action == "query_home_tasks":
        return query_home_tasks(raw_text, resolved_chat_id)

    if route.action == "query_home_task_history":
        return query_home_task_history(raw_text, resolved_chat_id)

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

    if route.action == "create_home_task":
        return create_home_task_from_route(raw_text, line_user_id, resolved_chat_id, message_id, route)

    if route.action == "complete_home_task":
        return complete_home_task(raw_text, line_user_id, resolved_chat_id)

    if route.action == "cancel_home_task":
        return cancel_home_task(raw_text, line_user_id, resolved_chat_id)

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

    return ensure_chat_prefix(build_chat_reply(raw_text))


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


def get_line_event_dedup_key(event: dict) -> str | None:
    webhook_event_id = event.get("webhookEventId")
    if isinstance(webhook_event_id, str) and webhook_event_id:
        return f"webhook:{webhook_event_id}"
    message = event.get("message", {})
    message_id = message.get("id")
    if isinstance(message_id, str) and message_id:
        return f"message:{message_id}"
    return None


def claim_line_event_receipt(event: dict, now: datetime | None = None) -> bool:
    dedup_key = get_line_event_dedup_key(event)
    if not dedup_key:
        return True

    current_time = now or datetime.now(timezone.utc)
    cutoff = current_time - timedelta(days=14)
    message = event.get("message", {})
    source = event.get("source", {})
    try:
        with get_db() as conn:
            conn.execute(
                "DELETE FROM line_event_receipts WHERE created_at < ?",
                (cutoff.isoformat(),),
            )
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO line_event_receipts (
                    dedup_key, webhook_event_id, message_id, event_type, chat_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    dedup_key,
                    event.get("webhookEventId"),
                    message.get("id"),
                    event.get("type"),
                    get_line_chat_id(source),
                    current_time.isoformat(),
                ),
            )
            conn.commit()
            return cursor.rowcount > 0
    except sqlite3.Error:
        logger.exception("Failed to claim LINE event receipt dedup_key=%s", dedup_key)
        return True


def is_duplicate_line_event(event: dict, now: datetime | None = None) -> bool:
    dedup_key = get_line_event_dedup_key(event)
    if not dedup_key:
        return False

    current_time = now or datetime.now(timezone.utc)
    cutoff = current_time - timedelta(seconds=LINE_EVENT_DEDUP_WINDOW_SECONDS)
    expired_keys = [key for key, seen_at in recent_line_event_times.items() if seen_at < cutoff]
    for key in expired_keys:
        recent_line_event_times.pop(key, None)

    seen_at = recent_line_event_times.get(dedup_key)
    if seen_at and seen_at >= cutoff:
        return True

    recent_line_event_times[dedup_key] = current_time
    return False


def process_message_sync(ctx: ProcessingContext, event: dict) -> object:
    message = event.get("message", {})
    raw_text = message.get("text", "").strip()
    source = event.get("source", {})
    line_user_id = source.get("userId")
    message_id = message.get("id")
    scope_type = source.get("type")
    scope_id = get_line_chat_id(source)
    ctx.status = "\u5206\u6790\u8a0a\u606f\u4e2d"
    feedback_reply = apply_last_utterance_feedback(raw_text, line_user_id, scope_id)
    if feedback_reply is not None:
        ctx.status = "\u5b8c\u6210"
        return feedback_reply
    special_reply = handle_special_command(raw_text, line_user_id)
    if special_reply is not None:
        ctx.status = "\u5b8c\u6210"
        return special_reply
    ctx.status = "\u7b49\u5f85 ChatGPT \u56de\u61c9\u4e2d"
    route = route_action(raw_text, line_user_id)
    log_utterance(raw_text, message_id, line_user_id, scope_type, scope_id, route)
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
    reply_payload = execute_action_route(route, raw_text, line_user_id, message_id, scope_id)
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
            await reply_or_push_text(
                reply_token,
                chat_id,
                f"\u6211\u9084\u5728\u8655\u7406\u4e0a\u4e00\u500b\u67e5\u8a62\uff0c\u76ee\u524d\u6b63\u5728{ctx.status}\uff0c\u7b49\u6211\u4e00\u4e0b\u3002",
            )
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
        await reply_or_push_payload(reply_token, chat_id, reply_payload)
        in_flight_tasks.pop(key, None)
    except asyncio.TimeoutError:
        ctx.interim_replied = True
        await reply_or_push_text(reply_token, chat_id, build_interim_reply(ctx.status))
        task.add_done_callback(lambda done_task: asyncio.create_task(finalize_background_message(ctx, done_task, key)))
    except Exception:
        logger.exception("Failed to handle LINE text: %s", raw_text)
        in_flight_tasks.pop(key, None)
        try:
            await reply_or_push_text(
                reply_token,
                chat_id,
                "\u525b\u525b\u8655\u7406\u5931\u6557\u4e86\uff0c\u6211\u6709\u8a18\u9304\u932f\u8aa4\uff0c\u8acb\u7a0d\u5f8c\u518d\u8a66\u3002",
            )
        except Exception:
            logger.exception("Failed to send LINE error notification.")


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
            dedup_key = get_line_event_dedup_key(event)
            if not claim_line_event_receipt(event):
                logger.info("Skip persisted duplicate LINE event dedup_key=%s", dedup_key)
                continue
            if is_duplicate_line_event(event):
                logger.info("Skip in-memory duplicate LINE event dedup_key=%s", dedup_key)
                continue
            try:
                await handle_text_event(event)
            except Exception:
                logger.exception("Unhandled LINE event processing failure.")

    return {"status": "ok"}

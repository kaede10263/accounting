import os
import asyncio
import json
import sys
import types

import httpx
from fastapi.testclient import TestClient

import main


def setup_tmp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "accounting_test.db"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_PATH", str(db_path))
    main.pending_expense_by_user.clear()
    main.pending_delete_by_user.clear()
    main.last_query_by_user.clear()
    main.last_route_debug_by_user.clear()
    main.last_expense_parse_debug_by_user.clear()
    main.last_openai_route_plan = None
    main.last_openai_expense_parse = None
    main.in_flight_tasks.clear()
    main.pending_setting_changes_by_user.clear()
    main.pending_chart_cleanup_by_user.clear()
    main.recent_line_event_times.clear()
    main.init_db()
    with main.get_db() as conn:
        conn.execute("DELETE FROM line_event_receipts")
        conn.execute("DELETE FROM sql_change_logs")
        conn.commit()


def disable_payable_openai(monkeypatch):
    def raise_parser(text):
        raise RuntimeError("skip OpenAI in test")

    monkeypatch.setattr(main, "parse_payable_query_with_openai", raise_parser)


def count_expenses():
    with main.get_db() as conn:
        return int(conn.execute("SELECT COUNT(*) AS count FROM expenses").fetchone()["count"])


def expense_amounts():
    with main.get_db() as conn:
        return [
            int(row["amount"])
            for row in conn.execute("SELECT amount FROM expenses ORDER BY id").fetchall()
        ]


def fake_expense_parser(text):
    return main.ExpenseEntry(
        date="2026-07-07",
        time=None,
        amount=main.parse_amount(text) or 0,
        currency="TWD",
        category=main.infer_category(text),
        merchant=None,
        note=text,
        confidence=0.95,
    )


def make_line_text_event(
    text="\u6e2c\u8a66\u8a0a\u606f",
    user_id="U1",
    reply_token="reply-token",
    message_id="m1",
    source_type="user",
    group_id=None,
    room_id=None,
):
    source = {"type": source_type, "userId": user_id}
    if source_type == "group":
        source["groupId"] = group_id or "G1"
    if source_type == "room":
        source["roomId"] = room_id or "R1"
    return {
        "type": "message",
        "replyToken": reply_token,
        "source": source,
        "message": {"type": "text", "id": message_id, "text": text},
    }


def make_httpx_status_error(
    status_code=400,
    body='{"message":"Invalid reply token"}',
    url=main.LINE_REPLY_URL,
):
    request = httpx.Request("POST", url)
    response = httpx.Response(
        status_code,
        request=request,
        text=body,
        headers={"x-line-request-id": "test-request-id"},
    )
    return httpx.HTTPStatusError("LINE API error", request=request, response=response)


def test_watchdog_fast_process_replies_final_only(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    replies = []
    pushes = []

    async def fake_process(ctx, event):
        await asyncio.sleep(0.1)
        return "完成結果"

    async def fake_reply(reply_token, messages):
        replies.append((reply_token, messages))

    async def fake_push(user_id, payload):
        pushes.append((user_id, payload))

    monkeypatch.setattr(main, "process_message_with_context", fake_process)
    monkeypatch.setattr(main, "reply_line_messages", fake_reply)
    monkeypatch.setattr(main, "push_reply_payload", fake_push)

    asyncio.run(main.handle_text_event(make_line_text_event()))

    assert replies == [("reply-token", [{"type": "text", "text": "完成結果"}])]
    assert pushes == []


def test_watchdog_slow_process_replies_interim_then_pushes_final(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    replies = []
    pushes = []

    async def fake_process(ctx, event):
        ctx.status = "\u7b49\u5f85 ChatGPT \u56de\u61c9\u4e2d"
        await asyncio.sleep(3.2)
        return "最後結果"

    async def fake_reply(reply_token, messages):
        replies.append((reply_token, messages))

    async def fake_push(user_id, payload):
        pushes.append((user_id, payload))

    monkeypatch.setattr(main, "process_message_with_context", fake_process)
    monkeypatch.setattr(main, "reply_line_messages", fake_reply)
    monkeypatch.setattr(main, "push_reply_payload", fake_push)

    async def run_case():
        await main.handle_text_event(make_line_text_event())
        await asyncio.sleep(0.4)

    asyncio.run(run_case())

    assert len(replies) == 1
    assert "\u7b49\u5f85 ChatGPT \u56de\u61c9" in replies[0][1][0]["text"]
    assert pushes == [("U1", "最後結果")]


def test_line_chat_id_resolves_user_group_room():
    assert main.get_line_chat_id({"type": "user", "userId": "U1"}) == "U1"
    assert main.get_line_chat_id({"type": "group", "userId": "U1", "groupId": "G1"}) == "G1"
    assert main.get_line_chat_id({"type": "room", "userId": "U1", "roomId": "R1"}) == "R1"


def test_handle_text_event_falls_back_to_push_when_reply_token_invalid(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    replies = []
    pushes = []

    async def fake_process(ctx, event):
        raise RuntimeError("boom")

    async def fake_reply(reply_token, messages):
        replies.append((reply_token, messages))
        raise make_httpx_status_error()

    async def fake_push(chat_id, messages):
        pushes.append((chat_id, messages))

    monkeypatch.setattr(main, "process_message_with_context", fake_process)
    monkeypatch.setattr(main, "reply_line_messages", fake_reply)
    monkeypatch.setattr(main, "push_line_messages", fake_push)

    asyncio.run(main.handle_text_event(make_line_text_event()))

    expected = [{"type": "text", "text": "剛剛處理失敗了，我有記錄錯誤，請稍後再試。"}]
    assert replies == [("reply-token", expected)]
    assert pushes == [("U1", expected)]


def test_line_webhook_skips_duplicate_redelivery(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    handled = []

    async def fake_handle(event):
        handled.append(event["message"]["id"])

    monkeypatch.setattr(main, "verify_line_signature", lambda body, signature: True)
    monkeypatch.setattr(main, "handle_text_event", fake_handle)

    payload = {"events": [dict(make_line_text_event(message_id="m-dup"), webhookEventId="evt-dup")]}
    client = TestClient(main.app)

    first = client.post("/line/webhook", content=json.dumps(payload), headers={"X-Line-Signature": "ok"})
    main.recent_line_event_times.clear()
    second = client.post("/line/webhook", content=json.dumps(payload), headers={"X-Line-Signature": "ok"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert handled == ["m-dup"]


def test_watchdog_group_final_push_uses_group_id(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    replies = []
    pushes = []
    seen_ctx = []

    async def fake_process(ctx, event):
        seen_ctx.append(ctx)
        ctx.status = "\u7b49\u5f85 ChatGPT \u56de\u61c9\u4e2d"
        await asyncio.sleep(3.2)
        return "\u7fa4\u7d44\u6700\u7d42\u7d50\u679c"

    async def fake_reply(reply_token, messages):
        replies.append((reply_token, messages))

    async def fake_push(target_id, payload):
        pushes.append((target_id, payload))

    monkeypatch.setattr(main, "process_message_with_context", fake_process)
    monkeypatch.setattr(main, "reply_line_messages", fake_reply)
    monkeypatch.setattr(main, "push_reply_payload", fake_push)

    async def run_case():
        await main.handle_text_event(make_line_text_event(source_type="group", user_id="U1", group_id="G1"))
        await asyncio.sleep(0.4)

    asyncio.run(run_case())

    assert seen_ctx[0].actor_user_id == "U1"
    assert seen_ctx[0].chat_id == "G1"
    assert seen_ctx[0].source_type == "group"
    assert replies[0][0] == "reply-token"
    assert pushes == [("G1", "\u7fa4\u7d44\u6700\u7d42\u7d50\u679c")]
    assert all(target != "U1" for target, _payload in pushes)


def test_watchdog_room_final_push_uses_room_id(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    pushes = []

    async def fake_process(ctx, event):
        ctx.status = "\u67e5\u8a62\u8cc7\u6599\u5eab\u4e2d"
        await asyncio.sleep(3.2)
        return "\u804a\u5929\u5ba4\u6700\u7d42\u7d50\u679c"

    async def fake_reply(reply_token, messages):
        return None

    async def fake_push(target_id, payload):
        pushes.append((target_id, payload))

    monkeypatch.setattr(main, "process_message_with_context", fake_process)
    monkeypatch.setattr(main, "reply_line_messages", fake_reply)
    monkeypatch.setattr(main, "push_reply_payload", fake_push)

    async def run_case():
        await main.handle_text_event(make_line_text_event(source_type="room", user_id="U1", room_id="R1"))
        await asyncio.sleep(0.4)

    asyncio.run(run_case())

    assert pushes == [("R1", "\u804a\u5929\u5ba4\u6700\u7d42\u7d50\u679c")]


def test_interim_reply_status_texts():
    assert "\u7b49\u5f85 ChatGPT \u56de\u61c9" in main.build_interim_reply("\u7b49\u5f85 ChatGPT \u56de\u61c9\u4e2d")
    assert "\u88fd\u4f5c\u5716\u8868" in main.build_interim_reply("\u7522\u751f\u5716\u8868\u4e2d")


def test_watchdog_duplicate_message_reuses_in_flight_task(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    replies = []
    started = 0

    async def fake_process(ctx, event):
        nonlocal started
        started += 1
        ctx.status = "\u67e5\u8a62\u8cc7\u6599\u5eab\u4e2d"
        await asyncio.sleep(3.2)
        return "完成"

    async def fake_reply(reply_token, messages):
        replies.append((reply_token, messages))

    async def fake_push(user_id, payload):
        return None

    monkeypatch.setattr(main, "process_message_with_context", fake_process)
    monkeypatch.setattr(main, "reply_line_messages", fake_reply)
    monkeypatch.setattr(main, "push_reply_payload", fake_push)

    async def run_case():
        first = asyncio.create_task(main.handle_text_event(make_line_text_event(text="同一句", reply_token="r1", message_id="m1")))
        await asyncio.sleep(0.1)
        await main.handle_text_event(make_line_text_event(text="同一句", reply_token="r2", message_id="m2"))
        await first
        await asyncio.sleep(0.3)

    asyncio.run(run_case())

    assert started == 1
    assert any("\u6211\u9084\u5728\u8655\u7406\u4e0a\u4e00\u500b\u67e5\u8a62" in item[1][0]["text"] for item in replies)


def test_read_setting_shows_current_values(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("SSL_VERIFY", "true")

    reply = main.handle_special_command("readSetting", "U1")

    assert "OPENAI_MODEL" in reply
    assert "DATABASE_PATH" in reply
    assert "SSL_VERIFY" in reply
    assert "REMINDER" in reply
    assert "TIMEOUT" in reply


def test_write_setting_lists_editable_items(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    reply = main.handle_special_command("\u6539\u8a2d\u5b9a", "U1")

    assert "OPENAI_MODEL" in reply
    assert "SSL_VERIFY" in reply
    assert "REMINDER" in reply
    assert "TIMEOUT" in reply
    assert "PUBLIC_BASE_URL" in reply


def test_commands_help_lists_available_commands(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    reply = main.handle_special_command("commands", "U1")

    assert "可用指令" in reply
    assert "commands" in reply
    assert "checkSql" in reply
    assert "checkRoute" in reply
    assert "readSetting" in reply
    assert "writeSetting" in reply
    assert "usage" in reply
    assert "補充" not in reply


def test_setting_openai_model_requires_confirmation(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")

    reply = main.handle_special_command("\u8a2d\u5b9a OPENAI_MODEL gpt-4.1-mini", "U1")

    assert "\u5373\u5c07\u4fee\u6539" in reply
    assert main.get_setting("OPENAI_MODEL") is None


def test_confirm_setting_openai_model_writes_settings(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    main.handle_special_command("\u8a2d\u5b9a OPENAI_MODEL gpt-4.1-mini", "U1")
    reply = main.handle_special_command("\u78ba\u8a8d\u8a2d\u5b9a OPENAI_MODEL gpt-4.1-mini", "U1")

    assert "\u5df2\u66f4\u65b0\u8a2d\u5b9a" in reply
    assert main.get_setting("OPENAI_MODEL") == "gpt-4.1-mini"


def test_setting_database_path_is_rejected(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    reply = main.handle_special_command("\u8a2d\u5b9a DATABASE_PATH accounting_test.db", "U1")

    assert "DATABASE_PATH" in reply
    assert "\u4e0d\u5141\u8a31\u5f9e LINE \u4fee\u6539" in reply


def test_setting_reminder_and_timeout_write_internal_keys(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    main.handle_special_command("\u8a2d\u5b9a REMINDER 12", "U1")
    main.handle_special_command("\u78ba\u8a8d\u8a2d\u5b9a REMINDER 12", "U1")
    main.handle_special_command("\u8a2d\u5b9a TIMEOUT 3", "U1")
    main.handle_special_command("\u78ba\u8a8d\u8a2d\u5b9a TIMEOUT 3", "U1")

    assert main.get_setting("REMINDER_INTERVAL_HOURS") == "12"
    assert main.get_setting("PROCESSING_TIMEOUT_SECONDS") == "3"


def test_usage_command_shows_local_stats(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "CHART_DIR", tmp_path / "charts")
    (tmp_path / "charts").mkdir()
    (tmp_path / "charts" / "chart_a.png").write_bytes(b"png")
    main.log_usage("openai", "chat_completion", detail="gpt-4.1-mini", success=True, latency_ms=120)
    main.log_usage("line", "webhook", success=True)
    main.log_usage("line", "reply", success=True)
    main.log_usage("line", "push", success=True)
    main.log_usage("line", "image", success=True)

    reply = main.handle_special_command("usage", "U1")

    assert "ChatGPT" in reply
    assert "LINE" in reply
    assert "webhook" in reply
    assert "ngrok" in reply
    assert "\u5716\u8868\u5feb\u53d6" in reply
    assert "\u5716\u7247\u6578\u91cf\uff1a1" in reply


def test_recent_sql_changes_command_lists_today_insert_and_delete(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 10)
    expense = main.ExpenseEntry(
        date="2026-07-10",
        time=None,
        amount=50,
        currency="TWD",
        category="餐飲",
        merchant=None,
        note="早餐",
        confidence=0.9,
    )
    expense_id = main.save_expense(expense, "早餐50", "U1", "m-sql-log")
    main.delete_expense(f"刪除代號 {expense_id}", "U1")

    reply = main.handle_special_command("checkSql", "U1")

    assert reply.startswith("最近5筆 SQL 新增/刪除")
    assert "今天還沒有資料" not in reply
    assert "記帳 新增" in reply
    assert "記帳 刪除" in reply
    assert "#"+str(expense_id) in reply
    assert "早餐50" in reply


def test_recent_sql_changes_command_includes_household_entries(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 10)
    expense = main.ExpenseEntry(
        date="2026-07-10",
        time=None,
        amount=60,
        currency="TWD",
        category="交通",
        merchant=None,
        note="停車費",
        confidence=0.9,
    )
    expense_id = main.save_expense(expense, "停車費60", "U2", "m-sql-shared")

    reply = main.handle_special_command("checkSql", "U1")

    assert f"#{expense_id}" in reply
    assert "停車費60" in reply


def test_checkgpt_command_shows_last_parse_result(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    main.last_expense_parse_debug_by_user.clear()
    main.save_last_expense_parse_debug(
        "U1",
        "今天加油100",
        [
            {
                "normalized_text": "今天加油100",
                "source": "openai",
                "raw_model_output": '{"amount": 100, "category": "交通"}',
                "final_payload": {
                    "date": "2026-07-13",
                    "time": None,
                    "amount": 100,
                    "currency": "TWD",
                    "category": "交通",
                    "merchant": None,
                    "note": "加油",
                    "confidence": 0.95,
                },
            }
        ],
    )

    reply = main.handle_special_command("checkGpt", "U1")

    assert "最近一次 ChatGPT 解析" in reply
    assert "original_text：今天加油100" in reply
    assert "normalized_text：今天加油100" in reply
    assert 'raw_model_output：{"amount": 100, "category": "交通"}' in reply
    assert "final_payload：" in reply
    assert '"amount": 100' in reply
    assert '"category": "交通"' in reply


def test_checkroute_command_shows_last_openai_route_result(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    def fake_route_action_with_openai(text):
        main.last_openai_route_plan = {
            "input": text,
            "output": '{"operation":"query","target":"expenses","aggregation":"list"}',
        }
        return main.ActionRoute(
            action="list_expenses",
            should_mutate_db=False,
            confidence=0.91,
            reason="test openai route",
        )

    monkeypatch.setattr(main, "route_action_with_openai", fake_route_action_with_openai)

    route = main.route_action("香蕉芭樂蓮霧", "U1")
    reply = main.handle_special_command("checkRoute", "U1")

    assert route.action == "list_expenses"
    assert "最近一次 route 判斷" in reply
    assert "original_text：香蕉芭樂蓮霧" in reply
    assert "route_source：openai" in reply
    assert 'raw_route_model_output：{"operation":"query","target":"expenses","aggregation":"list"}' in reply
    assert '"action": "list_expenses"' in reply


def test_checkroute_command_shows_last_deterministic_route_result(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    route = main.route_action("薪資收入清單", "U1")
    reply = main.handle_special_command("checkRoute", "U1")

    assert route.action == "list_incomes"
    assert "route_source：deterministic" in reply
    assert "raw_route_model_output：(無)" in reply
    assert '"action": "list_incomes"' in reply


def test_semantic_parse_multi_expense_and_db_rows(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    parsed_inputs = []

    def fake_parser(text):
        parsed_inputs.append(text)
        return main.parse_expense_text_fallback(text)

    monkeypatch.setattr(main, "parse_expense_text", fake_parser)

    semantic_result = main.semantic_parse_multi_expense("7/10食材費用兩筆\n一筆521\n一筆859")

    assert semantic_result is not None
    assert parsed_inputs == ["7/10 食材費用 521", "7/10 食材費用 859"]
    assert [entry.normalized_text for entry in semantic_result.entries] == [
        "7/10 食材費用 521",
        "7/10 食材費用 859",
    ]

    db_rows = main.build_db_expense_rows(semantic_result, "m-food")

    assert [row.raw_text for row in db_rows] == ["7/10 食材費用 521", "7/10 食材費用 859"]
    assert [row.message_id for row in db_rows] == ["m-food:1", "m-food:2"]
    assert [row.expense.amount for row in db_rows] == [521, 859]


def test_clean_charts_command_removes_old_charts(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "CHART_DIR", tmp_path / "charts")
    (tmp_path / "charts").mkdir()
    old_file = tmp_path / "charts" / "old.png"
    new_file = tmp_path / "charts" / "new.png"
    old_file.write_bytes(b"old")
    new_file.write_bytes(b"new")
    old_timestamp = main.datetime.now(main.timezone.utc).timestamp() - 10 * 86400
    os.utime(old_file, (old_timestamp, old_timestamp))

    reply = main.handle_special_command("cleanCharts", "U1")

    assert "刪除 1" in reply
    assert not old_file.exists()
    assert new_file.exists()


def test_clean_charts_all_requires_confirmation(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "CHART_DIR", tmp_path / "charts")
    (tmp_path / "charts").mkdir()
    chart_file = tmp_path / "charts" / "chart_a.png"
    chart_file.write_bytes(b"png")

    first = main.handle_special_command("cleanCharts all", "U1")
    second = main.handle_special_command("\u78ba\u8a8d\u6e05\u7a7a\u5716\u8868", "U1")

    assert "\u78ba\u8a8d\u6e05\u7a7a\u5716\u8868" in first
    assert "刪除 1" in second
    assert not chart_file.exists()


def test_process_message_logs_utterance_after_route(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "route_action_with_openai",
        lambda text: main.ActionRoute(action="query_incomes", should_mutate_db=False, confidence=0.91, reason="test route"),
    )
    ctx = main.ProcessingContext(
        actor_user_id="U1",
        chat_id="U1",
        reply_token="reply-token",
        source_type="user",
        raw_text="\u6536\u5165\u6e05\u55ae",
        status="",
        started_at=main.datetime.now(main.timezone.utc),
    )

    main.process_message_sync(ctx, make_line_text_event(text="\u6536\u5165\u6e05\u55ae", user_id="U1", message_id="msg-log"))

    with main.get_db() as conn:
        row = conn.execute("SELECT * FROM utterance_logs WHERE message_id = ?", ("msg-log",)).fetchone()
    assert row is not None
    assert row["raw_text"] == "\u6536\u5165\u6e05\u55ae"
    assert row["actor_user_id"] == "U1"
    assert row["scope_id"] == "U1"
    assert row["predicted_domain"] == "finance"
    assert row["predicted_intent"] == "list_incomes"
    assert row["final_intent"] == "list_incomes"


def test_feedback_updates_last_utterance_log(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    route = main.ActionRoute(action="create_expense", should_mutate_db=True, confidence=0.8, reason="test")
    main.log_utterance("\u63d0\u9192\u6211\u4e0b\u9031\u516d\u6e05\u51b7\u6c23\u6ffe\u7db2", "m1", "U1", "user", "U1", route)

    reply = main.apply_last_utterance_feedback("\u4e0a\u4e00\u53e5\u5224\u932f\uff0c\u61c9\u8a72\u662f \u5c45\u5bb6\u63d0\u9192", "U1", "U1")

    assert "\u5df2\u8a18\u9304\u4fee\u6b63" in reply
    with main.get_db() as conn:
        row = conn.execute("SELECT final_domain, final_intent, is_correct, user_feedback FROM utterance_logs WHERE message_id = ?", ("m1",)).fetchone()
    assert row["final_domain"] == "reminder"
    assert row["final_intent"] == "create"
    assert row["is_correct"] == 0
    assert "\u5c45\u5bb6\u63d0\u9192" in row["user_feedback"]


def test_export_training_data_writes_jsonl(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    route = main.ActionRoute(action="create_expense", should_mutate_db=True, confidence=0.95)
    main.log_utterance("\u5348\u9910 120", "m1", "U1", "user", "U1", route)
    main.apply_last_utterance_feedback("\u4e0a\u4e00\u53e5\u61c9\u8a72\u662f \u623f\u5c4b\u4fee\u7e55", "U1", "U1")
    output_path = tmp_path / "training_intents.jsonl"

    count, path = main.export_training_data(str(output_path))

    assert count == 1
    assert path == str(output_path)
    line = output_path.read_text(encoding="utf-8").strip()
    assert '"text": "\u5348\u9910 120"' in line
    assert '"label": "home.create_maintenance_record"' in line


def test_large_planning_expense_requires_confirmation(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "parse_expense_text", fake_expense_parser)
    route = main.ActionRoute(action="create_expense", should_mutate_db=True, confidence=0.95)

    reply = main.execute_action_route(route, "買車 100萬", "U1", "m1")

    assert count_expenses() == 0
    assert "確認記帳" in reply
    assert "U1" in main.pending_expense_by_user


def test_confirm_pending_expense_saves_to_db(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "parse_expense_text", fake_expense_parser)
    route = main.ActionRoute(action="create_expense", should_mutate_db=True, confidence=0.95)

    main.execute_action_route(route, "買車 100萬", "U1", "m1")
    confirm_route = main.ActionRoute(action="chat", should_mutate_db=False, confidence=1.0)
    main.execute_action_route(confirm_route, "確認記帳", "U1", "m2")

    assert count_expenses() == 1
    assert expense_amounts() == [1_000_000]
    assert "U1" not in main.pending_expense_by_user


def test_overdraft_question_routes_to_balance(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "route_action_with_openai",
        lambda text: main.ActionRoute(
            action="query_expenses",
            should_mutate_db=False,
            confidence=0.9,
        ),
    )

    route = main.route_action("這個月是否透支？", "U1")

    assert route.action == "query_balance"


def test_income_aggregate_query_routes_to_query_incomes(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    def fail_openai(text):
        raise AssertionError("deterministic income query should run before OpenAI")

    monkeypatch.setattr(main, "route_action_with_openai", fail_openai)

    route = main.route_action("\u85aa\u8cc7\u6536\u5165\u662f\u591a\u5c11", "U1")

    assert route.action == "query_incomes"
    assert route.action != "query_balance"
    assert route.income_type == "\u85aa\u8cc7\u6536\u5165"


def test_salary_income_list_routes_to_list_incomes(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    def fail_openai(text):
        raise AssertionError("deterministic income list query should run before OpenAI")

    monkeypatch.setattr(main, "route_action_with_openai", fail_openai)

    route = main.route_action("\u85aa\u8cc7\u6536\u5165\u6e05\u55ae", "U1")

    assert route.action == "list_incomes"
    assert route.action != "query_balance"
    assert route.income_type == "\u85aa\u8cc7\u6536\u5165"


def test_income_list_routes_to_list_incomes(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    def fail_openai(text):
        raise AssertionError("deterministic income list query should run before OpenAI")

    monkeypatch.setattr(main, "route_action_with_openai", fail_openai)

    route = main.route_action("\u6536\u5165\u6e05\u55ae", "U1")

    assert route.action == "list_incomes"
    assert route.action != "query_balance"
    assert route.income_type is None


def test_this_month_income_total_routes_to_query_incomes(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    def fail_openai(text):
        raise AssertionError("deterministic income query should run before OpenAI")

    monkeypatch.setattr(main, "route_action_with_openai", fail_openai)

    route = main.route_action("\u9019\u500b\u6708\u6536\u5165\u662f\u591a\u5c11", "U1")

    assert route.action == "query_incomes"
    assert route.action != "query_balance"
    assert route.income_type is None


def test_balance_questions_route_to_query_balance(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    assert main.route_action("\u9019\u500b\u6708\u6536\u652f\u662f\u591a\u5c11", "U1").action == "query_balance"
    assert main.route_action("\u9019\u500b\u6708\u662f\u5426\u900f\u652f", "U1").action == "query_balance"
    assert main.route_action("\u9019\u500b\u6708\u9084\u5269\u591a\u5c11\u9322", "U1").action == "query_balance"


def test_small_simple_expense_saves_immediately(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "parse_expense_text", fake_expense_parser)
    route = main.ActionRoute(action="create_expense", should_mutate_db=True, confidence=0.95)

    reply = main.execute_action_route(route, "午餐 120", "U1", "m1")

    assert count_expenses() == 1
    assert "已記帳" in reply
    assert "確認記帳" not in reply


def test_multiline_yesterday_entries_save_two_rows(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 13)

    reply = main.execute_action_route(main.route_action("昨天\n停車費60\n飲料25", "U1"), "昨天\n停車費60\n飲料25", "U1", "m1")

    assert "已記帳多筆" in reply
    assert count_expenses() == 2
    with main.get_db() as conn:
        rows = conn.execute("SELECT date, amount, category, note FROM expenses ORDER BY id").fetchall()
    assert rows[0]["date"] == "2026-07-12"
    assert int(rows[0]["amount"]) == 60
    assert rows[0]["category"] == "交通"
    assert rows[1]["date"] == "2026-07-12"
    assert int(rows[1]["amount"]) == 25
    assert rows[1]["category"] == "餐飲"


def test_multiline_semantic_food_entries_parse_correctly(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    reply = main.execute_action_route(main.route_action("7/10食材費用兩筆\n一筆521\n一筆859", "U1"), "7/10食材費用兩筆\n一筆521\n一筆859", "U1", "m-food")

    assert "已記帳多筆" in reply
    assert "2026-07-10 食材費用 TWD 521" in reply
    assert "2026-07-10 食材費用 TWD 859" in reply
    with main.get_db() as conn:
        rows = conn.execute("SELECT date, amount, note FROM expenses ORDER BY id").fetchall()
    assert [row["date"] for row in rows] == ["2026-07-10", "2026-07-10"]
    assert [int(row["amount"]) for row in rows] == [521, 859]
    assert [row["note"] for row in rows] == ["食材費用", "食材費用"]


def test_list_and_delete_duplicate_data_flow(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "parse_expense_text", fake_expense_parser)
    route = main.ActionRoute(action="create_expense", should_mutate_db=True, confidence=0.95)

    main.execute_action_route(route, "早餐 50", "U1", "m1")
    main.execute_action_route(route, "早餐 50", "U1", "m2")

    route = main.route_action("列出來重複的資料", "U1")
    reply = main.execute_action_route(route, "列出來重複的資料", "U1", "m3")
    assert "可能重複的資料" in reply

    route = main.route_action("刪除重複的資料", "U1")
    reply = main.execute_action_route(route, "刪除重複的資料", "U1", "m4")
    assert "確認刪除重複資料" in reply

    confirm_route = main.ActionRoute(action="chat", should_mutate_db=False, confidence=1.0)
    reply = main.execute_action_route(confirm_route, "確認刪除重複資料", "U1", "m5")
    assert "已刪除重複資料" in reply
    assert count_expenses() == 1


def test_delete_requires_88_confirmation(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "parse_expense_text", fake_expense_parser)
    route = main.ActionRoute(action="create_expense", should_mutate_db=True, confidence=0.95)
    main.execute_action_route(route, "早餐 50", "U1", "m1")

    wrong_route = main.ActionRoute(action="chat", should_mutate_db=False, confidence=1.0)
    reply = main.execute_action_route(wrong_route, "88", "U1", "m2")

    assert "目前沒有等待確認刪除的記帳項目" in reply or "目前沒有等待確認" in reply


def seed_expense_statistics_data():
    rows = (
        ("stats-food-1", "2026-07-06", 175, "餐飲", "麵包"),
        ("stats-food-2", "2026-07-07", 65, "餐飲", "早餐"),
        ("stats-fuel-1", "2026-07-08", 100, "交通", "加油"),
        ("stats-food-3", "2026-07-12", 170, "餐飲", "宵夜"),
        ("stats-loan-1", "2026-07-08", 8500, "貸款", "房貸"),
        ("stats-shop-1", "2026-07-09", 1230, "購物", "尿布"),
    )
    for message_id, expense_date, amount, category, note in rows:
        main.save_expense(
            main.ExpenseEntry(
                date=expense_date,
                time=None,
                amount=amount,
                currency="TWD",
                category=category,
                merchant=None,
                note=note,
                confidence=0.95,
            ),
            f"{note} {amount}",
            "U1",
            message_id,
        )


def fake_bad_stat_parse(text):
    return main.ExpenseQuery(
        date_range_type="today",
        date_ranges=[main.DateRange(start_date="2026-07-08", end_date="2026-07-08")],
        aggregation="sum",
        confidence=0.95,
    )


def test_weekly_food_by_day_line_chart(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 8)
    seed_expense_statistics_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.ngrok-free.app")
    monkeypatch.setattr(main, "CHART_DIR", tmp_path / "charts")

    raw_text = "\u672c\u9031\u9910\u98f2\u6bcf\u5929\u82b1\u591a\u5c11\uff0c\u8acb\u7528\u6298\u7dda\u5716"
    query = main.parse_expense_query(raw_text)
    reply = main.build_expense_reply(raw_text, "U1")

    assert query.mode == "grouped_aggregate"
    assert query.group_by == ["day"]
    assert query.wants_chart is True
    assert query.chart_type == "line"
    assert "2026-07-06" in reply["text"]
    assert "2026-07-12" in reply["text"]
    assert reply["image_url"] is not None
    assert str(reply["image_url"]).startswith("https://example.ngrok-free.app/static/charts/")


def test_weekly_food_chart_route_corrects_list_to_query(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 8)
    seed_expense_statistics_data()
    monkeypatch.setattr(
        main,
        "route_action_with_openai",
        lambda text: main.ActionRoute(action="list_expenses", should_mutate_db=False, confidence=0.92),
    )
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)

    raw_text = "\u672c\u9031\u9910\u98f2\u6bcf\u5929\u82b1\u591a\u5c11 + \u6298\u7dda\u5716"
    route = main.route_action(raw_text, "U1")
    reply = main.execute_action_route(route, raw_text, "U1", "m-chart")

    assert route.action == "query_expenses"
    assert isinstance(reply, dict)
    assert "\u6bcf\u65e5\u82b1\u8cbb" in reply["text"]
    assert "2026-07-06" in reply["text"]
    assert "2026-07-12" in reply["text"]


def test_weekly_food_grouped_query_matches_legacy_food_category(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 8)
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)
    with main.get_db() as conn:
        conn.execute(
            """
            INSERT INTO expenses (
                line_user_id, message_id, raw_text, date, expense_time, amount,
                currency, category, merchant, note, confidence, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "U1",
                "legacy-food-1",
                "宵夜 100",
                "2026-07-07",
                "20:00",
                100,
                "TWD",
                "擗ㄡ",
                None,
                "宵夜",
                0.95,
                "2026-07-07T20:00:00+00:00",
            ),
        )
        conn.commit()

    raw_text = "\u672c\u9031\u9910\u98f2\u6bcf\u5929\u82b1\u591a\u5c11 + \u6298\u7dda\u5716"
    query = main.parse_expense_query(raw_text)
    result = main.execute_expense_query(query, "U1")
    rows = {row["group"]: row for row in result["rows"]}

    assert query.category == "\u9910\u98f2"
    assert rows["2026-07-07"]["total"] == 100


def test_chart_request_without_public_base_url_explains_missing_image(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 8)
    seed_expense_statistics_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.setattr(main, "PUBLIC_BASE_URL", None)

    reply = main.build_expense_reply("\u672c\u9031\u9910\u98f2\u6bcf\u5929\u82b1\u591a\u5c11 + \u6298\u7dda\u5716", "U1")

    assert reply["image_url"] is None
    assert "PUBLIC_BASE_URL" in reply["text"]


def seed_weekly_food_240_data():
    rows = (
        ("food-week-1", "2026-07-06", 175, "麵包"),
        ("food-week-2", "2026-07-07", 65, "早餐"),
    )
    for message_id, expense_date, amount, note in rows:
        main.save_expense(
            main.ExpenseEntry(
                date=expense_date,
                time=None,
                amount=amount,
                currency="TWD",
                category="\u9910\u98f2",
                merchant=None,
                note=note,
                confidence=0.95,
            ),
            f"{note} {amount}",
            "U1",
            message_id,
        )


def test_weekly_food_aggregate_has_total_and_item_names(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 8)
    seed_weekly_food_240_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)

    reply = main.build_expense_reply("\u672c\u9031\u9910\u98f2\u82b1\u591a\u5c11", "U1")

    assert "\u7b46\u6578\uff1a2" in reply["text"]
    assert "\u7e3d\u82b1\u8cbb\uff1aTWD 240" in reply["text"]
    assert "\u6700\u9ad8\uff1aTWD 175\uff08\u9eb5\u5305\uff09" in reply["text"]
    assert "\u6700\u4f4e\uff1aTWD 65\uff08\u65e9\u9910\uff09" in reply["text"]


def test_weekly_food_daily_uses_same_filter_as_aggregate(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 8)
    seed_weekly_food_240_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.ngrok-free.app")
    monkeypatch.setattr(main, "CHART_DIR", tmp_path / "charts")

    reply = main.build_expense_reply("\u672c\u9031\u9910\u98f2\u6bcf\u5929\u82b1\u591a\u5c11 + \u6298\u7dda\u5716", "U1")

    assert "2026-07-06\uff1aTWD 175" in reply["text"]
    assert "2026-07-07\uff1aTWD 65" in reply["text"]
    assert "\u5408\u8a08\uff1aTWD 240" in reply["text"]
    assert reply["image_url"] is not None


def test_weekly_food_daily_chart_suffix_does_not_change_query_result(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 8)
    seed_weekly_food_240_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.ngrok-free.app")
    monkeypatch.setattr(main, "CHART_DIR", tmp_path / "charts")

    plain_text = "\u672c\u9031\u9910\u98f2\u6bcf\u5929\u82b1\u591a\u5c11"
    chart_text = "\u672c\u9031\u9910\u98f2\u6bcf\u5929\u82b1\u591a\u5c11 + \u6298\u7dda\u5716"
    plain_query = main.parse_expense_query(plain_text)
    chart_query = main.parse_expense_query(chart_text)
    plain_result = main.execute_expense_query(plain_query, "U1")
    chart_result = main.execute_expense_query(chart_query, "U1")
    chart_reply = main.build_expense_reply(chart_text, "U1")

    assert plain_query.wants_chart is False
    assert chart_query.wants_chart is True
    assert chart_query.chart_type == "line"
    assert plain_query.date_ranges == chart_query.date_ranges
    assert plain_query.category == chart_query.category
    assert plain_query.group_by == chart_query.group_by
    assert plain_result["rows"] == chart_result["rows"]
    assert plain_result["total"] == chart_result["total"] == 240
    assert chart_reply["image_url"] is not None


def test_chart_public_base_url_does_not_affect_daily_query_data(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 8)
    seed_weekly_food_240_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.setattr(main, "PUBLIC_BASE_URL", None)

    plain_query = main.parse_expense_query("\u672c\u9031\u9910\u98f2\u6bcf\u5929\u82b1\u591a\u5c11")
    chart_query = main.parse_expense_query("\u672c\u9031\u9910\u98f2\u6bcf\u5929\u82b1\u591a\u5c11 + \u6298\u7dda\u5716")
    plain_result = main.execute_expense_query(plain_query, "U1")
    chart_result = main.execute_expense_query(chart_query, "U1")
    chart_reply = main.build_expense_reply("\u672c\u9031\u9910\u98f2\u6bcf\u5929\u82b1\u591a\u5c11 + \u6298\u7dda\u5716", "U1")

    assert plain_result["rows"] == chart_result["rows"]
    assert chart_result["total"] == 240
    assert "2026-07-06\uff1aTWD 175" in chart_reply["text"]
    assert "2026-07-07\uff1aTWD 65" in chart_reply["text"]
    assert chart_reply["image_url"] is None


def test_app_env_test_uses_accounting_test_db(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    assert main.get_app_env() == "test"
    assert main.get_db_path().endswith("accounting_test.db")


def test_get_db_path_requires_non_prod_name_under_test(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DB_PATH", "accounting.db")

    try:
        main.get_db_path()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "APP_ENV=test" in str(exc)


def test_checksql_limit_parsing():
    assert main.parse_recent_sql_changes_limit("checkSql") == 5
    assert main.parse_recent_sql_changes_limit("checkSql3") == 3
    assert main.parse_recent_sql_changes_limit("checkSql99") == 20
    assert main.parse_recent_sql_changes_limit("checkSql0") == 1
    assert main.parse_recent_sql_changes_limit("other") is None


def test_recent_sql_changes_empty(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    reply = main.handle_special_command("checkSql", "U1")

    assert reply.startswith("最近5筆 SQL 新增/刪除")
    assert "今天還沒有資料" in reply


def test_openai_route_failure_falls_back(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "route_action_with_openai", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))

    route = main.route_action("午餐 120", "U1")

    assert route.action == "create_expense"


def test_openai_expense_parse_failure_falls_back(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "parse_expense_text", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))

    expense = main.parse_expense_text_fallback("午餐 120")

    assert expense.amount == 120
    assert expense.category == "餐飲"


def test_parse_due_date_weekday_variants(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 14)

    assert main.parse_due_date("下禮拜三") == "2026-07-22"
    assert main.parse_due_date("下週三") == "2026-07-22"
    assert main.parse_due_date("下個星期一") == "2026-07-20"
    assert main.parse_due_date("星期六") == "2026-07-18"
    assert main.parse_due_date("這週三") == "2026-07-15"


def test_payable_query_unpaid_status(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    disable_payable_openai(monkeypatch)
    insert_payable(item_type="\u623f\u8cb8", amount=60000, due_date="2026-07-15", status="unpaid")

    reply = main.build_payable_query_reply("\u623f\u8cb8\u7e73\u4e86\u55ce?", "test_user")

    assert "還沒繳" in reply


def test_payable_query_paid_status(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    disable_payable_openai(monkeypatch)
    insert_payable(item_type="\u623f\u8cb8", amount=60000, due_date="2026-07-15", status="paid")

    reply = main.build_payable_query_reply("\u623f\u8cb8\u7e73\u4e86\u55ce?", "test_user")

    assert "已經繳了" in reply


def test_create_payable_draft_then_due_date(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    route = main.route_action("房貸 60000", "test_user")
    reply = main.execute_action_route(route, "房貸 60000", "test_user", "m-draft-1")
    assert "繳費日期" in reply

    route = main.route_action("7/15", "test_user")
    reply = main.execute_action_route(route, "7/15", "test_user", "m-draft-2")
    assert "已建立待繳提醒" in reply


def test_cancel_payable_draft(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    main.save_payable_draft("test_user", "房貸", 60000, None, None, "房貸 60000")
    reply = main.execute_action_route(main.ActionRoute(action="chat", should_mutate_db=False, confidence=1), "取消", "test_user", "m-cancel")

    assert "已取消這筆待繳提醒草稿" in reply


def test_available_cash_reply(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    with main.get_db() as conn:
        conn.execute(
            "INSERT INTO incomes (line_user_id, raw_text, income_date, amount, currency, income_type, item_name, owner, category, note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("U1", "薪水 50000", "2026-07-08", 50000, "TWD", "薪資收入", "薪水", None, "薪資收入", "薪水 50000", "2026-07-08T00:00:00+00:00"),
        )
        conn.commit()
    main.save_expense(main.ExpenseEntry(date="2026-07-08", time=None, amount=1000, currency="TWD", category="餐飲", merchant=None, note="聚餐", confidence=0.95), "聚餐 1000", "U1", "m-avail")

    reply = main.build_available_cash_reply("有多少錢可以買玩具?", "U1", main.ActionRoute(action="query_available_cash", should_mutate_db=False, confidence=0.9, purchase_purpose="玩具"))

    assert "目前可動用金額" in reply
    assert "玩具" in reply


def test_available_investment_cash_reply(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    with main.get_db() as conn:
        conn.execute(
            "INSERT INTO incomes (line_user_id, raw_text, income_date, amount, currency, income_type, item_name, owner, category, note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("U1", "薪水 50000", "2026-07-08", 50000, "TWD", "薪資收入", "薪水", None, "薪資收入", "薪水 50000", "2026-07-08T00:00:00+00:00"),
        )
        conn.commit()

    reply = main.build_available_investment_cash_reply("可以投資多少?", "U1")

    assert "投資可動用金額" in reply
    assert "保守建議投入" in reply


def test_cleanup_old_charts(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "CHART_DIR", tmp_path / "charts")
    (tmp_path / "charts").mkdir()
    old_file = tmp_path / "charts" / "old.png"
    new_file = tmp_path / "charts" / "new.png"
    old_file.write_bytes(b"old")
    new_file.write_bytes(b"new")
    old_timestamp = main.datetime.now(main.timezone.utc).timestamp() - 10 * 86400
    os.utime(old_file, (old_timestamp, old_timestamp))

    deleted = main.cleanup_old_charts(days=7)

    assert deleted == 1
    assert not old_file.exists()
    assert new_file.exists()


def test_monthly_food_bar_chart_fills_zero_month(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_expense_statistics_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.ngrok-free.app")
    monkeypatch.setattr(main, "CHART_DIR", tmp_path / "charts")

    raw_text = "2 3 4 5 6\u6708\u9910\u98f2\u6bcf\u6708\u7e3d\u82b1\u8cbb\uff0c\u7528\u9577\u689d\u5716"
    query = main.parse_expense_query(raw_text)
    result = main.execute_expense_query(query, "U1")
    reply = main.build_expense_reply(raw_text, "U1")

    assert query.group_by == ["month"]
    assert query.chart_type == "bar"
    rows = {row["group"]: row for row in result["rows"]}
    assert rows["2026-04"]["total"] == 0
    assert reply["image_url"] is not None


def test_monthly_line_chart_fills_full_month_range(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 8)
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.ngrok-free.app")
    monkeypatch.setattr(main, "CHART_DIR", tmp_path / "charts")
    captured = {}

    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.use = lambda backend: None
    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_pyplot.figure = lambda **kwargs: None
    fake_pyplot.plot = lambda labels, values, marker=None: captured.update({"labels": labels, "values": values})
    fake_pyplot.xticks = lambda *args, **kwargs: None
    fake_pyplot.ylabel = lambda *args, **kwargs: None
    fake_pyplot.tight_layout = lambda: None
    fake_pyplot.savefig = lambda path, dpi=None: path.write_bytes(b"png")
    fake_pyplot.close = lambda: None
    fake_pyplot.pie = lambda *args, **kwargs: None
    fake_pyplot.axis = lambda *args, **kwargs: None
    fake_pyplot.bar = lambda *args, **kwargs: None
    fake_matplotlib.pyplot = fake_pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)

    main.save_expense(
        main.ExpenseEntry(
            date="2026-07-07",
            time=None,
            amount=175,
            currency="TWD",
            category="\u9910\u98f2",
            merchant=None,
            note="\u9eb5\u5305",
            confidence=0.95,
        ),
        "\u9eb5\u5305 175",
        "U1",
        "month-line-food",
    )
    main.save_expense(
        main.ExpenseEntry(
            date="2026-07-08",
            time=None,
            amount=62218,
            currency="TWD",
            category="\u8cb8\u6b3e",
            merchant=None,
            note="\u623f\u8cb8",
            confidence=0.95,
        ),
        "\u623f\u8cb8 62218",
        "U1",
        "month-line-mortgage",
    )

    raw_text = "\u6392\u9664\u623f\u8cb8 \u7d66\u6211\u4e00\u6708\u5230\u4e03\u6708\u6bcf\u500b\u6708\u7684\u82b1\u8cbb + \u6298\u7dda\u5716"
    query = main.parse_expense_query(raw_text)
    result = main.execute_expense_query(query, "U1")
    reply = main.build_expense_reply(raw_text, "U1")
    expected_months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07"]
    expected_amounts = [0, 0, 0, 0, 0, 0, 175]

    assert query.group_by == ["month"]
    assert query.chart_type == "line"
    assert [row["group"] for row in result["rows"]] == expected_months
    assert [row["total"] for row in result["rows"]] == expected_amounts
    assert [row["count"] for row in result["rows"]] == [0, 0, 0, 0, 0, 0, 1]
    for month in expected_months:
        assert month in reply["text"]
    assert "62218" not in reply["text"]
    assert captured["labels"] == expected_months
    assert captured["values"] == expected_amounts
    assert reply["image_url"] is not None


def test_category_ratio_pie_chart(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_expense_statistics_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.ngrok-free.app")
    monkeypatch.setattr(main, "CHART_DIR", tmp_path / "charts")

    raw_text = "\u4e03\u6708\u5404\u985e\u5225\u7684\u82b1\u8cbb\u4f54\u6bd4\uff0c\u756b\u5713\u9905\u5716"
    query = main.parse_expense_query(raw_text)
    reply = main.build_expense_reply(raw_text, "U1")

    assert query.group_by == ["category"]
    assert query.include_ratio is True
    assert query.chart_type == "pie"
    assert reply["image_url"] is not None


def test_category_ratio_excluding_mortgage_bar_chart(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_expense_statistics_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.ngrok-free.app")
    monkeypatch.setattr(main, "CHART_DIR", tmp_path / "charts")

    raw_text = "\u4e03\u6708\u5404\u985e\u5225\u7684\u82b1\u8cbb\u4f54\u6bd4\uff0c\u4e0d\u8981\u8a08\u7b97\u623f\u8cb8\uff0c\u7528\u9577\u689d\u5716"
    query = main.parse_expense_query(raw_text)
    result = main.execute_expense_query(query, "U1")
    rows = {row["group"]: row for row in result["rows"]}

    assert query.exclude_keywords == ["\u623f\u8cb8"]
    assert query.ratio_denominator == "filtered_expenses"
    assert query.chart_type == "bar"
    assert result["denominator_total"] == 10070
    assert rows["\u8cb8\u6b3e"]["total"] == 8500


def test_category_ratio_without_chart_has_no_image(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_expense_statistics_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)

    raw_text = "\u4e03\u6708\u5404\u985e\u5225\u7684\u82b1\u8cbb\u4f54\u6bd4"
    query = main.parse_expense_query(raw_text)
    reply = main.build_expense_reply(raw_text, "U1")

    assert query.wants_chart is False
    assert reply["image_url"] is None


def test_chart_without_public_base_url_does_not_raise(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_expense_statistics_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.setattr(main, "PUBLIC_BASE_URL", None)

    raw_text = "\u4e03\u6708\u5404\u985e\u5225\u7684\u82b1\u8cbb\u4f54\u6bd4\uff0c\u756b\u5713\u9905\u5716"
    reply = main.build_expense_reply(raw_text, "U1")

    assert reply["image_url"] is None
    assert "\u67e5\u8a62\u7d50\u679c" in reply["text"]


def insert_payable(
    line_user_id="test_user",
    item_type="\u623f\u8cb8",
    amount=60000,
    due_date="2026-07-15",
    status="unpaid",
):
    with main.get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO payables (
                line_user_id, item_type, amount, currency, due_date,
                owner, bank, note, status, paid_at, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                line_user_id,
                item_type,
                amount,
                "TWD",
                due_date,
                None,
                None,
                item_type,
                status,
                "2026-07-16T00:00:00+00:00" if status == "paid" else None,
                "2026-07-07T00:00:00+00:00",
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def insert_home_task(
    chat_id="family-1",
    actor_user_id="test_user",
    title="換地漏",
    item_key="地漏",
    category="居家修繕",
    scheduled_date="2026-07-22",
    status="pending",
    completed_at=None,
    last_reminded_at=None,
):
    with main.get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO home_tasks (
                actor_user_id, chat_id, message_id, title, item_key, category,
                scheduled_date, scheduled_time, status, completed_at, completion_text,
                last_reminded_at, raw_text, created_at, updated_at
            )
            VALUES (?, ?, NULL, ?, ?, ?, ?, NULL, ?, ?, NULL, ?, ?, ?, ?)
            """,
            (
                actor_user_id,
                chat_id,
                title,
                item_key,
                category,
                scheduled_date,
                status,
                completed_at,
                last_reminded_at,
                title,
                "2026-07-14T00:00:00+08:00",
                "2026-07-14T00:00:00+08:00",
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def set_fake_today(monkeypatch, year=2026, month=7, day=8):
    class FakeDate(main.date):
        @classmethod
        def today(cls):
            return cls(year, month, day)

    monkeypatch.setattr(main, "date", FakeDate)


def count_payables(where="1 = 1", params=()):
    with main.get_db() as conn:
        return int(conn.execute(f"SELECT COUNT(*) AS count FROM payables WHERE {where}", params).fetchone()["count"])


def test_route_mortgage_paid_question_to_query_payables(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "route_action_with_openai",
        lambda text: main.ActionRoute(action="create_payable", should_mutate_db=True, confidence=0.9),
    )

    route = main.route_action("\u623f\u8cb8\u7e73\u4e86\u55ce?", "test_user")

    assert route.action == "query_payables"
    assert route.should_mutate_db is False
    assert route.item_type == "\u623f\u8cb8"


def test_route_this_month_mortgage_question_to_query_payables(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "route_action_with_openai",
        lambda text: main.ActionRoute(action="create_payable", should_mutate_db=True, confidence=0.9),
    )

    route = main.route_action("\u9019\u500b\u6708\u623f\u8cb8\u7e73\u4e86\u55ce?", "test_user")

    assert route.action == "query_payables"
    assert route.should_mutate_db is False
    assert route.item_type == "\u623f\u8cb8"


def test_mortgage_paid_text_routes_to_mark_paid_and_updates_db(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 8)
    insert_payable(line_user_id="test_user", item_type="\u623f\u8cb8", amount=62218, due_date="2026-07-08", status="unpaid")

    def fail_openai(text):
        raise AssertionError("deterministic paid update should run before OpenAI")

    monkeypatch.setattr(main, "route_action_with_openai", fail_openai)

    route = main.route_action("\u623f\u8cb8\u5df2\u7e73", "test_user")
    reply = main.execute_action_route(route, "\u623f\u8cb8\u5df2\u7e73", "test_user", "paid-m1")

    assert route.action == "mark_payable_paid"
    assert route.should_mutate_db is True
    assert route.item_type == "\u623f\u8cb8"
    assert "\u5df2\u6a19\u8a18\u70ba\u5df2\u7e73" in reply
    with main.get_db() as conn:
        row = conn.execute("SELECT status, paid_at FROM payables WHERE item_type = ?", ("\u623f\u8cb8",)).fetchone()
    assert row["status"] == "paid"
    assert row["paid_at"] is not None


def test_mortgage_paid_question_still_routes_to_query(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 8)

    route = main.route_action("\u623f\u8cb8\u7e73\u4e86\u55ce\uff1f", "test_user")

    assert route.action == "query_payables"
    assert route.should_mutate_db is False


def test_query_unpaid_mortgage_reply(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    disable_payable_openai(monkeypatch)
    insert_payable()

    reply = main.build_payable_query_reply("\u9019\u500b\u6708\u623f\u8cb8\u7e73\u4e86\u55ce?", "test_user")

    assert "\u9084\u6c92\u7e73" in reply
    assert "7/15" in reply
    assert "60000" in reply


def test_query_paid_mortgage_reply(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    disable_payable_openai(monkeypatch)
    insert_payable(status="paid")

    reply = main.build_payable_query_reply("\u9019\u500b\u6708\u623f\u8cb8\u7e73\u4e86\u55ce?", "test_user")

    assert "\u5df2\u7d93\u7e73\u4e86" in reply
    assert "\u6c92\u6709\u7e73\u8cbb\u7d00\u9304" not in reply


def test_query_no_mortgage_data_reply(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    disable_payable_openai(monkeypatch)

    reply = main.build_payable_query_reply("\u9019\u500b\u6708\u623f\u8cb8\u7e73\u4e86\u55ce?", "test_user")

    assert "\u9019\u500b\u6708\u6c92\u6709\u623f\u8cb8\u7e73\u8cbb\u7d00\u9304" in reply


def test_send_due_reminders_push_success_records(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    payable_id = insert_payable(due_date="2026-07-15")

    class FakeDate(main.date):
        @classmethod
        def today(cls):
            return cls(2026, 7, 12)

    pushed = []

    async def fake_push(line_user_id, text):
        pushed.append((line_user_id, text))

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "push_line_message", fake_push)

    asyncio.run(main.send_due_reminders())

    assert len(pushed) == 1
    with main.get_db() as conn:
        row = conn.execute(
            "SELECT * FROM payable_reminders WHERE payable_id = ?",
            (payable_id,),
        ).fetchone()
    assert row is not None
    assert int(row["remind_days_before"]) == 3


def test_send_due_reminders_push_failure_does_not_record(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    payable_id = insert_payable(due_date="2026-07-15")

    class FakeDate(main.date):
        @classmethod
        def today(cls):
            return cls(2026, 7, 12)

    async def fake_push(line_user_id, text):
        raise RuntimeError("push failed")

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "push_line_message", fake_push)

    asyncio.run(main.send_due_reminders())

    with main.get_db() as conn:
        row = conn.execute(
            "SELECT * FROM payable_reminders WHERE payable_id = ?",
            (payable_id,),
        ).fetchone()
    assert row is None


def test_create_payable_reply_lists_reminder_dates(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    reply = main.create_payable("test_user", "\u623f\u8cb8", 60000, "2026-07-15", "\u623f\u8cb8 7/15 6\u842c")

    assert "\u5df2\u5efa\u7acb\u5f85\u7e73\u63d0\u9192" in reply
    assert "\u63d0\u9192\u65e5\u671f\uff1a2026-07-12\u30012026-07-13\u30012026-07-14" in reply
    assert "\u63d0\u9192\u65b9\u5f0f\uff1aLINE push" in reply


def test_route_payable_reminder_create_with_date_amount(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 8)

    def fail_openai(text):
        raise AssertionError("deterministic payable create should run before OpenAI")

    monkeypatch.setattr(main, "route_action_with_openai", fail_openai)

    route = main.route_action("\u63d0\u9192\u6211 7/8 \u8981\u7e73\u623f\u8cb8 62218", "test_user")

    assert route.action == "create_payable"
    assert route.should_mutate_db is True
    assert route.item_type == "\u623f\u8cb8"
    assert route.amount == 62218
    assert route.due_date == "2026-07-08"

    reply = main.execute_action_route(route, "\u63d0\u9192\u6211 7/8 \u8981\u7e73\u623f\u8cb8 62218", "test_user", "payable-m1")

    assert "\u5df2\u5efa\u7acb\u5f85\u7e73\u63d0\u9192" in reply
    with main.get_db() as conn:
        row = conn.execute("SELECT * FROM payables WHERE line_user_id = ?", ("test_user",)).fetchone()
    assert row["item_type"] == "\u623f\u8cb8"
    assert int(row["amount"]) == 62218
    assert row["due_date"] == "2026-07-08"
    assert row["status"] == "unpaid"


def test_route_payable_create_plain_order(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 8)

    def fail_openai(text):
        raise AssertionError("deterministic payable create should run before OpenAI")

    monkeypatch.setattr(main, "route_action_with_openai", fail_openai)

    route = main.route_action("\u623f\u8cb8 62218 7/8", "test_user")

    assert route.action == "create_payable"
    assert route.item_type == "\u623f\u8cb8"
    assert route.amount == 62218
    assert route.due_date == "2026-07-08"


def test_route_payable_create_credit_card_owner_bank(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 8)

    def fail_openai(text):
        raise AssertionError("deterministic payable create should run before OpenAI")

    monkeypatch.setattr(main, "route_action_with_openai", fail_openai)

    route = main.route_action("\u8001\u5a46\u7389\u5c71\u4fe1\u7528\u5361 590 7/8", "test_user")

    assert route.action == "create_payable"
    assert route.item_type == "\u4fe1\u7528\u5361"
    assert route.owner == "\u8001\u5a46"
    assert route.bank == "\u7389\u5c71"
    assert route.amount == 590
    assert route.due_date == "2026-07-08"


def test_create_payable_dedupes_by_message_id(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    first = main.create_payable("test_user", "\u4fe1\u8cb8", 500, "2026-07-08", "\u4fe1\u8cb8 7/8 500", message_id="m1")
    second = main.create_payable("test_user", "\u4fe1\u8cb8", 500, "2026-07-08", "\u4fe1\u8cb8 7/8 500", message_id="m1")

    assert "\u5df2\u5efa\u7acb\u5f85\u7e73\u63d0\u9192" in first
    assert "\u9019\u7b46\u5f85\u7e73\u6b3e\u5df2\u7d93\u5efa\u7acb\u904e" in second
    assert count_payables("line_user_id = ? AND item_type = ?", ("test_user", "\u4fe1\u8cb8")) == 1


def test_create_payable_dedupes_without_message_id(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    main.create_payable("test_user", "\u4fe1\u8cb8", 500, "2026-07-08", "\u4fe1\u8cb8 7/8 500")
    reply = main.create_payable("test_user", "\u4fe1\u8cb8", 500, "2026-07-08", "\u4fe1\u8cb8 7/8 500")

    assert "\u9019\u7b46\u5f85\u7e73\u6b3e\u5df2\u7d93\u5b58\u5728" in reply
    assert count_payables("line_user_id = ? AND item_type = ? AND status = 'unpaid'", ("test_user", "\u4fe1\u8cb8")) == 1


def test_send_due_reminders_same_payable_only_pushes_once_per_day(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    insert_payable(item_type="\u4fe1\u8cb8", amount=500, due_date="2026-07-08")
    set_fake_today(monkeypatch, 2026, 7, 5)
    pushed = []

    async def fake_push(line_user_id, text):
        pushed.append((line_user_id, text))

    monkeypatch.setattr(main, "push_line_message", fake_push)

    asyncio.run(main.send_due_reminders())
    asyncio.run(main.send_due_reminders())

    assert len(pushed) == 1
    with main.get_db() as conn:
        reminders = conn.execute("SELECT * FROM payable_reminders").fetchall()
    assert len(reminders) == 1
    assert int(reminders[0]["remind_days_before"]) == 3


def test_dedupe_payables_keeps_one_duplicate_unpaid_loan(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    with main.get_db() as conn:
        conn.execute("DROP INDEX IF EXISTS idx_payables_dedupe")
        conn.commit()
    for _ in range(10):
        insert_payable(item_type="\u4fe1\u8cb8", amount=500, due_date="2026-07-08")

    deleted_count = main.dedupe_payables()

    assert deleted_count == 9
    assert count_payables("item_type = ? AND amount = ? AND due_date = ? AND status = 'unpaid'", ("\u4fe1\u8cb8", 500, "2026-07-08")) == 1


def test_mark_paid_credit_loan_does_not_use_generic_loan(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch)
    insert_payable(item_type="\u4fe1\u8cb8", amount=500, due_date="2026-07-08")
    insert_payable(item_type="\u8cb8\u6b3e", amount=700, due_date="2026-07-08")

    reply = main.mark_payable_paid("\u5df2\u7e73 \u4fe1\u8cb8", "test_user")

    assert "\u4fe1\u8cb8" in reply
    assert "\u8cb8\u6b3e" not in reply
    with main.get_db() as conn:
        credit = conn.execute("SELECT status FROM payables WHERE item_type = ?", ("\u4fe1\u8cb8",)).fetchone()
        generic = conn.execute("SELECT status FROM payables WHERE item_type = ?", ("\u8cb8\u6b3e",)).fetchone()
    assert credit["status"] == "paid"
    assert generic["status"] == "unpaid"


def test_query_credit_loan_after_paid_lists_only_remaining_unpaid(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    disable_payable_openai(monkeypatch)
    set_fake_today(monkeypatch)
    insert_payable(item_type="\u4fe1\u8cb8", amount=500, due_date="2026-07-08")
    insert_payable(item_type="\u4fe1\u8cb8", amount=7, due_date="2026-07-16")

    main.mark_payable_paid("\u5df2\u7e73 \u4fe1\u8cb8", "test_user")
    reply = main.build_payable_query_reply("\u9019\u500b\u6708\u9084\u6709\u4fe1\u8cb8\u9084\u6c92\u7e73\u55ce?", "test_user")

    assert "7/16" in reply
    assert "7 \u5143" in reply
    assert "7/8" not in reply
    assert "500" not in reply


def test_mortgage_credit_loan_generic_loan_are_distinct(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    assert main.normalize_payable_item_type("\u623f\u8cb8") == "\u623f\u8cb8"
    assert main.normalize_payable_item_type("\u4fe1\u8cb8") == "\u4fe1\u8cb8"
    assert main.normalize_payable_item_type("\u8cb8\u6b3e") == "\u8cb8\u6b3e"


def test_mark_paid_mortgage_does_not_affect_credit_loan(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch)
    insert_payable(item_type="\u623f\u8cb8", amount=60000, due_date="2026-07-15")
    insert_payable(item_type="\u4fe1\u8cb8", amount=500, due_date="2026-07-08")

    main.mark_payable_paid("\u5df2\u7e73 \u623f\u8cb8", "test_user")

    with main.get_db() as conn:
        mortgage = conn.execute("SELECT status FROM payables WHERE item_type = ?", ("\u623f\u8cb8",)).fetchone()
        credit = conn.execute("SELECT status FROM payables WHERE item_type = ?", ("\u4fe1\u8cb8",)).fetchone()
    assert mortgage["status"] == "paid"
    assert credit["status"] == "unpaid"


def test_mark_paid_credit_loan_does_not_affect_mortgage(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch)
    insert_payable(item_type="\u623f\u8cb8", amount=60000, due_date="2026-07-15")
    insert_payable(item_type="\u4fe1\u8cb8", amount=500, due_date="2026-07-08")

    main.mark_payable_paid("\u5df2\u7e73 \u4fe1\u8cb8", "test_user")

    with main.get_db() as conn:
        mortgage = conn.execute("SELECT status FROM payables WHERE item_type = ?", ("\u623f\u8cb8",)).fetchone()
        credit = conn.execute("SELECT status FROM payables WHERE item_type = ?", ("\u4fe1\u8cb8",)).fetchone()
    assert mortgage["status"] == "unpaid"
    assert credit["status"] == "paid"


def test_route_home_task_create_next_weekday(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 14)

    route = main.route_action("下禮拜三提醒我要換地漏", "U1")

    assert route.action == "create_home_task"
    assert route.task_title == "換地漏"
    assert route.task_item_key == "地漏"
    assert route.task_category == "居家修繕"
    assert route.scheduled_date == "2026-07-22"
    assert route.should_mutate_db is True


def test_route_home_task_create_tomorrow_not_chat(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 14)

    route = main.route_action("明天記得洗冷氣", "U1")

    assert route.action == "create_home_task"
    assert route.task_title == "洗冷氣"
    assert route.scheduled_date == "2026-07-15"


def test_execute_home_task_create_persists_row(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 14)

    raw_text = "7/30 要換濾芯"
    route = main.route_action(raw_text, "U1")
    reply = main.execute_action_route(route, raw_text, "U1", "task-m1", "family-1")

    assert route.action == "create_home_task"
    assert "[chat]" not in reply
    assert "已建立家庭事項" in reply
    with main.get_db() as conn:
        row = conn.execute(
            "SELECT title, item_key, scheduled_date, status, chat_id, actor_user_id FROM home_tasks WHERE message_id = ?",
            ("task-m1",),
        ).fetchone()
    assert row["title"] == "換濾芯"
    assert row["item_key"] == "濾芯"
    assert row["scheduled_date"] == "2026-07-30"
    assert row["status"] == "pending"
    assert row["chat_id"] == "family-1"
    assert row["actor_user_id"] == "U1"


def test_ambiguous_home_task_create_uses_sqlite_draft_confirmation(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 14)

    raw_text = "下禮拜三可能要換地漏"
    route = main.route_action(raw_text, "U1")
    reply = main.execute_action_route(route, raw_text, "U1", "draft-m1", "family-1")

    assert route.action == "create_home_task"
    assert route.task_requires_confirmation is True
    assert "請回覆「要」或「不要」" in reply
    with main.get_db() as conn:
        draft = conn.execute("SELECT title, scheduled_date FROM home_task_drafts WHERE chat_id = ?", ("family-1",)).fetchone()
    assert draft["title"] == "換地漏"
    assert draft["scheduled_date"] == "2026-07-22"

    confirm_route = main.route_action("要", "U1")
    confirm_reply = main.execute_action_route(confirm_route, "要", "U1", "draft-m2", "family-1")

    assert "已建立家庭事項" in confirm_reply
    with main.get_db() as conn:
        draft = conn.execute("SELECT * FROM home_task_drafts WHERE chat_id = ?", ("family-1",)).fetchone()
        task = conn.execute("SELECT title, scheduled_date, status FROM home_tasks WHERE chat_id = ?", ("family-1",)).fetchone()
    assert draft is None
    assert task["title"] == "換地漏"
    assert task["scheduled_date"] == "2026-07-22"
    assert task["status"] == "pending"


def test_complete_home_task_updates_pending_row(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 22)
    insert_home_task(chat_id="family-1", scheduled_date="2026-07-22")

    route = main.route_action("已換地漏", "U1")
    reply = main.execute_action_route(route, "已換地漏", "U1", "done-m1", "family-1")

    assert route.action == "complete_home_task"
    assert "已完成家庭事項" in reply
    with main.get_db() as conn:
        row = conn.execute(
            "SELECT status, completion_text, completed_at FROM home_tasks WHERE chat_id = ?",
            ("family-1",),
        ).fetchone()
    assert row["status"] == "completed"
    assert row["completion_text"] == "已換地漏"
    assert str(row["completed_at"]).startswith("2026-07-22")


def test_cancel_home_task_with_date_and_delete_verb(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    set_fake_today(monkeypatch, 2026, 7, 14)
    insert_home_task(chat_id="family-1", title="換地漏", item_key="地漏", scheduled_date="2026-07-23", status="pending")

    route = main.route_action("刪除 下星期四 換地漏", "U1")
    reply = main.execute_action_route(route, "刪除 下星期四 換地漏", "U1", "cancel-m1", "family-1")

    assert route.action == "cancel_home_task"
    assert "已取消家庭事項" in reply
    with main.get_db() as conn:
        row = conn.execute(
            "SELECT status FROM home_tasks WHERE chat_id = ? AND title = ?",
            ("family-1", "換地漏"),
        ).fetchone()
    assert row["status"] == "cancelled"


def test_generic_completion_picks_recently_reminded_task(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    fake_now = main.datetime(2026, 7, 22, 10, 0, tzinfo=main.TAIPEI_TZ)
    monkeypatch.setattr(main, "taipei_now", lambda: fake_now)
    insert_home_task(chat_id="family-1", title="換地漏", item_key="地漏", scheduled_date="2026-07-22", last_reminded_at="2026-07-22T09:00:00+08:00")
    insert_home_task(chat_id="family-1", title="洗冷氣", item_key="冷氣", scheduled_date="2026-07-23")

    route = main.route_action("已換", "U1")
    reply = main.execute_action_route(route, "已換", "U1", "done-m2", "family-1")

    assert route.action == "complete_home_task"
    assert "換地漏" in reply
    with main.get_db() as conn:
        rows = conn.execute(
            "SELECT title, status FROM home_tasks WHERE chat_id = ? ORDER BY id",
            ("family-1",),
        ).fetchall()
    assert rows[0]["title"] == "換地漏"
    assert rows[0]["status"] == "completed"
    assert rows[1]["title"] == "洗冷氣"
    assert rows[1]["status"] == "pending"


def test_query_home_task_history_uses_completed_at(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    insert_home_task(
        chat_id="family-1",
        scheduled_date="2026-07-22",
        status="completed",
        completed_at="2026-07-24T08:30:00+08:00",
    )

    route = main.route_action("上次換地漏是什麼時候？", "U1")
    reply = main.execute_action_route(route, "上次換地漏是什麼時候？", "U1", "history-m1", "family-1")

    assert route.action == "query_home_task_history"
    assert "2026-07-24" in reply
    assert "原定日期是 2026-07-22" in reply


def test_query_home_tasks_returns_pending_only(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    insert_home_task(chat_id="family-1", title="換地漏", item_key="地漏", scheduled_date="2026-07-22", status="pending")
    insert_home_task(chat_id="family-1", title="洗冷氣", item_key="冷氣", scheduled_date="2026-07-25", status="completed", completed_at="2026-07-25T10:00:00+08:00")

    route = main.route_action("還有哪些家庭事項沒完成？", "U1")
    reply = main.execute_action_route(route, "還有哪些家庭事項沒完成？", "U1", "query-m1", "family-1")

    assert route.action == "query_home_tasks"
    assert "尚未完成的家庭事項" in reply
    assert "換地漏" in reply
    assert "洗冷氣" not in reply


def test_send_home_task_reminders_pushes_once_and_updates_timestamp(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    fake_now = main.datetime(2026, 7, 22, 10, 0, tzinfo=main.TAIPEI_TZ)
    monkeypatch.setattr(main, "taipei_now", lambda: fake_now)
    pushes = []

    async def fake_push(chat_id, message):
        pushes.append((chat_id, message))

    monkeypatch.setattr(main, "push_line_message", fake_push)
    insert_home_task(chat_id="family-1", title="換地漏", item_key="地漏", scheduled_date="2026-07-22")

    asyncio.run(main.send_home_task_reminders())
    asyncio.run(main.send_home_task_reminders())

    assert len(pushes) == 1
    assert pushes[0][0] == "family-1"
    assert "家庭事項提醒" in pushes[0][1]
    with main.get_db() as conn:
        row = conn.execute("SELECT last_reminded_at FROM home_tasks WHERE chat_id = ?", ("family-1",)).fetchone()
    assert str(row["last_reminded_at"]).startswith("2026-07-22")


def test_chat_prefix_applies_only_to_chat_action(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "build_chat_reply", lambda text: "你好，請問需要什麼幫助？")

    chat_reply = main.execute_action_route(
        main.ActionRoute(action="chat", should_mutate_db=False, confidence=0.9, reason="test"),
        "你好",
        "U1",
        "chat-m1",
    )

    assert chat_reply == "[chat] 你好，請問需要什麼幫助？"

    monkeypatch.setattr(main, "parse_expense_text", fake_expense_parser)
    route = main.route_action("午餐 120", "U1")
    expense_reply = main.execute_action_route(route, "午餐 120", "U1", "expense-m1")

    assert route.action == "create_expense"
    assert "[chat]" not in expense_reply

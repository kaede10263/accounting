import os
import asyncio
import sys
import types

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
    main.in_flight_tasks.clear()
    main.pending_setting_changes_by_user.clear()
    main.pending_chart_cleanup_by_user.clear()
    main.init_db()


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

    assert "\u522a\u9664 1" in reply
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
    assert "\u522a\u9664 1" in second
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


def test_income_balance_phrases_still_route_to_query_balance(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "route_action_with_openai",
        lambda text: main.ActionRoute(action="query_expenses", should_mutate_db=False, confidence=0.9),
    )

    assert main.route_action("\u9019\u500b\u6708\u6536\u652f\u662f\u591a\u5c11", "U1").action == "query_balance"
    assert main.route_action("\u9019\u500b\u6708\u662f\u5426\u900f\u652f", "U1").action == "query_balance"
    assert main.route_action("\u9019\u500b\u6708\u9084\u5269\u591a\u5c11\u9322", "U1").action == "query_balance"


def test_small_lunch_expense_saves_directly(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "parse_expense_text", fake_expense_parser)
    route = main.ActionRoute(action="create_expense", should_mutate_db=True, confidence=0.95)

    reply = main.execute_action_route(route, "午餐 120", "U1", "m1")

    assert count_expenses() == 1
    assert expense_amounts() == [120]
    assert "確認記帳" not in reply


def test_duplicate_data_list_and_confirm_delete(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    expense = main.ExpenseEntry(
        date="2026-07-07",
        time=None,
        amount=120,
        currency="TWD",
        category=main.infer_category("午餐"),
        merchant=None,
        note="午餐",
        confidence=0.95,
    )
    with main.get_db() as conn:
        for message_id in ("m1", "m2"):
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
                    message_id,
                    "午餐 120",
                    expense.date,
                    expense.time,
                    expense.amount,
                    expense.currency,
                    expense.category,
                    expense.merchant,
                    expense.note,
                    expense.confidence,
                    "now",
                ),
            )
        conn.commit()

    monkeypatch.setattr(
        main,
        "route_action_with_openai",
        lambda text: main.ActionRoute(action="list_expenses", should_mutate_db=False, confidence=0.9),
    )

    route = main.route_action("列出來重複的資料", "U1")
    assert route.action == "list_duplicates"
    reply = main.execute_action_route(route, "列出來重複的資料", "U1", "m3")
    assert "重複" in reply
    assert count_expenses() == 2

    route = main.route_action("刪除重複的資料", "U1")
    assert route.action == "delete_duplicates"
    reply = main.execute_action_route(route, "刪除重複的資料", "U1", "m4")
    assert "確認刪除重複資料" in reply
    assert count_expenses() == 2

    confirm_route = main.ActionRoute(action="chat", should_mutate_db=False, confidence=1.0)
    main.execute_action_route(confirm_route, "確認刪除重複資料", "U1", "m5")
    assert count_expenses() == 1


def test_confirm_delete_expense_takes_priority_over_duplicate_action(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    expense = main.ExpenseEntry(
        date="2026-07-07",
        time=None,
        amount=1_000_000,
        currency="TWD",
        category=main.infer_category("買車"),
        merchant=None,
        note="買車",
        confidence=0.95,
    )
    expense_id = main.save_expense(expense, "買車 100萬", "U1", "m1")
    main.pending_delete_by_user["U1"] = expense_id

    wrong_route = main.ActionRoute(action="delete_duplicates", should_mutate_db=False, confidence=0.9)
    reply = main.execute_action_route(wrong_route, "確認刪除", "U1", "m2")

    assert count_expenses() == 0
    assert "U1" not in main.pending_delete_by_user
    assert "重複資料" not in reply


def test_purchase_cash_for_toys_is_not_investment_reply(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "route_action_with_openai",
        lambda text: main.ActionRoute(
            action="ask_available_investment_cash",
            should_mutate_db=False,
            confidence=0.9,
        ),
    )
    route = main.route_action("有多少錢可以買玩具?", "U1")
    assert route.action == "query_available_cash"
    reply = main.execute_action_route(route, "有多少錢可以買玩具?", "U1", "m1")
    assert "股票" not in reply


def test_stock_cash_routes_to_neutral_available_cash(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "route_action_with_openai",
        lambda text: main.ActionRoute(action="ask_available_investment_cash", should_mutate_db=False, confidence=0.9),
    )
    route = main.route_action("還有多少錢可以買股票?", "U1")
    assert route.action == "query_available_cash"
    reply = main.execute_action_route(route, "還有多少錢可以買股票?", "U1", "m1")
    assert "建議不要全部投入" not in reply


def test_investment_amount_without_advice_routes_to_available_cash(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "route_action_with_openai",
        lambda text: main.ActionRoute(action="ask_available_investment_cash", should_mutate_db=False, confidence=0.9),
    )
    route = main.route_action("可以投資多少?", "U1")
    assert route.action == "query_available_cash"


def test_explicit_stock_advice_routes_to_investment(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "route_action_with_openai",
        lambda text: main.ActionRoute(action="query_available_cash", should_mutate_db=False, confidence=0.9),
    )
    route = main.route_action("建議投入多少股票?", "U1")
    assert route.action == "ask_available_investment_cash"
    reply = main.execute_action_route(route, "建議投入多少股票?", "U1", "m1")
    assert "建議" in reply


def test_appliance_purchase_routes_to_available_cash(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "route_action_with_openai",
        lambda text: main.ActionRoute(action="ask_available_investment_cash", should_mutate_db=False, confidence=0.9),
    )
    route = main.route_action("這個月可以買家電嗎?", "U1")
    assert route.action == "query_available_cash"


def test_today_transport_summary_filters_transport_only(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "parse_expense_query",
        lambda text: main.ExpenseQuery(
            date_range_type="today",
            date_ranges=[main.DateRange(start_date="2026-07-07", end_date="2026-07-07")],
            category="\u4ea4\u901a",
            aggregation="sum",
            confidence=0.95,
        ),
    )

    transport = main.ExpenseEntry(
        date="2026-07-07",
        time=None,
        amount=100,
        currency="TWD",
        category=main.infer_category("\u52a0\u6cb9"),
        merchant=None,
        note="\u52a0\u6cb9\u652f\u51fa",
        confidence=0.95,
    )
    lunch = main.ExpenseEntry(
        date="2026-07-07",
        time=None,
        amount=60,
        currency="TWD",
        category=main.infer_category("\u5348\u9910"),
        merchant=None,
        note="\u5348\u9910",
        confidence=0.95,
    )
    main.save_expense(transport, "\u52a0\u6cb9 100", "U1", "m1")
    main.save_expense(lunch, "\u5348\u9910 60", "U1", "m2")

    raw_text = "\u4eca\u5929\u4ea4\u901a\u82b1\u4e86\u591a\u5c11\u9322?"
    category = main.get_query_category(raw_text)
    reply = main.build_summary_reply(raw_text, "U1")

    assert category == "\u4ea4\u901a"
    assert "TWD 100" in reply
    assert "TWD 60" not in reply


def test_agent_plan_query_question_never_mutates_db():
    plan = main.AgentPlan(
        operation="query",
        target="expenses",
        should_mutate_db=True,
        confidence=0.95,
        date_range_type="today",
        start_date=None,
        end_date=None,
        category="\u4ea4\u901a",
        merchant=None,
        keywords=[],
        amount=None,
        currency="TWD",
        note=None,
        aggregation="sum",
        reason="\u67e5\u8a62\u4eca\u5929\u4ea4\u901a\u652f\u51fa",
    )

    route = main.agent_plan_to_action_route(plan, "\u4eca\u5929\u4ea4\u901a\u82b1\u4e86\u591a\u5c11\u9322?")

    assert route.action == "query_expenses"
    assert route.should_mutate_db is False
    assert route.category == "\u4ea4\u901a"


def test_summary_uses_route_category_from_agent_plan(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "parse_expense_query",
        lambda text: main.ExpenseQuery(
            date_range_type="today",
            date_ranges=[main.DateRange(start_date="2026-07-07", end_date="2026-07-07")],
            category=None,
            aggregation="sum",
            confidence=0.95,
        ),
    )
    main.save_expense(
        main.ExpenseEntry(
            date="2026-07-07",
            time=None,
            amount=100,
            currency="TWD",
            category=main.infer_category("\u52a0\u6cb9"),
            merchant=None,
            note="\u52a0\u6cb9",
            confidence=0.95,
        ),
        "\u52a0\u6cb9 100",
        "U1",
        "m1",
    )
    main.save_expense(
        main.ExpenseEntry(
            date="2026-07-07",
            time=None,
            amount=60,
            currency="TWD",
            category=main.infer_category("\u5348\u9910"),
            merchant=None,
            note="\u5348\u9910",
            confidence=0.95,
        ),
        "\u5348\u9910 60",
        "U1",
        "m2",
    )
    route = main.ActionRoute(
        action="query_expenses",
        should_mutate_db=False,
        confidence=0.95,
        category="\u4ea4\u901a",
    )

    reply = main.build_summary_reply("\u4eca\u5929\u82b1\u4e86\u591a\u5c11\u9322?", "U1", route)

    assert "TWD 100" in reply
    assert "TWD 60" not in reply


def test_phone_bill_expense_fallback_category():
    expense = main.parse_expense_text_fallback("\u96fb\u8a71\u8cbb$100")

    assert expense.amount == 100
    assert expense.category == "\u96fb\u8a71\u8cbb"


def test_specific_months_phone_bill_query_uses_disjoint_ranges(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    for message_id, expense_date, amount in (
        ("m1", "2026-03-05", 100),
        ("m2", "2026-04-05", 999),
        ("m3", "2026-05-05", 200),
    ):
        main.save_expense(
            main.ExpenseEntry(
                date=expense_date,
                time=None,
                amount=amount,
                currency="TWD",
                category="\u96fb\u8a71\u8cbb",
                merchant=None,
                note="\u96fb\u8a71\u8cbb",
                confidence=0.95,
            ),
            f"\u96fb\u8a71\u8cbb {amount}",
            "U1",
            message_id,
        )
    monkeypatch.setattr(
        main,
        "parse_expense_query",
        lambda text: main.ExpenseQuery(
            date_range_type="specific_months",
            date_ranges=[
                main.DateRange(start_date="2026-03-01", end_date="2026-03-31"),
                main.DateRange(start_date="2026-05-01", end_date="2026-05-31"),
            ],
            category="\u96fb\u8a71\u8cbb",
            aggregation="sum",
            confidence=0.95,
        ),
    )

    reply = main.build_summary_reply("3\u6708\u548c5\u6708\u96fb\u8a71\u8cbb\u82b1\u591a\u5c11?", "U1")

    assert "TWD 300" in reply
    assert "TWD 1299" not in reply


def test_structured_query_finds_legacy_living_phone_bill(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    main.save_expense(
        main.ExpenseEntry(
            date="2026-07-07",
            time=None,
            amount=100,
            currency="TWD",
            category="\u751f\u6d3b\u7528\u54c1",
            merchant=None,
            note="\u96fb\u8a71\u8cbb",
            confidence=0.95,
        ),
        "\u96fb\u8a71\u8cbb$100",
        "U1",
        "m1",
    )
    main.save_expense(
        main.ExpenseEntry(
            date="2026-07-07",
            time=None,
            amount=50,
            currency="TWD",
            category="\u9910\u98f2",
            merchant=None,
            note="\u5348\u9910",
            confidence=0.95,
        ),
        "\u5348\u9910 50",
        "U1",
        "m2",
    )
    monkeypatch.setattr(
        main,
        "parse_expense_query",
        lambda text: main.ExpenseQuery(
            date_range_type="today",
            date_ranges=[main.DateRange(start_date="2026-07-07", end_date="2026-07-07")],
            category="\u96fb\u8a71\u8cbb",
            aggregation="sum",
            confidence=0.95,
        ),
    )

    reply = main.build_summary_reply("\u4eca\u5929\u96fb\u8a71\u8cbb\u591a\u5c11?", "U1")

    assert "TWD 100" in reply
    assert "TWD 50" not in reply


def test_chinese_specific_months_water_bill_query(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    for message_id, expense_date, amount in (
        ("m1", "2026-03-05", 100),
        ("m2", "2026-04-05", 999),
        ("m3", "2026-05-05", 200),
    ):
        main.save_expense(
            main.ExpenseEntry(
                date=expense_date,
                time=None,
                amount=amount,
                currency="TWD",
                category="\u6c34\u8cbb",
                merchant=None,
                note="\u6c34\u8cbb",
                confidence=0.95,
            ),
            f"\u6c34\u8cbb {amount}",
            "U1",
            message_id,
        )
    monkeypatch.setattr(main, "parse_expense_query_with_openai", lambda text: main.ExpenseQuery(
        date_range_type="today",
        date_ranges=[main.DateRange(start_date="2026-07-07", end_date="2026-07-07")],
        category="\u5176\u4ed6",
        aggregation="sum",
        confidence=0.95,
    ))

    query = main.parse_expense_query("\u4e09\u6708\u8ddf\u4e94\u6708\u7684\u6c34\u8cbb\u7e3d\u5171\u591a\u5c11\u9322\uff1f")
    reply = main.build_summary_reply("\u4e09\u6708\u8ddf\u4e94\u6708\u7684\u6c34\u8cbb\u7e3d\u5171\u591a\u5c11\u9322\uff1f", "U1")

    assert query.date_range_type == "specific_months"
    assert [(item.start_date, item.end_date) for item in query.date_ranges] == [
        ("2026-03-01", "2026-03-31"),
        ("2026-05-01", "2026-05-31"),
    ]
    assert query.category == "\u6c34\u8cbb"
    assert "TWD 300" in reply
    assert "TWD 1299" not in reply


def seed_july_expense_ratio_data():
    rows = (
        ("m1", "2026-07-07", 60, "\u9910\u98f2", "\u5348\u9910"),
        ("m2", "2026-07-07", 70, "\u9910\u98f2", "\u665a\u9910"),
        ("m3", "2026-07-07", 31, "\u9910\u98f2", "\u98f2\u6599"),
        ("m4", "2026-07-07", 100, "\u4ea4\u901a", "\u52a0\u6cb9"),
        ("m5", "2026-07-07", 200, "\u8cfc\u7269", "\u8cb7\u8863\u670d"),
        ("m6", "2026-07-15", 60000, "\u8cb8\u6b3e", "\u623f\u8cb8"),
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


def test_july_excluding_food_sum_and_ratio(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_july_expense_ratio_data()
    monkeypatch.setattr(
        main,
        "parse_expense_query_with_openai",
        lambda text: main.ExpenseQuery(
            date_range_type="today",
            date_ranges=[main.DateRange(start_date="2026-07-07", end_date="2026-07-07")],
            category="\u5176\u4ed6",
            aggregation="sum",
            confidence=0.95,
        ),
    )

    raw_text = "\u4e03\u6708\u7684\u82b1\u8cbb\uff0c\u9664\u4e86\u98f2\u98df\u5916\uff0c\u7e3d\u5171\u82b1\u591a\u5c11\u9322\uff1f\u4f54\u6bd4\u662f\u591a\u5c11\uff1f"
    query = main.parse_expense_query(raw_text)
    summary = main.calculate_expense_summary_with_ratio(query, "U1")
    reply = main.build_summary_reply(raw_text, "U1")

    assert query.exclude_categories == ["\u9910\u98f2"]
    assert [(item.start_date, item.end_date) for item in query.date_ranges] == [("2026-07-01", "2026-07-31")]
    assert query.aggregation == "sum_and_ratio"
    assert query.ratio_denominator == "all_expenses"
    assert summary["filtered_total"] == 60300
    assert summary["denominator_total"] == 60461
    assert round(summary["ratio_percent"], 2) == 99.73
    assert "\u6392\u9664\u5206\u985e\uff1a\u9910\u98f2" in reply
    assert "TWD 60300" in reply
    assert "99.73%" in reply
    assert "\u5348\u9910" not in reply
    assert "\u665a\u9910" not in reply
    assert "\u98f2\u6599" not in reply


def test_july_excluding_transport_query(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "parse_expense_query_with_openai",
        lambda text: main.ExpenseQuery(
            date_range_type="today",
            date_ranges=[main.DateRange(start_date="2026-07-07", end_date="2026-07-07")],
            aggregation="sum",
            confidence=0.95,
        ),
    )

    query = main.parse_expense_query("\u4e03\u6708\u9664\u4e86\u4ea4\u901a\u4ee5\u5916\u82b1\u591a\u5c11\uff1f")

    assert query.exclude_categories == ["\u4ea4\u901a"]
    assert [(item.start_date, item.end_date) for item in query.date_ranges] == [("2026-07-01", "2026-07-31")]


def test_july_not_including_food_query(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "parse_expense_query_with_openai",
        lambda text: main.ExpenseQuery(
            date_range_type="today",
            date_ranges=[main.DateRange(start_date="2026-07-07", end_date="2026-07-07")],
            aggregation="sum",
            confidence=0.95,
        ),
    )

    query = main.parse_expense_query("7\u6708\u4e0d\u542b\u9910\u98f2\u7e3d\u5171\u591a\u5c11\uff1f")

    assert query.exclude_categories == ["\u9910\u98f2"]
    assert [(item.start_date, item.end_date) for item in query.date_ranges] == [("2026-07-01", "2026-07-31")]


def test_july_food_ratio_query(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "parse_expense_query_with_openai",
        lambda text: main.ExpenseQuery(
            date_range_type="today",
            date_ranges=[main.DateRange(start_date="2026-07-07", end_date="2026-07-07")],
            category=None,
            aggregation="sum",
            confidence=0.95,
        ),
    )

    query = main.parse_expense_query("\u4e03\u6708\u9910\u98f2\u4f54\u6bd4\u662f\u591a\u5c11\uff1f")

    assert query.category == "\u9910\u98f2"
    assert query.aggregation == "sum_and_ratio"
    assert query.ratio_denominator == "all_expenses"


def seed_july_category_breakdown_data():
    rows = (
        ("m1", "2026-07-07", 60, "\u9910\u98f2", "\u5348\u9910"),
        ("m2", "2026-07-07", 70, "\u9910\u98f2", "\u665a\u9910"),
        ("m3", "2026-07-07", 31, "\u9910\u98f2", "\u98f2\u6599"),
        ("m4", "2026-07-07", 100, "\u4ea4\u901a", "\u52a0\u6cb9"),
        ("m5", "2026-07-07", 200, "\u8cfc\u7269", "\u8cb7\u8863\u670d"),
        ("m6", "2026-07-15", 60000, "\u8cb8\u6b3e", "\u623f\u8cb8"),
        ("m7", "2026-07-16", 8000, "\u8cb8\u6b3e", "\u8cb8\u6b3e"),
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


def seed_july_category_breakdown_with_loan_notes():
    rows = (
        ("loan1", "2026-07-15", 60000, "\u8cb8\u6b3e", "\u623f\u8cb8"),
        ("loan2", "2026-07-08", 500, "\u8cb8\u6b3e", "\u4fe1\u8cb8"),
        ("loan3", "2026-07-16", 8000, "\u8cb8\u6b3e", "\u8cb8\u6b3e"),
        ("card1", "2026-07-08", 1200, "\u4fe1\u7528\u5361", "\u4fe1\u7528\u5361"),
        ("food1", "2026-07-07", 60, "\u9910\u98f2", "\u5348\u9910"),
        ("food2", "2026-07-07", 70, "\u9910\u98f2", "\u665a\u9910"),
        ("shop1", "2026-07-07", 200, "\u8cfc\u7269", "\u8cb7\u73a9\u5177"),
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


def fake_bad_breakdown_parse(text):
    return main.ExpenseQuery(
        date_range_type="today",
        date_ranges=[main.DateRange(start_date="2026-07-07", end_date="2026-07-07")],
        category=None,
        include_categories=[
            "\u9910\u98f2",
            "\u4ea4\u901a",
            "\u8cfc\u7269",
            "\u751f\u6d3b\u7528\u54c1",
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
        ],
        aggregation="sum_and_ratio",
        ratio_denominator="all_expenses",
        confidence=0.95,
    )


def test_july_category_breakdown_ratio(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_july_category_breakdown_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_breakdown_parse)

    raw_text = "\u4e03\u6708\u5404\u985e\u5225\u7684\u82b1\u8cbb\u4f54\u6bd4"
    query = main.parse_expense_query(raw_text)
    summary = main.calculate_category_breakdown(query, "U1")
    reply = main.build_summary_reply(raw_text, "U1")

    assert query.aggregation == "category_breakdown"
    assert query.category is None
    assert query.include_categories == []
    assert [(item.start_date, item.end_date) for item in query.date_ranges] == [("2026-07-01", "2026-07-31")]
    assert summary["denominator_total"] == 68461
    rows = {row["category"]: row for row in summary["rows"]}
    assert rows["\u8cb8\u6b3e"]["total"] == 68000
    assert round(rows["\u8cb8\u6b3e"]["ratio_percent"], 2) == 99.33
    assert rows["\u8cfc\u7269"]["total"] == 200
    assert rows["\u9910\u98f2"]["total"] == 161
    assert rows["\u4ea4\u901a"]["total"] == 100
    assert "\u5404\u985e\u5225\u4f54\u6bd4" in reply
    assert "\u8cb8\u6b3e\uff1aTWD 68000" in reply
    assert "99.33%" in reply
    assert "\u8cfc\u7269\uff1aTWD 200" in reply
    assert "\u9910\u98f2\uff1aTWD 161" in reply
    assert "\u4ea4\u901a\uff1aTWD 100" in reply
    assert "\u7e3d\u82b1\u8cbb\uff1aTWD" not in reply
    assert "\u5206\u985e\uff1a\u9910\u98f2\u3001\u4ea4\u901a\u3001\u8cfc\u7269" not in reply


def test_july_category_breakdown_excluding_mortgage_keyword(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_july_category_breakdown_with_loan_notes()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_breakdown_parse)

    raw_text = "\u4e03\u6708\u5404\u985e\u5225\u7684\u82b1\u8cbb\u4f54\u6bd4\uff08\u4e0d\u8981\u8a08\u7b97\u623f\u8cb8\uff09"
    query = main.parse_expense_query(raw_text)
    summary = main.calculate_category_breakdown(query, "U1")
    reply = main.build_summary_reply(raw_text, "U1")

    assert query.aggregation == "category_breakdown"
    assert query.exclude_keywords == ["\u623f\u8cb8"]
    assert query.exclude_categories == []
    assert query.ratio_denominator == "filtered_expenses"
    assert summary["denominator_total"] == 10030
    rows = {row["category"]: row for row in summary["rows"]}
    assert rows["\u8cb8\u6b3e"]["total"] == 8500
    assert round(rows["\u8cb8\u6b3e"]["ratio_percent"], 2) == 84.75
    assert "\u6392\u9664\u9805\u76ee\uff1a\u623f\u8cb8" in reply
    assert "TWD 10030" in reply
    assert "TWD 70030" not in reply
    assert "\u623f\u8cb8" not in "\n".join(line for line in reply.splitlines() if "\u8cb8\u6b3e\uff1a" in line)
    assert "\u8cb8\u6b3e\uff1aTWD 8500" in reply


def test_july_category_breakdown_without_exclusion_uses_all_expenses(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_july_category_breakdown_with_loan_notes()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_breakdown_parse)

    query = main.parse_expense_query("\u4e03\u6708\u5404\u985e\u5225\u7684\u82b1\u8cbb\u4f54\u6bd4")
    summary = main.calculate_category_breakdown(query, "U1")

    assert query.ratio_denominator == "all_expenses"
    assert summary["denominator_total"] == 70030
    rows = {row["category"]: row for row in summary["rows"]}
    assert rows["\u8cb8\u6b3e"]["total"] == 68500


def test_july_category_breakdown_excluding_loan_category(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_july_category_breakdown_with_loan_notes()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_breakdown_parse)

    query = main.parse_expense_query("\u4e03\u6708\u5404\u985e\u5225\u7684\u82b1\u8cbb\u4f54\u6bd4\uff08\u4e0d\u8981\u8a08\u7b97\u8cb8\u6b3e\uff09")
    summary = main.calculate_category_breakdown(query, "U1")
    rows = {row["category"]: row for row in summary["rows"]}

    assert query.exclude_categories == ["\u8cb8\u6b3e"]
    assert query.exclude_keywords == ["\u8cb8\u6b3e"]
    assert query.ratio_denominator == "filtered_expenses"
    assert summary["denominator_total"] == 1530
    assert "\u8cb8\u6b3e" not in rows


def test_july_category_breakdown_excluding_credit_loan_keyword(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_july_category_breakdown_with_loan_notes()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_breakdown_parse)

    query = main.parse_expense_query("\u4e03\u6708\u5404\u985e\u5225\u7684\u82b1\u8cbb\u4f54\u6bd4\uff08\u4e0d\u8981\u8a08\u7b97\u4fe1\u8cb8\uff09")
    summary = main.calculate_category_breakdown(query, "U1")
    rows = {row["category"]: row for row in summary["rows"]}

    assert query.exclude_keywords == ["\u4fe1\u8cb8"]
    assert query.exclude_categories == []
    assert summary["denominator_total"] == 69530
    assert rows["\u8cb8\u6b3e"]["total"] == 68000


def seed_actual_spending_with_paid_payable():
    main.save_expense(
        main.ExpenseEntry(
            date="2026-07-06",
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
        "actual-expense-1",
    )
    insert_payable(
        line_user_id="U1",
        item_type="\u623f\u8cb8",
        amount=7,
        due_date="2026-07-15",
        status="paid",
    )


def test_actual_spending_rows_include_expenses_and_paid_payables(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_actual_spending_with_paid_payable()

    rows = main.get_actual_spending_rows("2026-07-01", "2026-07-31", "U1")
    by_source = {str(row["source"]): row for row in rows}

    assert by_source["expense"]["note"] == "\u9eb5\u5305"
    assert by_source["paid_payable"]["note"] == "\u623f\u8cb8"
    assert by_source["paid_payable"]["category"] == "\u8cb8\u6b3e"
    assert by_source["paid_payable"]["date"] == "2026-07-16"


def test_all_spending_ratio_pie_includes_paid_payables(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_actual_spending_with_paid_payable()
    set_fake_today(monkeypatch, 2026, 7, 8)

    def raise_expense_parser(text):
        raise RuntimeError("skip OpenAI in test")

    monkeypatch.setattr(main, "parse_expense_query_with_openai", raise_expense_parser)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.ngrok-free.app")
    monkeypatch.setattr(main, "CHART_DIR", tmp_path / "charts")

    raw_text = "\u9019\u500b\u6708\u7684\u6240\u6709\u82b1\u8cbb\u6bd4\u4f8b + \u5713\u9905\u5716"
    query = main.parse_expense_query(raw_text)
    result = main.execute_expense_query(query, "U1")
    reply = main.build_expense_reply(raw_text, "U1")
    rows = {row["group"]: row for row in result["rows"]}

    assert query.aggregation == "category_breakdown"
    assert query.group_by == ["category"]
    assert query.chart_type == "pie"
    assert result["denominator_total"] == 182
    assert rows["\u9910\u98f2"]["total"] == 175
    assert round(float(rows["\u9910\u98f2"]["ratio_percent"]), 2) == 96.15
    assert rows["\u8cb8\u6b3e"]["total"] == 7
    assert round(float(rows["\u8cb8\u6b3e"]["ratio_percent"]), 2) == 3.85
    assert "\u9910\u98f2\uff1aTWD 175" in reply["text"]
    assert "\u8cb8\u6b3e\uff1aTWD 7" in reply["text"]
    assert reply["image_url"] is not None


def test_actual_spending_aggregate_max_min_include_paid_payables(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_actual_spending_with_paid_payable()

    query = main.ExpenseQuery(
        date_range_type="custom",
        date_ranges=[main.DateRange(start_date="2026-07-01", end_date="2026-07-31")],
        category=None,
        aggregation="sum",
        confidence=0.95,
    )
    result = main.execute_expense_query(query, "U1")

    assert result["count"] == 2
    assert result["total"] == 182
    assert result["max"] == 175
    assert result["max_item_name"] == "\u9eb5\u5305"
    assert result["min"] == 7
    assert result["min_item_name"] == "\u623f\u8cb8"


def test_month_finance_counts_paid_payables_as_actual_spending(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_actual_spending_with_paid_payable()
    insert_payable(
        line_user_id="U1",
        item_type="\u4fe1\u8cb8",
        amount=10,
        due_date="2026-07-20",
        status="unpaid",
    )

    finance = main.get_month_finance("\u9019\u500b\u6708\u9810\u4f30\u7e3d\u652f\u51fa", "U1")

    assert finance["expense_total"] == 182
    assert finance["unpaid_total"] == 10
    assert finance["available_cash"] == -192


def test_july_category_ratio_short_text(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_breakdown_parse)

    query = main.parse_expense_query("7\u6708\u5206\u985e\u4f54\u6bd4")

    assert query.aggregation == "category_breakdown"
    assert query.category is None
    assert query.include_categories == []


def test_july_each_category_spending_text(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_breakdown_parse)

    query = main.parse_expense_query("\u4e03\u6708\u6bcf\u500b\u985e\u5225\u82b1\u591a\u5c11")

    assert query.aggregation == "category_breakdown"
    assert query.category is None
    assert query.include_categories == []


def test_this_month_each_category_ratio_text(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_breakdown_parse)

    query = main.parse_expense_query("\u9019\u500b\u6708\u5404\u5206\u985e\u652f\u51fa\u6bd4\u4f8b")

    assert query.aggregation == "category_breakdown"
    assert query.category is None
    assert query.include_categories == []


def seed_july_list_sort_data():
    rows = (
        ("sort1", "2026-07-07", "12:00", 60, "\u9910\u98f2", "\u5348\u9910"),
        ("sort2", "2026-07-07", "20:00", 120, "\u9910\u98f2", "\u665a\u9910"),
        ("sort3", "2026-07-07", "15:00", 30, "\u9910\u98f2", "\u9ede\u5fc3"),
        ("sort4", "2026-07-08", None, 100, "\u96fb\u8a71\u8cbb", "\u96fb\u8a71\u8cbb"),
        ("sort5", "2026-07-05", None, 500, "\u4ea4\u901a", "\u52a0\u6cb9"),
    )
    for message_id, expense_date, expense_time, amount, category, note in rows:
        main.save_expense(
            main.ExpenseEntry(
                date=expense_date,
                time=expense_time,
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


def fake_bad_list_parse(text):
    return main.ExpenseQuery(
        date_range_type="today",
        date_ranges=[main.DateRange(start_date="2026-07-08", end_date="2026-07-08")],
        aggregation="list",
        confidence=0.95,
    )


def assert_order(text: str, *labels: str):
    positions = [text.index(label) for label in labels]
    assert positions == sorted(positions)


def test_list_expenses_sort_amount_desc(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_july_list_sort_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_list_parse)

    raw_text = "\u5217\u51fa\u4e03\u6708\u82b1\u8cbb\uff0c\u6309\u91d1\u984d\u7531\u5927\u5230\u5c0f"
    query = main.parse_expense_query(raw_text)
    reply = main.build_list_reply(raw_text, "U1")

    assert query.sort_by == "amount"
    assert query.sort_direction == "desc"
    assert "\u6392\u5e8f\uff1a\u91d1\u984d\u7531\u5927\u5230\u5c0f" in reply
    assert_order(reply, "\u52a0\u6cb9", "\u665a\u9910")


def test_list_expenses_sort_amount_asc(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_july_list_sort_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_list_parse)

    raw_text = "\u5217\u51fa\u4e03\u6708\u82b1\u8cbb\uff0c\u91d1\u984d\u5c0f\u5230\u5927"
    reply = main.build_list_reply(raw_text, "U1")

    assert_order(reply, "\u9ede\u5fc3", "\u5348\u9910")


def test_list_expenses_sort_time_asc(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_july_list_sort_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_list_parse)

    raw_text = "\u5217\u51fa7\u67087\u865f\u82b1\u8cbb\uff0c\u6642\u9593\u65e9\u5230\u665a"
    query = main.parse_expense_query(raw_text)
    reply = main.build_list_reply(raw_text, "U1")

    assert query.sort_by == "time"
    assert query.sort_direction == "asc"
    assert [(item.start_date, item.end_date) for item in query.date_ranges] == [("2026-07-07", "2026-07-07")]
    assert_order(reply, "12:00 \u5348\u9910", "15:00 \u9ede\u5fc3", "20:00 \u665a\u9910")


def test_list_expenses_sort_time_desc(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_july_list_sort_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_list_parse)

    raw_text = "\u5217\u51fa7\u67087\u865f\u82b1\u8cbb\uff0c\u6642\u9593\u665a\u5230\u65e9"
    reply = main.build_list_reply(raw_text, "U1")

    assert_order(reply, "20:00 \u665a\u9910", "15:00 \u9ede\u5fc3", "12:00 \u5348\u9910")


def test_list_expenses_sort_date_asc(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_july_list_sort_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_list_parse)

    raw_text = "\u5217\u51fa\u4e03\u6708\u82b1\u8cbb\uff0c\u65e5\u671f\u820a\u5230\u65b0"
    reply = main.build_list_reply(raw_text, "U1")

    assert_order(reply, "2026-07-05 \u52a0\u6cb9", "2026-07-08 \u96fb\u8a71\u8cbb")


def test_list_expenses_default_sort_date_desc(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_july_list_sort_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_list_parse)

    raw_text = "\u5217\u51fa\u4e03\u6708\u82b1\u8cbb"
    query = main.parse_expense_query(raw_text)
    reply = main.build_list_reply(raw_text, "U1")

    assert query.sort_by == "date"
    assert query.sort_direction == "desc"
    assert "\u6392\u5e8f\uff1a\u65e5\u671f\u7531\u65b0\u5230\u820a" in reply
    assert_order(reply, "2026-07-08 \u96fb\u8a71\u8cbb", "2026-07-05 \u52a0\u6cb9")


def seed_expense_statistics_data():
    rows = (
        ("stat1", "2026-07-06", None, 0, "\u9910\u98f2", "\u7121"),
        ("stat2", "2026-07-07", "12:00", 100, "\u9910\u98f2", "\u5348\u9910"),
        ("stat3", "2026-02-10", None, 300, "\u9910\u98f2", "\u9910\u98f2"),
        ("stat4", "2026-03-10", None, 50, "\u9910\u98f2", "\u9910\u98f2"),
        ("stat5", "2026-05-10", None, 300, "\u9910\u98f2", "\u9910\u98f2"),
        ("stat6", "2026-06-10", None, 400, "\u9910\u98f2", "\u9910\u98f2"),
        ("stat7", "2026-07-15", None, 60000, "\u8cb8\u6b3e", "\u623f\u8cb8"),
        ("stat8", "2026-07-08", None, 500, "\u8cb8\u6b3e", "\u4fe1\u8cb8"),
        ("stat9", "2026-07-16", None, 8000, "\u8cb8\u6b3e", "\u8cb8\u6b3e"),
        ("stat10", "2026-07-08", None, 1200, "\u4fe1\u7528\u5361", "\u4fe1\u7528\u5361"),
        ("stat11", "2026-07-07", None, 70, "\u9910\u98f2", "\u665a\u9910"),
        ("stat12", "2026-07-07", None, 200, "\u8cfc\u7269", "\u8cb7\u73a9\u5177"),
    )
    for message_id, expense_date, expense_time, amount, category, note in rows:
        if amount <= 0:
            continue
        main.save_expense(
            main.ExpenseEntry(
                date=expense_date,
                time=expense_time,
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
    seed_expense_statistics_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.setattr(main, "PUBLIC_BASE_URL", None)

    reply = main.build_expense_reply("\u672c\u9031\u9910\u98f2\u6bcf\u5929\u82b1\u591a\u5c11 + \u6298\u7dda\u5716", "U1")

    assert reply["image_url"] is None
    assert "PUBLIC_BASE_URL" in reply["text"]


def seed_weekly_food_240_data():
    rows = (
        ("food-week-1", "2026-07-06", 175, "\u9eb5\u5305"),
        ("food-week-2", "2026-07-07", 65, "\u65e9\u9910"),
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
    seed_weekly_food_240_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)

    reply = main.build_expense_reply("\u672c\u9031\u9910\u98f2\u82b1\u591a\u5c11", "U1")

    assert "\u7b46\u6578\uff1a2" in reply["text"]
    assert "\u7e3d\u82b1\u8cbb\uff1aTWD 240" in reply["text"]
    assert "\u6700\u9ad8\uff1aTWD 175\uff08\u9eb5\u5305\uff09" in reply["text"]
    assert "\u6700\u4f4e\uff1aTWD 65\uff08\u65e9\u9910\uff09" in reply["text"]


def test_weekly_food_daily_uses_same_filter_as_aggregate(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
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
    assert os.path.basename(main.get_db_path()) == "accounting_test.db"


def test_app_env_test_rejects_prod_db(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DB_PATH", "accounting.db")
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    try:
        main.get_db_path()
    except RuntimeError as exc:
        assert "cannot use prod DB" in str(exc)
    else:
        raise AssertionError("APP_ENV=test should reject accounting.db")


def test_app_env_test_push_line_message_is_skipped(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)

    async def fail_post(*args, **kwargs):
        raise AssertionError("LINE Push API should not be called in APP_ENV=test")

    monkeypatch.setattr(main.httpx.AsyncClient, "post", fail_post, raising=False)

    asyncio.run(main.push_line_message("U1", "test"))


def test_grouped_food_query_ignores_wrong_route_category(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    seed_expense_statistics_data()
    monkeypatch.setattr(main, "parse_expense_query_with_openai", fake_bad_stat_parse)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)

    route = main.ActionRoute(
        action="query_expenses",
        should_mutate_db=False,
        confidence=0.95,
        category="\u5176\u4ed6",
    )
    reply = main.build_expense_reply("\u672c\u9031\u9910\u98f2\u6bcf\u5929\u82b1\u591a\u5c11 + \u6298\u7dda\u5716", "U1", route)

    assert "2026-07-07\uff1aTWD 170" in reply["text"]


def test_line_chart_keeps_zero_value_points(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
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

    query = main.ExpenseQuery(
        mode="grouped_aggregate",
        metric="sum",
        group_by=["day"],
        date_range_type="custom",
        date_ranges=[main.DateRange(start_date="2026-07-06", end_date="2026-07-08")],
        wants_chart=True,
        chart_type="line",
        aggregation="sum",
        confidence=0.95,
    )
    result = {
        "mode": "grouped_aggregate",
        "group_by": "day",
        "rows": [
            {"group": "2026-07-06", "total": 0, "count": 0},
            {"group": "2026-07-07", "total": 100, "count": 1},
            {"group": "2026-07-08", "total": 0, "count": 0},
        ],
    }

    url = main.generate_expense_chart(query, result)

    assert url is not None
    assert captured["labels"] == ["2026-07-06", "2026-07-07", "2026-07-08"]
    assert captured["values"] == [0, 100, 0]


def install_fake_matplotlib(monkeypatch):
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.use = lambda backend: None
    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_pyplot.figure = lambda **kwargs: None
    fake_pyplot.plot = lambda *args, **kwargs: None
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


def make_chart_query():
    return main.ExpenseQuery(
        mode="grouped_aggregate",
        metric="sum",
        group_by=["day"],
        date_range_type="custom",
        date_ranges=[main.DateRange(start_date="2026-07-06", end_date="2026-07-07")],
        wants_chart=True,
        chart_type="line",
        aggregation="sum",
        confidence=0.95,
    )


def make_chart_result(value=100):
    return {
        "mode": "grouped_aggregate",
        "group_by": "day",
        "rows": [
            {"group": "2026-07-06", "total": 0, "count": 0},
            {"group": "2026-07-07", "total": value, "count": 1},
        ],
    }


def test_chart_cache_reuses_same_payload(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.ngrok-free.app")
    monkeypatch.setattr(main, "CHART_DIR", tmp_path / "charts")
    install_fake_matplotlib(monkeypatch)

    query = make_chart_query()
    result = make_chart_result()
    url1 = main.generate_expense_chart(query, result, "U1")
    url2 = main.generate_expense_chart(query, result, "U1")

    assert url1 == url2
    assert len(list((tmp_path / "charts").glob("*.png"))) == 1


def test_chart_cache_different_payload_creates_different_png(monkeypatch, tmp_path):
    setup_tmp_db(monkeypatch, tmp_path)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.ngrok-free.app")
    monkeypatch.setattr(main, "CHART_DIR", tmp_path / "charts")
    install_fake_matplotlib(monkeypatch)

    query = make_chart_query()
    url1 = main.generate_expense_chart(query, make_chart_result(100), "U1")
    url2 = main.generate_expense_chart(query, make_chart_result(200), "U1")

    assert url1 != url2
    assert len(list((tmp_path / "charts").glob("*.png"))) == 2


def test_cleanup_old_charts_deletes_expired_png(monkeypatch, tmp_path):
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

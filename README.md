# LINE Accounting Bot

FastAPI 版 LINE 記帳 Bot 第一版。支援 LINE webhook、簽章驗證、OpenAI Structured Outputs 解析記帳文字，並把結果寫入 SQLite。

## 功能

- `POST /line/webhook`
- 驗證 `X-Line-Signature`
- 使用 LINE Reply API 回覆記帳結果
- 支援文字格式，例如：
  - `午餐 120`
  - `全聯 328`
  - `保母費 18000`
- 使用 OpenAI Structured Outputs 解析：
  - `date`
  - `amount`
  - `currency`
  - `category`
  - `merchant`
  - `note`
  - `confidence`
- 寫入 `accounting.db`

## 安裝

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 環境變數

複製 `.env.example` 成 `.env`，並填入你的 token：

```bash
copy .env.example .env
```

`.env` 需要包含：

```env
LINE_CHANNEL_SECRET=your_line_channel_secret
LINE_CHANNEL_ACCESS_TOKEN=your_line_channel_access_token
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
DATABASE_PATH=accounting.db
```

## 啟動

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

健康檢查：

```bash
curl http://localhost:8000/health
```

## LINE Webhook 設定

本機開發時可用 ngrok 或 Cloudflare Tunnel 暴露服務：

```bash
ngrok http 8000
```

到 LINE Developers 後台設定 Webhook URL：

```text
https://你的公開網址/line/webhook
```

並確認：

- Channel secret 對應 `.env` 的 `LINE_CHANNEL_SECRET`
- Channel access token 對應 `.env` 的 `LINE_CHANNEL_ACCESS_TOKEN`
- 啟用 Webhook

## 資料庫

啟動時會自動建立 SQLite 資料庫與 `expenses` table。預設檔案是：

```text
accounting.db
```

欄位包含：

- `line_user_id`
- `message_id`
- `raw_text`
- `date`
- `amount`
- `currency`
- `category`
- `merchant`
- `note`
- `confidence`
- `created_at`

## 測試提醒

`/line/webhook` 需要有效 LINE 簽章，直接用 curl 打 webhook 會因簽章錯誤回傳 `400`。可先用 `/health` 確認服務啟動，再透過 LINE Developers 的 Verify 或實際 LINE 訊息測試。

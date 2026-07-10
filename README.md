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
ngrok http --url=sixth-impulse-spiffy.ngrok-free.dev 8000
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

## 指令集

### 記帳與查詢

- 新增支出：`午餐 120`、`全聯 328`、`加油 100`
- 查支出總額：`今天交通花了多少錢?`
- 查支出明細：`列出七月花費`
- 排序明細：`列出七月花費，按金額由大到小`
- 每日統計：`本週餐飲每天花多少`
- 圖表：`本週餐飲每天花多少 + 折線圖`
- 各類別佔比：`七月各類別的花費佔比`
- 排除分類或關鍵字：`七月各類別的花費佔比（不要計算房貸）`

### 收入

- 新增收入：`薪水 50000`、`7月薪水 老公 50000`
- 查收入總額：`薪資收入是多少`、`這個月收入是多少`
- 查收入清單：`薪資收入清單`、`收入清單`

### 待繳款與提醒

- 建立待繳提醒：`提醒我 7/8 要繳房貸 62218`
- 查未繳：`這個月還有哪些沒繳?`
- 查特定項目：`這個月房貸繳了嗎?`
- 標記已繳：`房貸已繳`、`已繳 信貸`

### 收支與可動用金額

- 查收支：`這個月收支是多少`
- 查透支：`這個月是否透支`
- 查剩餘：`這個月還剩多少錢`
- 查購買可動用金額：`有多少錢可以買玩具?`

### 系統設定與用量

- 讀設定：`readSetting` 或 `讀設定`
- 修改設定說明：`writeSetting` 或 `改設定`
- 設定參數：`設定 OPENAI_MODEL gpt-4.1-mini`
- 確認設定：`確認設定 OPENAI_MODEL gpt-4.1-mini`
- 查用量：`usage`、`usage 7d`、`usage month`

### 圖表快取

- 清理過期圖表：`cleanCharts`
- 清空全部圖表：`cleanCharts all`
- 確認清空：`確認清空圖表`

### v2-agent 訓練資料

- 修正上一句分類：`上一句判錯，應該是 居家提醒`
- 修正為待辦：`上一句不是記帳，是居家待辦`
- 修正為收入查詢：`上一句應該是 收入查詢`
- 修正為房屋修繕：`上一句應該是 房屋修繕`
- 匯出訓練資料：`exportTrainingData`

匯出的檔案為 `training_intents.jsonl`，預設已加入 `.gitignore`，不會被提交到 GitHub。

## 測試提醒

`/line/webhook` 需要有效 LINE 簽章，直接用 curl 打 webhook 會因簽章錯誤回傳 `400`。可先用 `/health` 確認服務啟動，再透過 LINE Developers 的 Verify 或實際 LINE 訊息測試。

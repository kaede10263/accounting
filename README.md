# LINE Accounting Bot

以 FastAPI 實作的 LINE 家計簿 Bot，負責接收 LINE webhook、解析中文記帳訊息、寫入 SQLite，並提供查詢、刪除、待繳管理、設定查詢與使用量檢查等功能。

## 功能重點

- 支援 LINE webhook 與簽章驗證
- 使用 OpenAI Structured Outputs 解析記帳內容
- 資料儲存在 SQLite
- 支援支出、收入、待繳項目、圖表與統計查詢
- 支援刪除記帳，單筆刪除確認碼為 `88`
- 支援 `checkSql` 檢查最近 SQL 新增或刪除紀錄
- 針對 LINE reply token 失效時自動 fallback 為 push message
- 具備 webhook 去重與部分資料去重邏輯

## 家計簿查詢規則

- 新增資料時仍會保留 `line_user_id`
- 查詢類功能會以家計簿共用視角彙總資料，不只看目前發訊息的人
- 例如「這星期在交通的花費總共是多少」會把家人新增的同類支出一起算入

## 主要指令

輸入 `commands`、`help` 或 `所有指令` 可查看指令列表。

目前內建指令：

1. `commands`
2. `checkSql`
3. `readSetting`
4. `writeSetting`
5. `usage`
6. `exportTrainingData`
7. `cleanCharts`

## 常見訊息範例

### 記帳

- `今天加油 100`
- `早餐 50`
- `7/10 高鐵 984`

### 查詢

- `這星期在交通的花費總共是多少`
- `這兩天的花費清單`
- `這個月花費最高的是什麼`
- `這個月還有多少錢可以用`

### 刪除

- `移除早餐`
- 系統列出候選資料後，回覆 `88` 進行單筆刪除確認

### SQL 檢查

- `checkSql`
- `checkSql10`

輸出格式例如：

```text
最近5筆 SQL 新增/刪除
2026/07/10 09:20:35 記帳 新增 #8 早餐50
2026/07/10 09:24:10 記帳 刪除 #8 早餐50
```

### 設定與維護

- `readSetting`
- `writeSetting`
- `設定 OPENAI_MODEL gpt-4.1-mini`
- `usage`
- `usage 7d`
- `cleanCharts`
- `cleanCharts all`

## 開發環境

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 環境變數

先複製 `.env.example`：

```bash
copy .env.example .env
```

`.env` 主要欄位：

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

## LINE Webhook

本機開發可使用 ngrok 或 Cloudflare Tunnel 暴露服務：

```bash
ngrok http 8000
```

將 LINE Developers 的 Webhook URL 設為：

```text
https://your-domain/line/webhook
```

## 資料庫

預設資料庫檔案：

```text
accounting.db
```

目前會用到的主要資料表包含：

- `expenses`
- `incomes`
- `payables`
- `settings`
- `usage_logs`
- `sql_change_logs`

## 測試

```bash
python -m pytest tests/test_expense_confirmation.py -q
```

## 備註

- `training_intents.jsonl` 已列入 `.gitignore`
- `*.db` 已列入 `.gitignore`
- 若 `reply_token` 已失效，系統會改用 push 回訊，避免使用者完全收不到結果

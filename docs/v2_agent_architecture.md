# v2-agent Architecture

## 1. Goal

v2-agent is a family life management AI Agent, not only an accounting bot.

The current v1-stable LINE bot remains in service and should only receive bug fixes, utterance logging, user feedback capture, and training data export. v2-agent will be redesigned as a multi-domain agent that can manage family finance, payables, home tasks, maintenance records, recurring reminders, settings, usage, and general chat.

## 2. Domains

- `finance`
- `payable`
- `home`
- `reminder`
- `settings`
- `usage`
- `chat`

## 3. AgentDecision Schema

```python
from typing import Literal
from pydantic import BaseModel


class ToolCall(BaseModel):
    tool: str
    args: dict
    confidence: float


class AgentDecision(BaseModel):
    domain: Literal[
        "finance",
        "payable",
        "home",
        "reminder",
        "settings",
        "usage",
        "chat",
    ]
    intent: str
    should_mutate_db: bool
    needs_confirmation: bool
    confirmation_question: str | None
    tool_calls: list[ToolCall]
    confidence: float
    reason: str | None
```

## 4. Tool Registry

```python
TOOL_REGISTRY = {
    "finance.create_expense": ...,
    "finance.query_expenses": ...,
    "finance.query_incomes": ...,
    "finance.query_balance": ...,
    "payable.create": ...,
    "payable.query": ...,
    "payable.mark_paid": ...,
    "home.create_task": ...,
    "home.query_tasks": ...,
    "home.complete_task": ...,
    "home.create_maintenance_record": ...,
    "reminder.create": ...,
    "settings.read": ...,
    "usage.read": ...,
}
```

The planner only returns an `AgentDecision`. Python executes tools through the registry, applies safety checks, handles confirmation, and writes durable records.

## 5. Home Domain Schema

### tasks

- `id`
- `scope_type`
- `scope_id`
- `domain`
- `task_type`
- `title`
- `description`
- `item`
- `location`
- `due_date`
- `recurrence`
- `priority`
- `status`
- `assigned_to`
- `created_from_message_id`
- `created_at`
- `completed_at`

### maintenance_records

- `id`
- `scope_type`
- `scope_id`
- `item`
- `location`
- `vendor`
- `amount`
- `currency`
- `maintenance_date`
- `note`
- `linked_expense_id`
- `created_from_message_id`
- `created_at`

## 6. Hugging Face Local Model Strategy

The first local-model milestone is a domain/intent classifier only.

- Do not replace the OpenAI planner at the beginning.
- Train a local classifier from accumulated `utterance_logs`.
- If local classifier confidence is `>= 0.85`, use the local classifier result.
- If confidence is `< 0.85`, fall back to the OpenAI planner.
- OpenAI decisions and user corrections continue to be written back to `utterance_logs`.
- The local classifier predicts labels such as `finance.create_expense`, `home.create_task`, or `payable.mark_paid`.
- Tool argument extraction can remain OpenAI-driven until the classifier has enough data and evaluation coverage.

## 7. Training Data Format

JSONL:

```jsonl
{"text":"午餐 120","label":"finance.create_expense"}
{"text":"薪資收入清單","label":"finance.list_incomes"}
{"text":"提醒我下週六清冷氣濾網","label":"home.create_task"}
{"text":"今天請水電修馬桶 1800","label":"home.record_repair"}
```

The label source is:

1. Use `final_domain.final_intent` when user feedback exists.
2. Otherwise use `predicted_domain.predicted_intent`.

## 8. Migration Plan

### Phase 0

Add `utterance_logs` and `exportTrainingData` to v1.

### Phase 1

Create the v2 app skeleton.

### Phase 2

Implement `AgentDecision` and `TOOL_REGISTRY`.

### Phase 3

Move finance and payable capabilities into tools.

### Phase 4

Add home tasks and maintenance records.

### Phase 5

Connect a Hugging Face local classifier.

### Phase 6

Use accumulated data for fine-tuning or LoRA.

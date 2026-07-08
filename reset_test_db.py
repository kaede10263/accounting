import os
from pathlib import Path

import main


def reset_test_db() -> None:
    os.environ["APP_ENV"] = "test"
    os.environ["DB_PATH"] = os.getenv("DB_PATH", "accounting_test.db")
    db_path = Path(main.get_db_path())
    if db_path.name == "accounting.db":
        raise RuntimeError("Refusing to reset prod DB in APP_ENV=test.")
    if db_path.exists():
        db_path.unlink()
    main.init_db()
    print(f"Reset test DB: {db_path}")


if __name__ == "__main__":
    reset_test_db()

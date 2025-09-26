import sqlite3
import os


class SQLiteManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        dir_path = os.path.dirname(db_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.connection.cursor()
        self._create_tables()

    def _create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type INTEGER NOT NULL,
                source TEXT NOT NULL,
                task_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                cid TEXT NOT NULL,
                request_data TEXT NOT NULL,
                response_data TEXT NOT NULL,
                status_code INTEGER DEFAULT 200,
                tool_hit TEXT DEFAULT '[]',
                cost REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.connection.commit()

    def insert_request(
        self,
        type: int,
        source: str,
        task_id: str,
        project_id: str,
        cid: str,
        request_data: str,
        response_data: str,
        status_code: int = 200,
        tool_hit: str = '{}',
        cost: float = 0.0
    ):
        self.cursor.execute('''
            INSERT INTO requests (type, source, task_id, project_id, cid, request_data, response_data, status_code, tool_hit, cost) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (type, source, task_id, project_id, cid, request_data, response_data, status_code, tool_hit, cost))
        self.connection.commit()

    def fetch_all(self) -> list[tuple[any, ...]]:
        try:
            cur = self.cursor.execute('''
                SELECT * FROM requests ORDER BY id DESC LIMIT 25
            ''')
            return cur.fetchall()
        except Exception as e:
            print(f"Error fetching all requests: {e}")
            return []

    def fetch_newer_than(self, since_id: int) -> list[tuple[any, ...]]:
        """
        Fetch records with ID greater than since_id.
        Used for incremental updates to the UI.
        """
        try:
            cur = self.cursor.execute('''
                SELECT * FROM requests WHERE id > ? ORDER BY id DESC LIMIT 50
            ''', (since_id,))
            return cur.fetchall()
        except Exception as e:
            print(f"Error fetching requests newer than {since_id}: {e}")
            return []
    
    def cleanup_old_records(self, days: int = 3):
        """
        Delete records older than the specified number of days.
        """
        try:
            self.cursor.execute('''
                DELETE FROM requests WHERE created_at < datetime('now', ? || ' days')
            ''', (-days,))
            self.connection.commit()
        except Exception as e:
            print(f"Error cleaning up old records: {e}")

    def close(self):
        self.connection.close()

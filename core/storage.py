import sqlite3
import numpy as np
import os
import tempfile
from datetime import datetime

DB_NAME = os.path.join(tempfile.gettempdir(), "database.db")


def init_db():
    """Initialize the database with the users table."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()
    print(f" Database {DB_NAME} initialized.")


def add_user(name, embedding):
    """Add a new user and return their assigned ID."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Convert numpy array to bytes
    emb_bytes = embedding.astype(np.float32).tobytes()
    # Use local system time (not SQLite CURRENT_TIMESTAMP which is always UTC)
    local_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO users (name, embedding, created_at) VALUES (?, ?, ?)",
        (name, emb_bytes, local_ts),
    )
    user_id = c.lastrowid
    conn.commit()
    conn.close()
    return user_id


def get_all_embeddings():
    """Retrieve all embeddings and IDs."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, name, embedding FROM users")
    rows = c.fetchall()
    conn.close()

    ids = []
    names = {}
    embeddings = []

    for r in rows:
        uid, name, emb_blob = r
        # Convert bytes back to numpy array
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        ids.append(uid)
        names[uid] = name
        embeddings.append(emb)

    if not ids:
        return [], {}, np.array([])

    return ids, names, np.vstack(embeddings)


def get_all_users():
    """Retrieve all users (ID, Name, Created At)."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, name, created_at FROM users ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    users = []
    for r in rows:
        users.append({"id": r[0], "name": r[1], "created_at": r[2]})
    return users


def delete_user_by_id(user_id):
    """Delete a user by ID."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def update_user_embedding(user_id, new_embedding):
    """
    Replace the stored embedding for a user with the given numpy array.

    Called by the adaptive learning system after applying EMA to blend the
    stored embedding with the latest recognition result. The embedding has
    already been updated (EMA applied + re-normalized) before this call.

    Args:
        user_id:       Integer ID of the user to update
        new_embedding: numpy float32 array of the updated face embedding
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    emb_bytes = new_embedding.astype(np.float32).tobytes()
    c.execute(
        "UPDATE users SET embedding = ? WHERE id = ?",
        (emb_bytes, user_id),
    )
    conn.commit()
    conn.close()


def get_user_embedding(user_id):
    """
    Retrieve the stored embedding for a single user by ID.

    Returns:
        numpy float32 array, or None if user not found.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT embedding FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32).copy()


def get_user_count():
    """Get total number of enrolled users."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    count = c.fetchone()[0]
    conn.close()
    return count


def reset_db():
    """Drop and recreate the users table."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS users")
    conn.commit()
    conn.close()
    init_db()

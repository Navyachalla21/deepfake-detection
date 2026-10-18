"""User + result storage. Uses MongoDB when MONGO_URI is set and reachable,
otherwise a local JSON file (instance/local_db.json) so data survives restarts."""
import datetime
import json
import os
import threading
import uuid
from pathlib import Path


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class JsonStore:
    kind = "local JSON file"

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        if not self.path.exists():
            self._write({"users": [], "results": []})

    def _read(self):
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def get_user_by_email(self, email):
        with self.lock:
            return next((u for u in self._read()["users"] if u["email"] == email), None)

    def create_user(self, username, email, password_hash):
        user = {"id": uuid.uuid4().hex, "username": username, "email": email,
                "password_hash": password_hash, "created_at": now_iso()}
        with self.lock:
            data = self._read()
            data["users"].append(user)
            self._write(data)
        return user

    def add_result(self, record):
        with self.lock:
            data = self._read()
            data["results"].append(record)
            self._write(data)

    def recent_results(self, user_id, limit=8):
        with self.lock:
            rows = [r for r in self._read()["results"] if r.get("user_id") == user_id]
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return rows[:limit]


class MongoStore:
    kind = "MongoDB"

    def __init__(self, uri):
        from pymongo import MongoClient
        self.client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self.client.server_info()  # raises if unreachable
        db = self.client["deepfake_db"]
        self.users, self.results = db["users"], db["results"]

    def get_user_by_email(self, email):
        u = self.users.find_one({"email": email})
        if u:
            u["id"] = str(u.pop("_id"))
        return u

    def create_user(self, username, email, password_hash):
        doc = {"username": username, "email": email, "password_hash": password_hash, "created_at": now_iso()}
        res = self.users.insert_one(doc)
        return {"id": str(res.inserted_id), **{k: v for k, v in doc.items() if k != "_id"}}

    def add_result(self, record):
        self.results.insert_one(dict(record))

    def recent_results(self, user_id, limit=8):
        return list(self.results.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1).limit(limit))


def get_store(instance_dir):
    uri = os.environ.get("MONGO_URI")
    if uri:
        try:
            return MongoStore(uri)
        except Exception as e:
            print(f"[db] MongoDB not reachable ({e.__class__.__name__}); using local JSON file instead.")
    return JsonStore(Path(instance_dir) / "local_db.json")

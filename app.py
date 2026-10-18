import datetime
import os
import re
import secrets
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

load_dotenv()

from models import analyze, model_status  # noqa: E402
from models.errors import MediaError, ModelNotTrainedError  # noqa: E402
from utils.db import get_store, now_iso  # noqa: E402
from utils.storage import get_storage  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
INSTANCE_DIR.mkdir(exist_ok=True)


def load_secret_key() -> str:
    """Stable secret so logins survive restarts (a new random key per start logs everyone out,
    and breaks sessions entirely when several workers each pick their own)."""
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    key_file = INSTANCE_DIR / "secret_key"
    if key_file.exists():
        return key_file.read_text().strip()
    key = secrets.token_hex(32)
    key_file.write_text(key)
    return key


MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))

app = Flask(__name__)
app.secret_key = load_secret_key()
app.config.update(
    MAX_CONTENT_LENGTH=MAX_MB * 1024 * 1024,
    PERMANENT_SESSION_LIFETIME=datetime.timedelta(hours=2),  # idle timeout
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

ALLOWED = {
    "image": {"jpg", "jpeg", "png", "webp"},
    "video": {"mp4", "avi", "mov", "mkv", "webm"},
    "audio": {"wav", "mp3", "flac", "ogg", "m4a"},
}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

store = get_store(INSTANCE_DIR)
storage = get_storage(BASE_DIR)
print(f"[app] database: {store.kind} | file storage: {storage.backend}")


# ---------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        if not username or not EMAIL_RE.match(email):
            flash("Enter a username and a valid email address.", "error")
            return redirect(url_for("register"))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return redirect(url_for("register"))
        if store.get_user_by_email(email):
            flash("Email already registered", "error")
            return redirect(url_for("register"))
        store.create_user(username, email, generate_password_hash(password))
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        user = store.get_user_by_email(email)
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session.permanent = True
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        flash("Invalid email or password", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", status=model_status())


# ---------------------------------------------------------------- API
@app.route("/api/status")
def api_status():
    return jsonify(model_status())


@app.route("/api/history")
def api_history():
    if "user_id" not in session:
        return jsonify(error="Please log in again."), 401
    rows = store.recent_results(session["user_id"], limit=8)
    keys = ("original_filename", "media_type", "prediction", "confidence", "created_at")
    return jsonify([{k: r.get(k) for k in keys} for r in rows])


@app.route("/upload", methods=["POST"])
def upload():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(error="Please log in again."), 401

    media_type = request.form.get("media_type", "image")
    if media_type not in ALLOWED:
        return jsonify(error="Invalid media type."), 400
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="No file selected."), 400

    ext = Path(file.filename).suffix.lstrip(".").lower()
    if ext not in ALLOWED[media_type]:
        other = next((t for t, exts in ALLOWED.items() if ext in exts), None)
        if other:
            msg = f"This looks like a {other} file, but the {media_type.title()} tab is selected."
        else:
            msg = f"Unsupported file type .{ext or '?'}. Allowed here: " + ", ".join(sorted(ALLOWED[media_type]))
        return jsonify(error=msg), 400

    original = secure_filename(file.filename)
    if not original.lower().endswith("." + ext):
        original = f"upload.{ext}"
    unique = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')}_{original}"

    tmp_dir = tempfile.mkdtemp(prefix="dfd_")
    tmp_path = os.path.join(tmp_dir, unique)
    try:
        file.save(tmp_path)
        try:
            result = analyze(tmp_path, media_type)
        except ModelNotTrainedError as e:
            return jsonify(error=str(e), code="model_not_trained"), 503
        except MediaError as e:
            return jsonify(error=str(e)), 400
        except Exception as e:  # noqa: BLE001
            app.logger.exception("Detection failed")
            return jsonify(error=f"Detection failed: {e.__class__.__name__}: {e}"), 500

        saved = storage.save(tmp_path, user_id, media_type, unique)
        store.add_result({
            "user_id": user_id,
            "filename": unique,
            "original_filename": original,
            "media_type": media_type,
            "prediction": result["label"],
            "confidence": result["confidence"],
            "fake_probability": result.get("fake_probability"),
            "details": result.get("details", ""),
            "frames_analyzed": result.get("frames_analyzed"),
            "temporal_score": result.get("temporal_score"),
            "acoustic_score": result.get("acoustic_score"),
            "storage": saved,
            "created_at": now_iso(),
        })
        return jsonify(success=True, filename=original, result=result)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error=f"File is too large (limit {MAX_MB} MB)."), 413


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", "5000")))

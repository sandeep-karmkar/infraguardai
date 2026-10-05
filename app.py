import os
import uuid
import math
from datetime import datetime, timezone
from urllib.parse import urlparse, unquote

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from supabase import create_client, Client

from ml.src.predict import predict_asset


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is missing from .env")

if not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_ANON_KEY is missing from .env")

if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is missing from .env")


app = Flask(__name__)

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_ANON_KEY
)

db: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY
)


# ============================================================
# STORAGE
# ============================================================

COMPLAINT_PHOTO_BUCKET = "complaint-photos"
PHOTO_SIGNED_URL_SECONDS = 60 * 60

ALLOWED_PHOTO_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}

ALLOWED_PHOTO_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}

MAX_PHOTO_SIZE = 8 * 1024 * 1024


# ============================================================
# AUTH HELPERS
# ============================================================

def get_access_token():
    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        return None

    return auth_header.split(" ", 1)[1].strip()


def get_authenticated_user():
    token = get_access_token()

    if not token:
        return None

    try:
        response = supabase.auth.get_user(token)

        if response is None:
            return None

        return response.user

    except Exception as e:
        print("AUTH ERROR:", e)
        return None


def get_profile(user_id):
    try:
        response = (
            db.table("profiles")
            .select("*")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )

        return response.data

    except Exception as e:
        print("PROFILE ERROR:", e)
        return None


def require_officer():
    user = get_authenticated_user()

    if not user:
        return None, None, (
            jsonify({"error": "Authentication required"}),
            401
        )

    profile = get_profile(user.id)

    if not profile:
        return None, None, (
            jsonify({"error": "Profile not found"}),
            403
        )

    if profile.get("role") not in ["officer", "admin"]:
        return None, None, (
            jsonify({"error": "Officer access required"}),
            403
        )

    return user, profile, None


def require_inspector():
    user = get_authenticated_user()

    if not user:
        return None, None, (
            jsonify({"error": "Authentication required"}),
            401
        )

    profile = get_profile(user.id)

    if not profile:
        return None, None, (
            jsonify({"error": "Profile not found"}),
            403
        )

    if profile.get("role") not in ["inspector", "admin"]:
        return None, None, (
            jsonify({"error": "Inspector access required"}),
            403
        )

    return user, profile, None


# ============================================================
# GENERAL HELPERS
# ============================================================

def generate_complaint_code():
    return f"INF-{uuid.uuid4().hex[:8].upper()}"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize_coordinates(latitude, longitude):
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        return None, None

    if not -90 <= latitude <= 90:
        return None, None

    if not -180 <= longitude <= 180:
        return None, None

    return latitude, longitude


def haversine_distance_km(lat1, lon1, lat2, lon2):
    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except (TypeError, ValueError):
        return None

    earth_radius_km = 6371.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return earth_radius_km * c


# ============================================================
# PHOTO HELPERS
# ============================================================

def extract_storage_path(photo_value):
    if not photo_value:
        return None

    photo_value = str(photo_value).strip()

    if not photo_value:
        return None

    # Already a Storage path.
    if not photo_value.startswith(("http://", "https://")):
        path = photo_value.lstrip("/")

        prefix = f"{COMPLAINT_PHOTO_BUCKET}/"

        if path.startswith(prefix):
            path = path[len(prefix):]

        return path

    try:
        parsed = urlparse(photo_value)
        path = unquote(parsed.path)

        markers = [
            f"/storage/v1/object/public/{COMPLAINT_PHOTO_BUCKET}/",
            f"/storage/v1/object/sign/{COMPLAINT_PHOTO_BUCKET}/",
            f"/storage/v1/object/authenticated/{COMPLAINT_PHOTO_BUCKET}/"
        ]

        for marker in markers:
            if marker in path:
                return path.split(marker, 1)[1].lstrip("/")

    except Exception as e:
        print("PHOTO URL PARSE ERROR:", e)

    return None


def get_complaint_photo_url(photo_value):
    if not photo_value:
        return None

    photo_value = str(photo_value).strip()

    if not photo_value:
        return None

    storage_path = extract_storage_path(photo_value)

    if storage_path:
        try:
            response = (
                db.storage
                .from_(COMPLAINT_PHOTO_BUCKET)
                .create_signed_url(
                    storage_path,
                    PHOTO_SIGNED_URL_SECONDS
                )
            )

            signed_url = None

            if isinstance(response, dict):
                signed_url = (
                    response.get("signedURL")
                    or response.get("signedUrl")
                    or response.get("signed_url")
                )

                if not signed_url:
                    nested = response.get("data")

                    if isinstance(nested, dict):
                        signed_url = (
                            nested.get("signedURL")
                            or nested.get("signedUrl")
                            or nested.get("signed_url")
                        )

            if signed_url:
                return signed_url

            print("SIGNED URL RESPONSE:", response)

        except Exception as e:
            print("SIGNED URL ERROR:", e)

        return None

    if photo_value.startswith(
        ("http://", "https://", "data:")
    ):
        return photo_value

    return None


def attach_photo_url(complaint):
    if not complaint:
        return complaint

    complaint = dict(complaint)

    original = complaint.get("photo_url")

    complaint["storage_photo_path"] = original

    complaint["photo_url"] = (
        get_complaint_photo_url(original)
        if original
        else None
    )

    return complaint


def attach_photo_urls(complaints):
    return [
        attach_photo_url(c)
        for c in (complaints or [])
    ]


def upload_complaint_photo(file, user_id):
    """
    Upload citizen photo to Supabase Storage.

    Returns the Storage object path.
    """

    if not file:
        return None

    filename = file.filename or ""

    if not filename:
        return None

    extension = (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else ""
    )

    if extension not in ALLOWED_PHOTO_EXTENSIONS:
        raise ValueError(
            "Only JPG, JPEG, PNG and WEBP images are allowed."
        )

    content_type = (
        file.content_type
        or ""
    ).lower()

    if content_type not in ALLOWED_PHOTO_TYPES:
        raise ValueError(
            "Invalid image content type."
        )

    file_bytes = file.read()

    if not file_bytes:
        raise ValueError(
            "The uploaded photo is empty."
        )

    if len(file_bytes) > MAX_PHOTO_SIZE:
        raise ValueError(
            "Photo must be 8 MB or smaller."
        )

    safe_extension = (
        "jpg"
        if extension == "jpeg"
        else extension
    )

    storage_path = (
        f"complaints/"
        f"{user_id}/"
        f"{uuid.uuid4().hex}."
        f"{safe_extension}"
    )

    try:
        db.storage.from_(
            COMPLAINT_PHOTO_BUCKET
        ).upload(
            storage_path,
            file_bytes,
            {
                "content-type": content_type,
                "upsert": "false"
            }
        )

        print(
            "PHOTO UPLOADED:",
            storage_path
        )

        return storage_path

    except Exception as e:
        print(
            "PHOTO UPLOAD ERROR:",
            e
        )
        raise


# ============================================================
# ASSET HELPERS
# ============================================================

def get_asset_type_for_category(category):
    mapping = {
        "Road": "Road",
        "Streetlight": "Streetlight",
        "Drainage": "Drainage",
        "Water system": "Water_Pipeline",
        "Public building": "Public_Building",
        "Bridge": "Bridge"
    }

    return mapping.get(category)


def find_nearest_asset(category, latitude, longitude):
    latitude, longitude = normalize_coordinates(
        latitude,
        longitude
    )

    if latitude is None or longitude is None:
        return None

    try:
        response = (
            db.table("assets")
            .select("*")
            .eq("status", "ACTIVE")
            .execute()
        )

        assets = response.data or []

    except Exception as e:
        print("ASSET LOOKUP ERROR:", e)
        return None

    if not assets:
        return None

    compatible_type = get_asset_type_for_category(category)

    compatible = []
    fallback = []

    for asset in assets:

        asset_lat, asset_lon = normalize_coordinates(
            asset.get("latitude"),
            asset.get("longitude")
        )

        if asset_lat is None or asset_lon is None:
            continue

        distance = haversine_distance_km(
            latitude,
            longitude,
            asset_lat,
            asset_lon
        )

        if distance is None or distance > 3:
            continue

        item = {
            "asset": asset,
            "distance_km": distance
        }

        fallback.append(item)

        if (
            compatible_type
            and asset.get("asset_type") == compatible_type
        ):
            compatible.append(item)

    if compatible:
        compatible.sort(
            key=lambda x: x["distance_km"]
        )
        return compatible[0]

    if fallback:
        fallback.sort(
            key=lambda x: x["distance_km"]
        )
        return fallback[0]

    return None


# ============================================================
# PAGE ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "InfraGuardAI"
    })


@app.route("/login")
def login_page():
    return render_template("login.html")


@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/officer-dashboard")
def officer_dashboard_page():
    return render_template("officer_dashboard.html")


@app.route("/inspector-dashboard")
def inspector_dashboard_page():
    return render_template("inspector_dashboard.html")


@app.route("/report-issue")
def report_issue_page():
    return render_template("report_issue.html")


# ============================================================
# AUTH
# ============================================================

@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json(silent=True) or {}

        email = data.get("email", "").strip()
        password = data.get("password", "")
        full_name = data.get("full_name", "").strip()
        phone = data.get("phone", "").strip()

        if not email or not password or not full_name:
            return jsonify({
                "error": "Name, email and password are required"
            }), 400

        response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "full_name": full_name,
                    "phone": phone
                }
            }
        })

        if not response.user:
            return jsonify({
                "error": "Registration failed"
            }), 400

        return jsonify({
            "message": "Registration successful",
            "user_id": response.user.id
        }), 201

    except Exception as e:
        print("REGISTER ERROR:", e)
        return jsonify({"error": str(e)}), 400


@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json(silent=True) or {}

        email = data.get("email", "").strip()
        password = data.get("password", "")

        if not email or not password:
            return jsonify({
                "error": "Email and password are required"
            }), 400

        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        if not response.user or not response.session:
            return jsonify({
                "error": "Invalid email or password"
            }), 401

        profile = get_profile(response.user.id)

        return jsonify({
            "message": "Login successful",
            "access_token": response.session.access_token,
            "refresh_token": response.session.refresh_token,
            "user_id": response.user.id,
            "profile": profile
        })

    except Exception as e:
        print("LOGIN ERROR:", e)
        return jsonify({"error": str(e)}), 401


# ============================================================
# CITIZEN — CREATE COMPLAINT
# ============================================================

@app.route("/api/complaints", methods=["POST"])
def create_complaint():

    try:
        user = get_authenticated_user()

        if not user:
            return jsonify({
                "error": "Authentication required"
            }), 401

        # ----------------------------------------------------
        # IMPORTANT:
        # Support multipart/form-data for real photo upload.
        # Also support JSON for complaints without photos.
        # ----------------------------------------------------

        is_multipart = (
            request.content_type
            and request.content_type.startswith(
                "multipart/form-data"
            )
        )

        if is_multipart:
            data = request.form
            uploaded_photo = request.files.get("photo")
        else:
            data = request.get_json(
                silent=True
            ) or {}
            uploaded_photo = None

        required_fields = [
            "category",
            "problem_type",
            "description",
            "severity"
        ]

        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    "error": f"{field} is required"
                }), 400

        category = data.get("category")

        latitude, longitude = normalize_coordinates(
            data.get("latitude"),
            data.get("longitude")
        )

        # ----------------------------------------------------
        # Find nearest asset
        # ----------------------------------------------------

        matched_asset = None
        asset_linked = False
        asset_distance_km = None

        if latitude is not None and longitude is not None:
            matched_asset = find_nearest_asset(
                category,
                latitude,
                longitude
            )

        if matched_asset:

            matched_asset_data = matched_asset["asset"]

            asset_id = matched_asset_data.get("id")

            asset_linked = bool(asset_id)

            asset_distance_km = round(
                matched_asset["distance_km"],
                3
            )

            print(
                "ASSET MATCH:",
                matched_asset_data.get("asset_code"),
                asset_distance_km,
                "km"
            )

        else:
            asset_id = None

            print(
                "ASSET MATCH: No active asset within 3 km"
            )

        # ----------------------------------------------------
        # PHOTO UPLOAD
        # ----------------------------------------------------

        photo_url = None

        if uploaded_photo:

            try:
                photo_url = upload_complaint_photo(
                    uploaded_photo,
                    user.id
                )

            except ValueError as e:

                return jsonify({
                    "error": str(e)
                }), 400

            except Exception as e:

                print(
                    "PHOTO UPLOAD FAILED:",
                    e
                )

                return jsonify({
                    "error": (
                        "Photo upload failed. "
                        "Please try again."
                    )
                }), 500

        else:

            # Backward compatibility:
            # allow existing photo_url submissions.
            photo_url = data.get("photo_url")

        # ----------------------------------------------------
        # Save complaint
        # ----------------------------------------------------

        complaint = {
            "complaint_code": generate_complaint_code(),
            "citizen_id": user.id,
            "asset_id": asset_id,
            "category": category,
            "problem_type": data.get("problem_type"),
            "description": data.get("description"),
            "latitude": latitude,
            "longitude": longitude,
            "photo_url": photo_url,
            "severity": data.get("severity"),
            "status": "SUBMITTED"
        }

        response = (
            db.table("complaints")
            .insert(complaint)
            .execute()
        )

        if not response.data:
            return jsonify({
                "error": "Complaint could not be saved"
            }), 500

        saved_complaint = attach_photo_url(
            response.data[0]
        )

        return jsonify({
            "message": "Complaint submitted successfully",
            "complaint": saved_complaint,
            "asset_linked": asset_linked,
            "asset_distance_km": asset_distance_km,
            "matched_asset": (
                matched_asset["asset"]
                if matched_asset
                else None
            )
        }), 201

    except Exception as e:

        print(
            "CREATE COMPLAINT ERROR:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# CITIZEN — COMPLAINTS
# ============================================================

@app.route("/api/complaints", methods=["GET"])
def get_citizen_complaints():

    try:
        user = get_authenticated_user()

        if not user:
            return jsonify({
                "error": "Authentication required"
            }), 401

        response = (
            db.table("complaints")
            .select("*")
            .eq("citizen_id", user.id)
            .order("created_at", desc=True)
            .execute()
        )

        return jsonify({
            "complaints": attach_photo_urls(
                response.data or []
            )
        })

    except Exception as e:

        print("CITIZEN COMPLAINT ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/api/complaints/<complaint_id>", methods=["GET"])
def get_complaint(complaint_id):

    try:
        user = get_authenticated_user()

        if not user:
            return jsonify({
                "error": "Authentication required"
            }), 401

        response = (
            db.table("complaints")
            .select("*")
            .eq("id", complaint_id)
            .maybe_single()
            .execute()
        )

        complaint = response.data

        if not complaint:
            return jsonify({
                "error": "Complaint not found"
            }), 404

        if complaint.get("citizen_id") != user.id:

            profile = get_profile(user.id)

            if (
                not profile
                or profile.get("role") not in [
                    "officer",
                    "admin",
                    "inspector"
                ]
            ):
                return jsonify({
                    "error": "Access denied"
                }), 403

        return jsonify({
            "complaint": attach_photo_url(
                complaint
            )
        })

    except Exception as e:

        print("GET COMPLAINT ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# OFFICER — COMPLAINTS
# ============================================================

@app.route("/api/officer/complaints", methods=["GET"])
def officer_complaints():

    user, profile, error = require_officer()

    if error:
        return error

    try:

        response = (
            db.table("complaints")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )

        complaints = attach_photo_urls(
            response.data or []
        )

        return jsonify({
            "complaints": complaints
        })

    except Exception as e:

        print("OFFICER COMPLAINT ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


@app.route(
    "/api/officer/complaints/<complaint_id>/status",
    methods=["PATCH"]
)
def update_complaint_status(complaint_id):

    user, profile, error = require_officer()

    if error:
        return error

    try:

        data = request.get_json(
            silent=True
        ) or {}

        new_status = data.get("status")

        allowed_statuses = [
            "SUBMITTED",
            "UNDER REVIEW",
            "VERIFIED",
            "PRIORITIZED",
            "ASSIGNED",
            "IN PROGRESS",
            "INSPECTION",
            "RESOLVED",
            "CITIZEN CONFIRMED",
            "CLOSED",
            "REJECTED",
            "DUPLICATE",
            "REOPENED"
        ]

        if new_status not in allowed_statuses:
            return jsonify({
                "error": "Invalid status"
            }), 400

        response = (
            db.table("complaints")
            .update({
                "status": new_status,
                "updated_at": utc_now()
            })
            .eq("id", complaint_id)
            .execute()
        )

        if not response.data:
            return jsonify({
                "error": "Complaint not found"
            }), 404

        return jsonify({
            "message": "Complaint status updated",
            "complaint": attach_photo_url(
                response.data[0]
            )
        })

    except Exception as e:

        print("STATUS UPDATE ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# OFFICER — ASSETS
# ============================================================

@app.route("/api/officer/assets", methods=["GET"])
def officer_assets():

    user, profile, error = require_officer()

    if error:
        return error

    try:

        response = (
            db.table("assets")
            .select("*")
            .order("asset_code")
            .execute()
        )

        return jsonify({
            "assets": response.data or []
        })

    except Exception as e:

        print("OFFICER ASSET ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/api/assets", methods=["GET"])
def get_assets():

    user, profile, error = require_officer()

    if error:
        return error

    try:

        response = (
            db.table("assets")
            .select("*")
            .order("asset_code")
            .execute()
        )

        return jsonify({
            "assets": response.data or []
        })

    except Exception as e:

        print("ASSETS ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


@app.route(
    "/api/officer/assets/<asset_id>",
    methods=["GET"]
)
def officer_asset(asset_id):

    user, profile, error = require_officer()

    if error:
        return error

    try:

        response = (
            db.table("assets")
            .select("*")
            .eq("id", asset_id)
            .maybe_single()
            .execute()
        )

        if not response.data:
            return jsonify({
                "error": "Asset not found"
            }), 404

        return jsonify({
            "asset": response.data
        })

    except Exception as e:

        print("ASSET DETAILS ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# AI PREDICTION
# ============================================================

@app.route(
    "/api/officer/predict/<asset_id>",
    methods=["POST"]
)
def predict(asset_id):

    user, profile, error = require_officer()

    if error:
        return error

    try:

        asset_response = (
            db.table("assets")
            .select("*")
            .eq("id", asset_id)
            .maybe_single()
            .execute()
        )

        asset = asset_response.data

        if not asset:
            return jsonify({
                "error": "Asset not found"
            }), 404

        ml_input = {
            "asset_type": asset.get("asset_type"),
            "age_years": float(
                asset.get("age_years", 0)
            ),
            "condition_score": float(
                asset.get("condition_score", 50)
            ),
            "complaint_count": int(
                asset.get("complaint_count", 0)
            ),
            "failure_count": int(
                asset.get("failure_count", 0)
            ),
            "days_since_maintenance": int(
                asset.get("days_since_maintenance", 0)
            ),
            "inspection_score": float(
                asset.get("inspection_score", 50)
            ),
            "environmental_risk": asset.get(
                "environmental_risk",
                "Low"
            ),
            "usage_level": asset.get(
                "usage_level",
                "Medium"
            )
        }

        result = predict_asset(ml_input)

        prediction_record = {
            "asset_id": asset_id,
            "risk_score": result["risk"]["score"],
            "risk_category": result["risk"]["category"],
            "maintenance_probability": (
                result["maintenance_probability"]
            ),
            "prediction_date": utc_now()
        }

        db.table("predictions").insert(
            prediction_record
        ).execute()

        return jsonify({
            "asset": asset,
            "prediction": result
        })

    except Exception as e:

        print("PREDICTION ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/api/predictions", methods=["GET"])
def get_predictions():

    user, profile, error = require_officer()

    if error:
        return error

    try:

        response = (
            db.table("predictions")
            .select("*")
            .order("prediction_date", desc=True)
            .execute()
        )

        return jsonify({
            "predictions": response.data or []
        })

    except Exception as e:

        print("PREDICTIONS ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# INSPECTOR — QUEUE
# ============================================================

@app.route(
    "/api/inspector/complaints",
    methods=["GET"]
)
def inspector_complaints():

    user, profile, error = require_inspector()

    if error:
        return error

    try:

        response = (
            db.table("complaints")
            .select("*")
            .eq("status", "INSPECTION")
            .order("updated_at", desc=True)
            .execute()
        )

        complaints = attach_photo_urls(
            response.data or []
        )

        for complaint in complaints:

            asset_id = complaint.get("asset_id")
            complaint["asset"] = None

            if asset_id:

                asset_response = (
                    db.table("assets")
                    .select("*")
                    .eq("id", asset_id)
                    .maybe_single()
                    .execute()
                )

                complaint["asset"] = (
                    asset_response.data
                )

        return jsonify({
            "complaints": complaints,
            "count": len(complaints)
        })

    except Exception as e:

        print("INSPECTOR COMPLAINT ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# INSPECTOR — HISTORY
# ============================================================

@app.route(
    "/api/inspector/inspections",
    methods=["GET"]
)
def inspector_inspections():

    user, profile, error = require_inspector()

    if error:
        return error

    try:

        query = (
            db.table("inspections")
            .select("*")
            .order("created_at", desc=True)
        )

        if profile.get("role") == "inspector":
            query = query.eq(
                "inspector_id",
                user.id
            )

        response = query.execute()

        inspections = response.data or []

        for inspection in inspections:

            complaint_id = inspection.get(
                "complaint_id"
            )

            asset_id = inspection.get(
                "asset_id"
            )

            inspection["complaint"] = None
            inspection["asset"] = None

            if complaint_id:

                complaint_response = (
                    db.table("complaints")
                    .select("*")
                    .eq("id", complaint_id)
                    .maybe_single()
                    .execute()
                )

                complaint = (
                    complaint_response.data
                )

                if complaint:
                    inspection["complaint"] = (
                        attach_photo_url(
                            complaint
                        )
                    )

            if asset_id:

                asset_response = (
                    db.table("assets")
                    .select("*")
                    .eq("id", asset_id)
                    .maybe_single()
                    .execute()
                )

                inspection["asset"] = (
                    asset_response.data
                )

        return jsonify({
            "inspections": inspections,
            "count": len(inspections)
        })

    except Exception as e:

        print("INSPECTOR HISTORY ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# INSPECTOR — SUBMIT INSPECTION
# ============================================================

@app.route(
    "/api/inspector/complaints/<complaint_id>/inspection",
    methods=["POST"]
)
def submit_inspection(complaint_id):

    user, profile, error = require_inspector()

    if error:
        return error

    try:

        data = request.get_json(
            silent=True
        ) or {}

        score = data.get("score")
        decision = data.get("decision")

        observations = (
            data.get("observations") or ""
        ).strip()

        evidence_url = data.get(
            "evidence_url"
        )

        if score is None:
            return jsonify({
                "error": "Inspection score is required"
            }), 400

        try:
            score = float(score)
        except (TypeError, ValueError):
            return jsonify({
                "error": "Inspection score must be a number"
            }), 400

        if score < 0 or score > 100:
            return jsonify({
                "error": "Inspection score must be between 0 and 100"
            }), 400

        if decision not in ["PASS", "FAIL"]:
            return jsonify({
                "error": "Decision must be PASS or FAIL"
            }), 400

        if not observations:
            return jsonify({
                "error": "Inspection observations are required"
            }), 400

        complaint_response = (
            db.table("complaints")
            .select("*")
            .eq("id", complaint_id)
            .maybe_single()
            .execute()
        )

        complaint = complaint_response.data

        if not complaint:
            return jsonify({
                "error": "Complaint not found"
            }), 404

        if complaint.get("status") != "INSPECTION":
            return jsonify({
                "error": (
                    "Complaint is not currently "
                    "waiting for inspection"
                )
            }), 400

        asset_id = complaint.get("asset_id")

        inspection_record = {
            "complaint_id": complaint_id,
            "inspector_id": user.id,
            "asset_id": asset_id,
            "score": score,
            "decision": decision,
            "observations": observations,
            "evidence_url": evidence_url
        }

        inspection_response = (
            db.table("inspections")
            .insert(inspection_record)
            .execute()
        )

        if not inspection_response.data:
            return jsonify({
                "error": "Inspection could not be saved"
            }), 500

        inspection = inspection_response.data[0]

        if asset_id:

            db.table("assets").update({
                "inspection_score": score,
                "updated_at": utc_now()
            }).eq(
                "id",
                asset_id
            ).execute()

        next_status = (
            "RESOLVED"
            if decision == "PASS"
            else "REOPENED"
        )

        complaint_update = (
            db.table("complaints")
            .update({
                "status": next_status,
                "updated_at": utc_now()
            })
            .eq("id", complaint_id)
            .execute()
        )

        if not complaint_update.data:
            return jsonify({
                "error": (
                    "Inspection saved but complaint "
                    "status could not be updated"
                ),
                "inspection": inspection
            }), 500

        return jsonify({
            "message": "Inspection submitted successfully",
            "inspection": inspection,
            "complaint": attach_photo_url(
                complaint_update.data[0]
            ),
            "next_status": next_status
        }), 201

    except Exception as e:

        print("SUBMIT INSPECTION ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# INSPECTOR — DETAIL
# ============================================================

@app.route(
    "/api/inspector/complaints/<complaint_id>",
    methods=["GET"]
)
def inspector_complaint_detail(complaint_id):

    user, profile, error = require_inspector()

    if error:
        return error

    try:

        complaint_response = (
            db.table("complaints")
            .select("*")
            .eq("id", complaint_id)
            .maybe_single()
            .execute()
        )

        complaint = complaint_response.data

        if not complaint:
            return jsonify({
                "error": "Complaint not found"
            }), 404

        complaint = attach_photo_url(
            complaint
        )

        asset = None

        if complaint.get("asset_id"):

            asset_response = (
                db.table("assets")
                .select("*")
                .eq(
                    "id",
                    complaint["asset_id"]
                )
                .maybe_single()
                .execute()
            )

            asset = asset_response.data

        inspection_response = (
            db.table("inspections")
            .select("*")
            .eq("complaint_id", complaint_id)
            .order("created_at", desc=True)
            .execute()
        )

        return jsonify({
            "complaint": complaint,
            "asset": asset,
            "inspections": (
                inspection_response.data or []
            )
        })

    except Exception as e:

        print(
            "INSPECTOR DETAIL ERROR:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Route not found",
        "path": request.path
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error"
    }), 500


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
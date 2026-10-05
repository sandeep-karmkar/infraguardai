"""
InfraGuard AI
Inspector workflow routes

Uses the same authentication system as app.py so the
Inspector Command Center receives and validates the
Supabase bearer token consistently.
"""

from flask import jsonify


def register_inspector_routes(app_module):
    """
    Register all inspector routes on the existing Flask app.
    """

    app = app_module.app
    db = app_module.db

    # ============================================================
    # AUTHENTICATION
    # ============================================================

    def require_inspector():
        """
        Reuse the canonical authentication logic from app.py.

        Returns:
            (user, profile, error_response)
        """

        user = app_module.get_authenticated_user()

        if not user:
            return (
                None,
                None,
                (
                    jsonify({
                        "error": "Authentication required"
                    }),
                    401
                )
            )

        profile = app_module.get_profile(user.id)

        if not profile:
            return (
                None,
                None,
                (
                    jsonify({
                        "error": "Profile not found"
                    }),
                    403
                )
            )

        if profile.get("role") not in [
            "inspector",
            "admin"
        ]:
            return (
                None,
                None,
                (
                    jsonify({
                        "error": "Inspector access required"
                    }),
                    403
                )
            )

        return user, profile, None

    # ============================================================
    # INSPECTOR DASHBOARD PAGE
    # ============================================================

    @app.route("/inspector-dashboard")
    def inspector_dashboard_page():

        return app_module.render_template(
            "inspector_dashboard.html"
        )

    # ============================================================
    # INSPECTION QUEUE
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
                .order(
                    "created_at",
                    desc=False
                )
                .execute()
            )

            complaints = response.data or []

            # Attach related asset information.
            for complaint in complaints:

                complaint["asset"] = None

                asset_id = complaint.get("asset_id")

                if not asset_id:
                    continue

                try:

                    asset_response = (
                        db.table("assets")
                        .select("*")
                        .eq("id", asset_id)
                        .limit(1)
                        .execute()
                    )

                    asset_rows = (
                        asset_response.data
                        or []
                    )

                    if asset_rows:
                        complaint["asset"] = (
                            asset_rows[0]
                        )

                except Exception as asset_error:

                    print(
                        "INSPECTOR ASSET LOOKUP ERROR:",
                        asset_error
                    )

            return jsonify({
                "complaints": complaints
            })

        except Exception as e:

            print(
                "INSPECTOR COMPLAINT ERROR:",
                e
            )

            return jsonify({
                "error": str(e)
            }), 500

    # ============================================================
    # INSPECTION HISTORY
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

            response = (
                db.table("inspections")
                .select("*")
                .eq(
                    "inspector_id",
                    user.id
                )
                .order(
                    "created_at",
                    desc=True
                )
                .execute()
            )

            return jsonify({
                "inspections": (
                    response.data
                    or []
                )
            })

        except Exception as e:

            print(
                "INSPECTION HISTORY ERROR:",
                e
            )

            return jsonify({
                "error": str(e)
            }), 500

    # ============================================================
    # SUBMIT FIELD INSPECTION
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

            data = (
                app_module.request
                .get_json(
                    silent=True
                )
                or {}
            )

            # ----------------------------------------------------
            # Validate score
            # ----------------------------------------------------

            score = data.get("score")

            try:

                score = float(score)

            except (
                TypeError,
                ValueError
            ):

                return jsonify({
                    "error":
                        "Inspection score must be a number."
                }), 400

            if score < 0 or score > 100:

                return jsonify({
                    "error":
                        "Inspection score must be between 0 and 100."
                }), 400

            # ----------------------------------------------------
            # Validate decision
            # ----------------------------------------------------

            decision = str(
                data.get(
                    "decision",
                    ""
                )
            ).strip().upper()

            if decision not in [
                "PASS",
                "FAIL"
            ]:

                return jsonify({
                    "error":
                        "Decision must be PASS or FAIL."
                }), 400

            # ----------------------------------------------------
            # Validate observations
            # ----------------------------------------------------

            observations = str(
                data.get(
                    "observations",
                    ""
                )
            ).strip()

            if not observations:

                return jsonify({
                    "error":
                        "Inspection observations are required."
                }), 400

            evidence_url = str(
                data.get(
                    "evidence_url",
                    ""
                )
            ).strip()

            # ----------------------------------------------------
            # Load complaint
            # ----------------------------------------------------

            complaint_response = (
                db.table("complaints")
                .select("*")
                .eq(
                    "id",
                    complaint_id
                )
                .limit(1)
                .execute()
            )

            complaint_rows = (
                complaint_response.data
                or []
            )

            if not complaint_rows:

                return jsonify({
                    "error":
                        "Complaint not found."
                }), 404

            complaint = complaint_rows[0]

            # ----------------------------------------------------
            # Ensure correct workflow stage
            # ----------------------------------------------------

            if complaint.get("status") != "INSPECTION":

                return jsonify({
                    "error":
                        "This complaint is not currently awaiting inspection.",
                    "current_status":
                        complaint.get("status")
                }), 400

            # ----------------------------------------------------
            # Insert inspection record
            # ----------------------------------------------------

            inspection = {
                "complaint_id": complaint_id,
                "inspector_id": user.id,
                "asset_id": complaint.get(
                    "asset_id"
                ),
                "score": score,
                "decision": decision,
                "observations": observations,
                "evidence_url": (
                    evidence_url
                    or None
                )
            }

            inspection_response = (
                db.table("inspections")
                .insert(inspection)
                .execute()
            )

            created_inspection = (
                inspection_response.data[0]
                if inspection_response.data
                else None
            )

            # ----------------------------------------------------
            # Update asset inspection score
            # ----------------------------------------------------

            asset_id = complaint.get(
                "asset_id"
            )

            if asset_id:

                try:

                    (
                        db.table("assets")
                        .update({
                            "inspection_score":
                                score,
                            "updated_at":
                                app_module.utc_now()
                        })
                        .eq(
                            "id",
                            asset_id
                        )
                        .execute()
                    )

                except Exception as asset_error:

                    print(
                        "ASSET INSPECTION UPDATE ERROR:",
                        asset_error
                    )

            # ----------------------------------------------------
            # Move complaint through workflow
            # ----------------------------------------------------

            if decision == "PASS":

                next_status = "RESOLVED"

            else:

                next_status = "REOPENED"

            (
                db.table("complaints")
                .update({
                    "status": next_status,
                    "updated_at":
                        app_module.utc_now()
                })
                .eq(
                    "id",
                    complaint_id
                )
                .execute()
            )

            return jsonify({
                "message":
                    "Inspection submitted successfully.",
                "inspection":
                    created_inspection,
                "next_status":
                    next_status
            }), 201

        except Exception as e:

            print(
                "SUBMIT INSPECTION ERROR:",
                e
            )

            return jsonify({
                "error": str(e)
            }), 500

    print(
        "Inspector routes registered successfully."
    )
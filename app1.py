# app.py
import streamlit as st
from datetime import datetime
import pandas as pd
import json
import os
import time
import tempfile
# Pipeline runs as subprocess — no direct import needed
import subprocess
import signal
import psutil
from streamlit_autorefresh import st_autorefresh
from database import get_incidents, get_workers, get_alerts, get_stats, init_db
init_db()  # creates tables if DB was deleted — safe to call every time
from Incident_Analysis_v2 import get_rag_status    
from Incident_Analysis_v2 import GROQ_report

# ====================== PRODUCTION DESIGN ======================

if 'PIPELINE_PROCESS' not in st.session_state:
    st.session_state.PIPELINE_PROCESS = None
st.set_page_config(
    page_title="AI Safety Monitor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"    
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%); }
    .main-header { font-size: 2.9rem; color: #1f2937; text-align: center; margin-bottom: 0.2rem; font-weight: 700; }
    .sub-header { color: #334155; text-align: center; font-weight: 500; margin-bottom: 1.8rem; }
    .metric-container { background: white; padding: 1.2rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
    .stButton>button { background: #2563eb; color: white; border-radius: 8px; height: 3.2rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://img.icons8.com/fluency/96/000000/construction-worker.png", width=85)
with col_title:
    st.markdown('<h1 class="main-header">AI Powered Construction Safety Monitoring</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time PPE Violation Detection • Risk Intelligence • Groq Analysis</p>', unsafe_allow_html=True)

st.markdown("---")

# ====================== SIDEBAR ======================
st.sidebar.header("📹 Camera Management")

# Load cameras from JSON (Customer can edit this file)
def load_cameras():
    if not os.path.exists("cameras.json"):
        default_cameras = [
            {
                "camera_id": "CAM_01",
                "name": "Main Gate",
                "source": 0,           # Default to your webcam
                "location": "Entrance Area",
                "status": "ONLINE"
            }
        ]
        with open("cameras.json", "w") as f:
            json.dump(default_cameras, f, indent=4)
        return default_cameras
    
    try:
        with open("cameras.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Camera config read error: {e}")
        return []

cameras = load_cameras()

# Camera Selection
camera_options = [f"{cam['camera_id']} - {cam['name']} ({cam['location']})" for cam in cameras]
selected_option = st.sidebar.selectbox("Select Active Camera", camera_options)

selected_cam = next((cam for cam in cameras if f"{cam['camera_id']} - {cam['name']}" in selected_option), cameras[0])

# ====================== IMPORTANT FOR YOU RIGHT NOW ======================
# For now, force all cameras to use your webcam (source = 0)
video_source = 1                    # 1 = back camera, 0 = front camera
camera_id = selected_cam["camera_id"]

# Uncomment below lines when you have multiple real cameras:
# video_source = selected_cam["source"]
# camera_id = selected_cam["camera_id"]

st.sidebar.info(f"📍 Location: **{selected_cam['location']}** | Using Webcam (for testing)")
st.sidebar.caption(f"Camera ID: {camera_id}")

# Upload Video Option (for testing)
st.sidebar.markdown("---")
st.sidebar.subheader("📤 Test with Video")
uploaded_file = st.sidebar.file_uploader("Upload Video File (for testing)", 
                                       type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.getvalue())
        video_source = tmp.name
    st.sidebar.success("✅ Video uploaded - Ready to test")

st.sidebar.markdown("---")

# Pipeline Buttons

if st.sidebar.button("🚀 Start AI Core Pipeline", type="primary", width="stretch"):
    if 'PIPELINE_PROCESS' not in st.session_state or not st.session_state.PIPELINE_PROCESS:
        try:
            # Using sys.executable ensures the exact same python environment and folder context
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pipeline_script = os.path.join(current_dir, "s_p4.py")
            
            st.session_state.PIPELINE_PROCESS = subprocess.Popen(
                [sys.executable, pipeline_script, str(video_source), camera_id],
                cwd=current_dir # Forces the database/logs to remain in this directory
            )
            st.sidebar.success(f"✅ Pipeline started for **{camera_id}**")
            st.sidebar.info("Check terminal for any camera errors")
        except Exception as e:
            st.sidebar.error(f"Failed to start: {e}")
    else:
        st.sidebar.warning("Pipeline already running")
        
if st.sidebar.button("⛔ Stop Pipeline", width="stretch"):
    if 'PIPELINE_PROCESS' in st.session_state and st.session_state.PIPELINE_PROCESS:
        try:
            parent = psutil.Process(st.session_state.PIPELINE_PROCESS.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            st.sidebar.info("Pipeline already stopped")
        except Exception as e:
            st.sidebar.warning(f"Stop error: {e}")
        finally:
            st.session_state.PIPELINE_PROCESS = None
            st.sidebar.success("✅ Pipeline stopped")

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", 
    ["Dashboard", "Executive Summary", "Ask AI", "Incidents", "Workers", "Site Safety", "Reports", "Messages", "Camera Management"])

# ====================== PER-CAMERA PATH HELPER ======================
def get_camera_path(filename):
    """All data is stored per camera folder"""
    base = os.path.join("data", camera_id)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, filename)

# ====================== SAFE READERS (Updated) ======================
def safe_read_json(filename, default=None):
    path = get_camera_path(filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return default or {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return data[0] if isinstance(data[0], dict) else {}
        return data if isinstance(data, dict) else {}
    except:
        return default or {}

def save_cameras(cameras):
    """Save cameras back to JSON"""
    try:
        with open("cameras.json", "w") as f:
            json.dump(cameras, f, indent=4)
        return True
    except:
        st.error("Failed to save cameras")
        return False

def safe_read_json_list(filename, default=None):
    path = get_camera_path(filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return default or []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data] if isinstance(data, dict) else []
    except:
        return default or []

def safe_read_csv(filename):
    path = get_camera_path(filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=["TIMESTAMP", "PERSON_ID", "VIOLATION", "CONFIDENCE", "DURATION", "IMAGE_PATH"])
    try:
        df = pd.read_csv(path)
        expected = ["TIMESTAMP", "PERSON_ID", "VIOLATION", "CONFIDENCE", "DURATION", "IMAGE_PATH"]
        for col in expected:
            if col not in df.columns:
                df[col] = None
        return df
    except:
        return pd.DataFrame(columns=["TIMESTAMP", "PERSON_ID", "VIOLATION", "CONFIDENCE", "DURATION", "IMAGE_PATH"])
    
#st_autorefresh(interval=2500, limit=1000, key="dashboard_refresh")
# ====================== PAGES ======================
if page == "Dashboard":
    st.header("Live Site Overview")
    st_autorefresh(interval=2500, limit=None, key="data_refresh")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Live Camera Feed")
        live_path = get_camera_path("live_frame.jpg")
        
        if os.path.exists(live_path) and os.path.getsize(live_path) > 5000:
            try:
                with open(live_path, "rb") as img_file:
                    img_bytes = img_file.read()
                if img_bytes:
                    st.image(img_bytes, width="stretch",
                             caption=f"🔴 LIVE — {camera_id} • {datetime.now().strftime('%H:%M:%S')}")
                else:
                    st.warning("Frame is empty")
            except Exception as e:
                st.warning(f"Read error: {e}")
        else:
            st.info("📷 **Waiting for live feed...**\n\n"
                    "1. Click **🚀 Start AI Core Pipeline** in sidebar\n"
                    "2. Wait 3-5 seconds\n"
                    "3. Refresh this page")

    with col2:
        st.subheader("Current Site Status")
        data = safe_read_json("Site_Safety.json", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Score", data.get("current_site_risk_score", 0))
        c2.metric("Site Level", data.get("current_site_risk_level", "SAFE"))
        c3.metric("Total Incidents", data.get("total_incidents", 0))
        c4.metric("Active Workers Violating", data.get("active_workers_in_violation", 0))

elif page == "Executive Summary":
    st.header("📊 Executive Safety Dashboard")
    st_autorefresh(interval=30000, limit=None, key="exec_refresh")  # refresh every 30s
 
    from database import get_stats, get_incidents, get_workers
    import pandas as pd
    from datetime import datetime, timedelta
 
    stats   = get_stats(camera_id)
    workers = get_workers(camera_id)
    incidents = get_incidents(camera_id, limit=500)
 
    # ── TOP KPI ROW ──────────────────────────────────────────────────────────
    st.subheader("Site Overview")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Incidents",       stats["total_incidents"])
    k2.metric("Workers Tracked",       stats["total_workers_tracked"])
    k3.metric("High Risk Workers",     stats["high_risk_workers"])
    k4.metric("Top Violation",         stats["top_violation"])
    k5.metric("Historical Risk Score", stats["historical_risk_score"])
 
    st.markdown("---")
 
    if incidents:
        df = pd.DataFrame(incidents)
        df["created_at"] = pd.to_datetime(df["created_at"])
 
        col1, col2 = st.columns(2)
 
        # ── VIOLATION BREAKDOWN ───────────────────────────────────────────────
        with col1:
            st.subheader("Violations by Type")
            violation_counts = df["violation_type"].value_counts().reset_index()
            violation_counts.columns = ["Violation", "Count"]
            st.bar_chart(violation_counts.set_index("Violation"))
 
        # ── RISK LEVEL BREAKDOWN ─────────────────────────────────────────────
        with col2:
            st.subheader("Incidents by Risk Level")
            risk_counts = df["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            st.bar_chart(risk_counts.set_index("Risk Level"))
 
        st.markdown("---")
 
        # ── DAILY TREND ───────────────────────────────────────────────────────
        st.subheader("Daily Incident Trend")
        df["date"] = df["created_at"].dt.date
        daily = df.groupby("date").size().reset_index(name="Incidents")
        daily = daily.sort_values("date")
        st.line_chart(daily.set_index("date"))
 
        st.markdown("---")
 
        col3, col4 = st.columns(2)
 
        # ── TOP VIOLATORS ────────────────────────────────────────────────────
        with col3:
            st.subheader("Top Risk Workers")
            if workers:
                top_workers = sorted(workers, key=lambda x: x["total_risk_score"], reverse=True)[:5]
                for w in top_workers:
                    color = "🔴" if w["risk_level"] == "HIGH" else "🟡" if w["risk_level"] == "MEDIUM" else "🟢"
                    st.write(f"{color} **{w['person_id']}** — Score: {w['total_risk_score']} | {w['total_incidents']} incidents")
            else:
                st.info("No worker data yet")
 
        # ── HOURLY HEATMAP ───────────────────────────────────────────────────
        with col4:
            st.subheader("Violations by Hour")
            df["hour"] = df["created_at"].dt.hour
            hourly = df.groupby("hour").size().reset_index(name="Count")
            hourly["Hour"] = hourly["hour"].apply(lambda x: f"{x:02d}:00")
            st.bar_chart(hourly.set_index("Hour")["Count"])
 
        st.markdown("---")
 
        # ── RECENT INCIDENTS TABLE ───────────────────────────────────────────
        st.subheader("Recent Incidents")
        recent = df[["created_at", "person_id", "violation_type", "risk_level", "risk_score", "duration_seconds"]].head(10)
        recent.columns = ["Time", "Worker", "Violation", "Risk", "Score", "Duration(s)"]
        recent["Duration(s)"] = recent["Duration(s)"].round(1)
        st.dataframe(recent, use_container_width=True)
 
    else:
        st.info("📭 No incident data yet. Run the pipeline to generate data.")

# Natural Language Query page for app.py
# Add "Ask AI" to navigation
# Requires: from Incident_Analysis_v2 import GROQ_report

elif page == "Ask AI":
    st.header("🤖 AI Safety Assistant")
    st.caption("Ask questions about violations, regulations, or worker safety")

    # ── EXAMPLE QUERIES ──────────────────────────────────────────────────────
    st.subheader("Try asking:")
    ex1, ex2, ex3 = st.columns(3)
    if ex1.button("Who violated most this week?"):
        st.session_state.nl_query = "Who violated safety rules most this week?"
    if ex2.button("What PPE is needed near welding?"):
        st.session_state.nl_query = "What PPE equipment should workers wear near welding areas?"
    if ex3.button("Summarize today's incidents"):
        st.session_state.nl_query = "Summarize all safety incidents that occurred today"

    # ── QUERY INPUT ───────────────────────────────────────────────────────────
    query = st.text_input(
        "Your question:",
        value=st.session_state.get("nl_query", ""),
        placeholder="e.g. Show all helmet violations from yesterday"
    )

    if st.button("Ask", type="primary") and query:
        with st.spinner("Thinking..."):
            try:
                from database import get_incidents, get_workers, get_stats
                from groq import Groq
                import os

                # Build context from database
                stats     = get_stats(camera_id)
                incidents = get_incidents(camera_id, limit=100)
                workers   = get_workers(camera_id)

                # Summarise data for context
                violation_summary = {}
                for inc in incidents:
                    v = inc["violation_type"]
                    violation_summary[v] = violation_summary.get(v, 0) + 1

                worker_summary = [
                    f"{w['person_id']}: {w['total_incidents']} incidents, score {w['total_risk_score']}, risk {w['risk_level']}"
                    for w in sorted(workers, key=lambda x: x["total_risk_score"], reverse=True)[:5]
                ]

                context = f"""
SITE DATA FOR {camera_id}:
Total Incidents: {stats['total_incidents']}
Total Workers: {stats['total_workers_tracked']}
High Risk Workers: {stats['high_risk_workers']}
Top Violation: {stats['top_violation']}

VIOLATION BREAKDOWN:
{chr(10).join(f"  {k}: {v} incidents" for k, v in violation_summary.items())}

TOP WORKERS BY RISK:
{chr(10).join(worker_summary) if worker_summary else "No worker data"}

RECENT INCIDENTS (last 10):
{chr(10).join(f"  [{i['created_at']}] {i['person_id']} — {i['violation_type']} — {i['risk_level']} risk" for i in incidents[:10])}
"""

                client = Groq(api_key=os.environ.get("GROQ_API_AI_SAFETY_REPORT"))
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an AI construction safety analyst. "
                                "Answer questions about site safety data concisely and professionally. "
                                "Use the provided site data to give specific, data-driven answers. "
                                "If asked about regulations, cite OSHA or Indian standards."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"{context}\n\nQUESTION: {query}"
                        }
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                answer = response.choices[0].message.content
                st.success("**AI Answer:**")
                st.write(answer)

                # Save to query history
                if "query_history" not in st.session_state:
                    st.session_state.query_history = []
                st.session_state.query_history.insert(0, {"q": query, "a": answer})

            except Exception as e:
                st.error(f"Query failed: {e}")

    # ── QUERY HISTORY ─────────────────────────────────────────────────────────
    if st.session_state.get("query_history"):
        st.markdown("---")
        st.subheader("Previous Questions")
        for item in st.session_state.query_history[:5]:
            with st.expander(f"Q: {item['q'][:60]}..."):
                st.write(item["a"])

elif page == "Incidents":
    st.header("Incident Records Ledger")
    df = safe_read_csv("violation_log.csv")
    if df.empty:
        st.info("📭 No incidents recorded yet.")
    else:
        if "TIMESTAMP" in df.columns and len(df) > 0:
            df = df.sort_values(by="TIMESTAMP", ascending=False)
        else:
            df = df.sort_index(ascending=False)
        st.dataframe(df, width="stretch", height=650)

# ─────────────────────────────────────────────────────────────────
# WORKERS PAGE
# ─────────────────────────────────────────────────────────────────

elif page == "Workers":
    st.header("👷 Worker Risk Analytics")
    try:
        workers = get_workers(camera_id)
    except Exception as e:
        st.info("📭 No worker data yet. Start the pipeline to begin tracking.")
        st.stop()

    if not workers:
        st.info("📭 No worker history yet. Run the pipeline on a video.")
    else:
        # Summary row
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Workers Tracked", len(workers))
        col2.metric("High Risk Workers",
                    sum(1 for w in workers if w["risk_level"] == "HIGH"))
        col3.metric("Total Violations",
                    sum(w["total_incidents"] for w in workers))

        st.markdown("---")

        for worker in workers:
            violations_list = json.loads(worker.get("violations_json", "[]"))
            with st.expander(
                f"👷 {worker['person_id']} | "
                f"Risk: {worker['risk_level']} | "
                f"Score: {worker['total_risk_score']}"
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Risk Score",    worker["total_risk_score"])
                    st.metric("Risk Level",    worker["risk_level"])
                    st.metric("Total Incidents", worker["total_incidents"])
                with c2:
                    st.write("**Violations:**", ", ".join(violations_list))
                    st.write("**Unique Violation Types:**",
                             worker["unique_violation_count"])
                    st.write("**Last Seen:**", worker.get("last_seen", "N/A"))

# ─────────────────────────────────────────────────────────────────
# REPORTS PAGE
# ─────────────────────────────────────────────────────────────────
elif page == "Reports":
    st.header("📋 AI Safety Reports (Groq Analysis)")
    try:
        incidents = get_incidents(camera_id, limit=50)
    except Exception as e:
        st.info("📭 No reports yet. Start the pipeline to generate incident reports.")
        st.stop()

    if not incidents:
        st.info("📭 No reports yet.")
    else:
        # Stats row
        stats = get_stats(camera_id)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Incidents",        stats["total_incidents"])
        c2.metric("Historical Risk Score",  stats["historical_risk_score"])
        c3.metric("Workers Tracked",        stats["total_workers_tracked"])
        c4.metric("Top Violation",          stats["top_violation"])

        st.markdown("---")

        # Filter controls
        violation_types = list({i["violation_type"] for i in incidents})
        selected_type = st.selectbox(
            "Filter by Violation", ["All"] + violation_types
        )

        filtered = (
            incidents if selected_type == "All"
            else [i for i in incidents if i["violation_type"] == selected_type]
        )

        for incident in filtered[:15]:
            risk_color = (
                "🔴" if incident["risk_level"] == "HIGH"
                else "🟡" if incident["risk_level"] == "MEDIUM"
                else "🟢"
            )
            st.subheader(
                f"{risk_color} {incident['violation_type']} — "
                f"Person-{incident['person_id']}"
            )
            st.caption(
                f"⏱ {incident['start_time']} | "
                f"Duration: {incident['duration_seconds']:.1f}s | "
                f"Risk Score: {incident['risk_score']}"
            )
            if incident.get("groq_analysis"):
                st.info(incident["groq_analysis"])

            img_path = incident.get("image_path", "")
            if img_path and os.path.exists(str(img_path)):
                st.image(img_path, width=400)

            st.markdown("---")

elif page == "Messages":
    st.header("⚠️ Alert History")
    try:
        alerts = get_alerts(camera_id, limit=50)
    except Exception as e:
        st.info("📭 No alerts yet. Start the pipeline to begin monitoring.")
        st.stop()

    if not alerts:
        st.info("No alerts yet.")
    else:
        st.metric("Total Alerts", len(alerts))
        st.markdown("---")
        for alert in alerts:
            if alert["alert_level"] == 3:
                st.error(
                    f"🔴 ESCALATION | {alert['timestamp']} | {alert['message']}"
                )
            elif alert["alert_level"] == 2:
                st.warning(
                    f"⚠️ REMINDER | {alert['timestamp']} | {alert['message']}"
                )
            else:
                st.info(
                    f"🚨 ALERT | {alert['timestamp']} | {alert['message']}"
                )

elif page == "Site Safety":
    st.header("Detailed Site Safety Matrix")
    data = safe_read_json("Site_Safety.json", {})
    if not data:
        st.info("📭 No site safety data yet.")
    else:
        st.json(data, expanded=True)



# ====================== CAMERA MANAGEMENT PAGE ======================
elif page == "Camera Management":
    st.header("🔧 Camera Management")
    st.markdown("Manage all cameras. Changes are saved to `cameras.json`")

    cameras = load_cameras()

    st.subheader("Registered Cameras")
    if not cameras:
        st.info("No cameras registered yet.")
    else:
        for i, cam in enumerate(cameras):
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.write(f"**{cam['camera_id']}** — {cam['name']} | 📍 {cam['location']}")
            with col2:
                if st.button("✏️ Edit", key=f"edit_{i}"):
                    st.session_state.editing_camera = i
            with col3:
                if st.button("🗑️ Delete", key=f"del_{i}"):
                    st.session_state.delete_confirm = i

    # Handle Delete Confirmation
    if 'delete_confirm' in st.session_state:
        i = st.session_state.delete_confirm
        if st.checkbox(f"Confirm delete {cameras[i]['camera_id']}?", key="confirm_del"):
            deleted_cam = cameras.pop(i)
            save_cameras(cameras)
            st.success(f"Deleted {deleted_cam['camera_id']}")
            del st.session_state.delete_confirm
            st.rerun()
        else:
            if st.button("Cancel"):
                del st.session_state.delete_confirm
                st.rerun()

    # Edit Form
    if 'editing_camera' in st.session_state:
        i = st.session_state.editing_camera
        cam = cameras[i]
        st.subheader(f"Editing {cam['camera_id']}")
        with st.form("edit_form"):
            cam["name"] = st.text_input("Name", cam["name"])
            cam["location"] = st.text_input("Location", cam["location"])
            cam["source"] = st.text_input("Source", str(cam.get("source", 0)))
            cam["description"] = st.text_area("Description", cam.get("description", ""))
            cam["status"] = st.selectbox("Status", ["ONLINE", "OFFLINE"], 
                                       index=0 if cam.get("status") == "ONLINE" else 1)
            
            if st.form_submit_button("💾 Save Changes"):
                save_cameras(cameras)
                st.success("Changes saved!")
                del st.session_state.editing_camera
                st.rerun()

    # Add New Camera
    st.subheader("➕ Add New Camera")
    with st.form("add_camera_form"):
        new_id = st.text_input("Camera ID", value=f"CAM_{len(cameras)+1:02d}")
        new_name = st.text_input("Camera Name", value="New Camera")
        new_location = st.text_input("Location", value="Construction Site")
        new_source = st.text_input("Source (0=webcam, RTSP URL)", value="0")
        new_desc = st.text_area("Description", "")

        if st.form_submit_button("Add Camera"):
            new_cam = {
                "camera_id": new_id,
                "name": new_name,
                "source": int(new_source) if new_source.isdigit() else new_source,
                "location": new_location,
                "status": "ONLINE",
                "description": new_desc
            }
            cameras.append(new_cam)
            save_cameras(cameras)
            st.success(f"✅ {new_id} added!")
            st.rerun()
st.markdown("---")
st.markdown("**AI Safety System** • YOLOv8 + Groq • Production Ready")


# if st.sidebar.button("🚀 Start AI Core Pipeline", type="primary", width="stretch"):
#     if 'PIPELINE_PROCESS' not in st.session_state or not st.session_state.PIPELINE_PROCESS:
#         try:
#             st.session_state.PIPELINE_PROCESS = subprocess.Popen(
#                 ["python", "s_p4.py", str(video_source), camera_id]
#             )
#             st.sidebar.success(f"✅ Pipeline started for **{camera_id}**")
#             st.sidebar.info("Check terminal for any camera errors")
#         except Exception as e:
#             st.sidebar.error(f"Failed to start: {e}")
#     else:
#         st.sidebar.warning("Pipeline already running")



# elif page == "Workers":
#     st.header("👷 Worker Risk Analytics Profile Ledger")
#     data = safe_read_json_list("worker_history.json", [])
#     if not data:
#         st.info("📭 No worker history recorded yet.")
#     else:
#         for worker in data:
#             with st.expander(f"👷 Person-{worker.get('person_id', 'N/A')} | Risk: {worker.get('risk_score', 0)}"):
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.metric("Risk Score", worker.get("risk_score", 0))
#                     st.metric("Risk Level", worker.get("risk_level", "LOW"))
#                 with col2:
#                     st.write("**Violations:**", ", ".join(worker.get("violations", [])))
#                     st.write("**Unique Violations:**", worker.get("unique_violation_count", 0))
#                 if worker.get("incidents"):
#                     st.write("**Recent Incidents:**")
#                     for inc in worker["incidents"][-3:]:
#                         st.caption(f"{inc.get('violation_type')} - {inc.get('start_time')}")
#                 st.json(worker, expanded=False)



# elif page == "Reports":
#     st.header("AI Safety Reports (Groq Analysis)")
#     incidents = safe_read_json_list("completed_incidents_analysis.json", [])
#     if not incidents:
#         st.info("📭 No reports generated yet.")
#     else:
#         recent = incidents[-8:] if len(incidents) >= 8 else incidents
#         for incident in reversed(recent):
#             st.subheader(f"Incident • Person-{incident.get('person_id', 'N/A')}")
#             st.caption(f"{incident.get('start_time', '')} | {incident.get('violation_type', '')}")
#             st.info(incident.get("GROQ_analysis", "Analysis pending..."))
#             img_path = incident.get("Image_path", "")
#             if img_path and os.path.exists(str(img_path)):
#                 st.image(img_path, width="stretch")
#             st.markdown("---")

# elif page == "Messages":
#     st.header("⚠️ Alert History")
#     alerts_path = get_camera_path("alerts.json")
#     if os.path.exists(alerts_path):
#         try:
#             with open(alerts_path, "r") as f:
#                 alerts = json.load(f)
#             if not alerts:
#                 st.info("No alerts yet.")
#             else:
#                 for alert in reversed(alerts[-15:]):
#                     if alert.get("alert_level") == 3:
#                         st.error(f"🔴 {alert['timestamp']} | {alert['message']}")
#                     elif alert.get("alert_level") == 2:
#                         st.warning(f"⚠️ {alert['timestamp']} | {alert['message']}")
#                     else:
#                         st.info(f"🚨 {alert['timestamp']} | {alert['message']}")
#         except:
#             st.info("No alerts yet.")
#     else:
#         st.info("No alerts recorded yet.")
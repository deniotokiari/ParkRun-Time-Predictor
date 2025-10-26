#!/usr/bin/env python3
"""
🏃‍♂️ ParkRun Time Predictor

AI-powered ParkRun finish time prediction with Game Boy design.
Uses machine learning to predict your finish time based on position.

Features:
- Neural network trained on historical ParkRun data
- Game Boy retro aesthetic with pixel-perfect design
- Position-based predictions with seasonal analysis
- Mobile-friendly responsive design

Data Source: ParkRun Krakow historical results
ML Model: TensorFlow/Keras neural network
Auto-Retraining: Weekly updates via GitHub Actions

Author: @deniotokiari
"""

import streamlit as st
import os
import warnings
import time

# Constants
PARKRUN_URL = "https://www.parkrun.pl/krakow/"
EMAIL = "deniotokiari@gmail.com"
MAX_POSITION = 1000
MIN_POSITION = 1
PREDICTION_TRIGGER_PROGRESS = 70
ANIMATION_STEPS = 101
ANIMATION_DELAY = 0.02

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ParkRun Predictor",
    page_icon="🏃‍♂️",
    layout="centered"
)

# CSS and JavaScript are now loaded from external files
# Fallback CSS for when external files are not available
CSS_STYLES = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
    .stApp { background: #9bbc0f; font-family: 'Press Start 2P', monospace; }
    header[data-testid="stHeader"] { display: none; }
    footer { display: none; }
    #MainMenu { visibility: hidden; }
    * { font-family: 'Press Start 2P', monospace !important; }
</style>
"""

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def format_pace(pace_min_per_km: float) -> str:
    """Format pace as MM:SS."""
    minutes = int(pace_min_per_km)
    seconds = int((pace_min_per_km - minutes) * 60)
    return f"{minutes}:{seconds:02d}"


def render_header() -> None:
    """Render the main header with ParkRun link."""
    st.markdown(f'<div class="main-header"><a href="{PARKRUN_URL}" style="color: #0f380f; text-decoration: none;">PARKRUN</a> PREDICTOR</div>', unsafe_allow_html=True)

def render_description() -> None:
    """Render the app description."""
    st.markdown(f"""
    <div style="text-align: center; color: #0f380f; margin-bottom: 2rem; font-size: 0.6rem; font-family: 'Press Start 2P', monospace;">
        PREDICT YOUR <a href="{PARKRUN_URL}" style="color: #0f380f; text-decoration: none;">PARKRUN</a> FINISH TIME<br>
        BASED ON YOUR EXPECTED POSITION
    </div>
    """, unsafe_allow_html=True)

def get_user_input() -> tuple[int, bool]:
    """Get and validate user input."""
    position_text = st.text_input("ENTER POSITION", value="100", placeholder="ENTER POSITION", max_chars=4, label_visibility="hidden")
    
    position = None
    is_valid = False
    
    if position_text:
        try:
            position = int(position_text)
            if MIN_POSITION <= position <= MAX_POSITION:
                is_valid = True
            else:
                st.error(f"POSITION MUST BE BETWEEN {MIN_POSITION} AND {MAX_POSITION}")
        except ValueError:
            st.error("ENTER A VALID NUMBER")
    else:
        st.error("ENTER A POSITION")
    
    return position, is_valid

def render_progress_bar(progress: int) -> str:
    """Render progress bar HTML."""
    return f"""
    <div style="
        background-color: #9bbc0f;
        border: 3px solid #0f380f;
        border-radius: 0;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 3px 3px 0 #0f380f;
    ">
        <div style="
            background-color: #0f380f;
            height: 20px;
            border: 2px solid #0f380f;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                background-color: #8bac0f;
                height: 100%;
                width: {progress}%;
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
            ">
            </div>
        </div>
    </div>
    """

def get_status_text(step: int) -> str:
    """Get status text based on animation step."""
    total_steps = ANIMATION_STEPS
    if step < total_steps * 0.25:
        return "INITIALIZING..."
    elif step < total_steps * 0.5:
        return "ANALYZING DATA..."
    elif step < total_steps * 0.75:
        return "CALCULATING..."
    else:
        return "FINALIZING..."

def render_status_text(status: str) -> str:
    """Render status text HTML."""
    return f"""
    <div style="
        text-align: center;
        margin: 10px 0;
        color: #0f380f;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 10px;
    ">{status}</div>
    """

def run_prediction_animation(position: int, month: int = None) -> dict:
    """Run the prediction animation with separate pipeline steps."""
    from park_run_speed_predict import ParkRunPredictor
    
    progress_container = st.empty()
    status_container = st.empty()
    
    predictor = None
    result = None
    
    for i in range(ANIMATION_STEPS):
        progress_container.markdown(render_progress_bar(i), unsafe_allow_html=True)
        
        status_text = get_status_text(i)
        status_container.markdown(render_status_text(status_text), unsafe_allow_html=True)
        
        # Run full pipeline at the start of "ANALYZING DATA" phase
        if status_text == "ANALYZING DATA..." and predictor is None:
            predictor = ParkRunPredictor()
            predictor.run_full_pipeline()
        
        # Run prediction at the start of "CALCULATING" phase
        elif status_text == "CALCULATING..." and predictor is not None and result is None:
            result = predictor.predict(position, month)
            # Convert result to expected format
            result = {
                "success": True,
                "time_seconds": float(result["time_seconds"]),
                "time_minutes": float(result["time_minutes"]),
                "pace_min_per_km": float(result["pace_min_per_km"]),
                "position": int(result["position"]),
                "month": int(result["month"]),
                "participants": float(result["participants"])
            }
        
        time.sleep(ANIMATION_DELAY)
    
    progress_container.empty()
    status_container.empty()
    
    return result

def render_results(result: dict) -> None:
    """Render prediction results."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">TIME</div>
            <div class="value">{format_time(result['time_seconds'])}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">PACE</div>
            <div class="value">{format_pace(result['pace_min_per_km'])}</div>
        </div>
        """, unsafe_allow_html=True)

def render_footer() -> None:
    """Render the footer."""
    st.markdown(f"""
    <div class="footer">
        <p><a href="{PARKRUN_URL}" style="color: #0f380f; text-decoration: none;">PARKRUN</a> PREDICTOR | POWERED BY <a href="mailto:{EMAIL}">@DENIOTOKIARI</a></p>
    </div>
    """, unsafe_allow_html=True)

def load_css() -> str:
    """Load CSS from external file."""
    try:
        with open('.streamlit/styles.css', 'r', encoding='utf-8') as f:
            return f"<style>{f.read()}</style>"
    except FileNotFoundError:
        # Fallback to embedded CSS if file not found
        return CSS_STYLES

def main():
    """Main application function."""
    st.markdown(load_css(), unsafe_allow_html=True)
    
    render_header()
    render_description()

    position, is_valid = get_user_input()
    month = None
    
    if is_valid:
        if st.button("PREDICT TIME", type="primary", use_container_width=True):
            try:
                result = run_prediction_animation(position, month)
                render_results(result)
                
            except Exception as e:
                st.error(f"ERROR: {e}")
                st.info("CHECK INPUT VALUES AND TRY AGAIN")
    else:
        st.button("PREDICT TIME", disabled=True, use_container_width=True)
        st.info("ENTER A VALID POSITION TO ENABLE PREDICTION")
    
    render_footer()

if __name__ == "__main__":
    main()
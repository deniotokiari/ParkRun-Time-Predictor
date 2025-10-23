#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ParkRun Predictor",
    page_icon="🏃‍♂️",
    layout="centered"
)

CSS_STYLES = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
    
    .stApp {
        background: #9bbc0f;
        font-family: 'Press Start 2P', monospace;
        image-rendering: pixelated;
    }
    
    * {
        font-family: 'Press Start 2P', monospace !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, div, span, label, input, button, textarea, select {
        font-family: 'Press Start 2P', monospace !important;
    }
    
    .stMarkdown, .stText, .stAlert, .stSuccess, .stError, .stWarning, .stInfo {
        font-family: 'Press Start 2P', monospace !important;
    }
    
    .stApp > div, .stApp > div > div, .stApp > div > div > div {
        font-family: 'Press Start 2P', monospace !important;
    }
    
    .main-header {
        font-size: 2.0rem !important;
        color: #0f380f !important;
        text-shadow: 2px 2px 0px #306230 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        line-height: 1.2 !important;
        font-weight: bold !important;
        display: block !important;
    }
    
    .metric-box {
        background: #8bac0f;
        border: 4px solid #0f380f;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 4px 4px 0px #0f380f;
    }
    
    .metric-box:hover {
        background: #9bbc0f;
        box-shadow: 6px 6px 0px #0f380f;
    }
    
    .metric-box .label {
        color: #0f380f;
        font-size: 0.6rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        line-height: 1.2;
        font-weight: bold;
    }
    
    .metric-box .value {
        color: #0f380f;
        font-size: 1.8rem;
        margin: 0;
        line-height: 1.2;
        font-weight: bold;
    }
    
    .stNumberInput > label {
        color: #0f380f !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
    }
    
    .stNumberInput > div {
        background-color: transparent;
    }
    
    .stNumberInput > div > div > input {
        background-color: #8bac0f;
        border: 3px solid #0f380f;
        border-radius: 0;
        color: #0f380f;
        font-family: 'Press Start 2P', monospace;
        font-size: 0.7rem;
        padding: 0.5rem;
        box-shadow: 3px 3px 0px #0f380f;
    }
    
    .stNumberInput > div > div > input:focus {
        background-color: #9bbc0f;
        border: 3px solid #306230;
        box-shadow: 3px 3px 0px #306230;
        outline: none;
    }
    
    .stNumberInput button {
        background-color: #306230 !important;
        border: 3px solid #0f380f !important;
        border-radius: 0 !important;
        color: #9bbc0f !important;
        font-family: 'Press Start 2P', monospace !important;
        box-shadow: 2px 2px 0px #0f380f !important;
    }
    
    .stNumberInput button:hover {
        background-color: #0f380f !important;
        color: #9bbc0f !important;
        box-shadow: 3px 3px 0px #306230 !important;
    }
    
    .stNumberInput button:focus {
        outline: none !important;
        box-shadow: 2px 2px 0px #0f380f !important;
    }
    
    .stTextInput > div > div > input {
        background-color: #8bac0f !important;
        border: 3px solid #0f380f !important;
        border-radius: 0 !important;
        color: #0f380f !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 0.7rem !important;
        padding: 0.5rem !important;
        box-shadow: 3px 3px 0px #0f380f !important;
        text-align: center !important;
        outline: none !important;
        -webkit-appearance: none !important;
        -moz-appearance: none !important;
        appearance: none !important;
    }
    
    .stTextInput > div > div > input:focus {
        background-color: #9bbc0f !important;
        border: 3px solid #306230 !important;
        box-shadow: 3px 3px 0px #306230 !important;
        outline: none !important;
        -webkit-appearance: none !important;
        -moz-appearance: none !important;
        appearance: none !important;
    }
    
    .stTextInput > div > div > input::-webkit-input-placeholder {
        color: #306230 !important;
    }
    
    .stTextInput > div > div > input::-moz-placeholder {
        color: #306230 !important;
    }
    
    .stTextInput > div > div > input:-ms-input-placeholder {
        color: #306230 !important;
    }
    
    .stTextInput > div {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    
    .stTextInput > div > div {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #306230 !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 0.6rem !important;
        text-align: center !important;
    }
    
    .stTextInput > div > div > input::-webkit-outer-spin-button,
    .stTextInput > div > div > input::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    
    .stTextInput > div > div > input[type=number] {
        -moz-appearance: textfield;
    }
    
    .stTextInput,
    .stTextInput > div,
    .stTextInput > div > div,
    .stTextInput > div > div > div {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        outline: none !important;
    }
    
    .stTextInput > div > div {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        outline: none !important;
        -webkit-box-shadow: none !important;
        -moz-box-shadow: none !important;
    }
    
    .stButton > button {
        background: #306230;
        color: #9bbc0f;
        border: 3px solid #0f380f;
        border-radius: 0;
        font-family: 'Press Start 2P', monospace;
        font-size: 0.7rem;
        padding: 1rem 2rem;
        box-shadow: 4px 4px 0px #0f380f;
    }
    
    .stButton > button:hover {
        background: #0f380f;
        color: #9bbc0f;
        box-shadow: 6px 6px 0px #306230;
    }
    
    .stMarkdown h3 {
        color: #0f380f;
        font-family: 'Press Start 2P', monospace;
        font-size: 0.8rem;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    
    .stMarkdown div {
        color: #0f380f;
        font-family: 'Press Start 2P', monospace;
        font-size: 0.6rem;
        line-height: 1.4;
    }
    
    .footer {
        text-align: center;
        color: #306230;
        padding: 2rem 1rem;
        font-size: 0.5rem;
        font-family: 'Press Start 2P', monospace;
    }
    
    .footer a {
        color: #0f380f;
        text-decoration: none;
    }
    
    .footer a:hover {
        color: #306230;
    }
    
    .stSpinner > div {
        border-color: #0f380f transparent #0f380f transparent;
    }
    
    .stAlert {
        border-radius: 0;
        border: 3px solid #0f380f;
        background: #8bac0f;
        color: #0f380f;
        font-family: 'Press Start 2P', monospace;
        font-size: 0.6rem;
    }
    
    .stSuccess {
        border-radius: 0;
        border: 3px solid #0f380f;
        background: #8bac0f;
        color: #0f380f;
        font-family: 'Press Start 2P', monospace;
        font-size: 0.6rem;
    }
    
    .stInfo {
        border-radius: 0;
        border: 3px solid #306230;
        background: #9bbc0f;
        color: #0f380f;
        font-family: 'Press Start 2P', monospace;
        font-size: 0.6rem;
    }
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
    const inputs = document.querySelectorAll('.stTextInput input');
    inputs.forEach(input => {
        input.addEventListener('input', function(e) {
            this.value = this.value.replace(/[^0-9]/g, '');
        });
        
        input.addEventListener('keypress', function(e) {
            if (!/[0-9]/.test(e.key) && !['Backspace', 'Delete', 'Tab', 'Escape', 'Enter'].includes(e.key)) {
                e.preventDefault();
            }
        });
    });
});
</script>
"""

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def format_pace(pace_min_per_km):
    minutes = int(pace_min_per_km)
    seconds = int((pace_min_per_km - minutes) * 60)
    return f"{minutes}:{seconds:02d}"

def predict_direct(position, month=None):
    from park_run_speed_predict import ParkRunPredictor
    
    predictor = ParkRunPredictor()
    predictor.run_full_pipeline()
    
    result = predictor.predict(position, month)
    
    return {
        "success": True,
        "time_seconds": float(result["time_seconds"]),
        "time_minutes": float(result["time_minutes"]),
        "pace_min_per_km": float(result["pace_min_per_km"]),
        "position": int(result["position"]),
        "month": int(result["month"]),
        "participants": float(result["participants"])
    }

def main():
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">PARKRUN PREDICTOR</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #0f380f; margin-bottom: 2rem; font-size: 0.6rem; font-family: 'Press Start 2P', monospace;">
        PREDICT YOUR PARKRUN FINISH TIME<br>
        BASED ON YOUR EXPECTED POSITION
    </div>
    """, unsafe_allow_html=True)
    
    position_text = st.text_input("ENTER POSITION", value="100", placeholder="ENTER POSITION", max_chars=4)
    
    position = None
    is_valid = False
    
    if position_text:
        try:
            position = int(position_text)
            if 1 <= position <= 1000:
                is_valid = True
            else:
                st.error("POSITION MUST BE BETWEEN 1 AND 1000")
        except ValueError:
            st.error("ENTER A VALID NUMBER")
    else:
        st.error("ENTER A POSITION")
    
    month = None
    
    if is_valid:
        if st.button("PREDICT TIME", type="primary", use_container_width=True):
            try:
                progress_container = st.empty()
                status_container = st.empty()
                
                result = None
                
                for i in range(101):
                    progress_container.markdown(f"""
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
                                width: {i}%;
                                position: relative;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                            ">
                                <div style="
                                    color: #0f380f;
                                    font-family: 'Press Start 2P', monospace !important;
                                    font-size: 8px;
                                    font-weight: bold;
                                ">{i}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if i < 30:
                        status_text = "INITIALIZING..."
                    elif i < 60:
                        status_text = "ANALYZING DATA..."
                    elif i < 90:
                        status_text = "CALCULATING..."
                        if i == 70 and result is None:
                            result = predict_direct(position, month)
                    else:
                        status_text = "FINALIZING..."
                    
                    status_container.markdown(f"""
                    <div style="
                        text-align: center;
                        margin: 10px 0;
                        color: #0f380f;
                        font-family: 'Press Start 2P', monospace !important;
                        font-size: 10px;
                    ">{status_text}</div>
                    """, unsafe_allow_html=True)
                    
                    time.sleep(0.02)
                
                progress_container.empty()
                status_container.empty()
                
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
                
            except Exception as e:
                st.error(f"ERROR: {e}")
                st.info("CHECK INPUT VALUES AND TRY AGAIN")
    else:
        st.button("PREDICT TIME", disabled=True, use_container_width=True)
        st.info("ENTER A VALID POSITION TO ENABLE PREDICTION")
    
    st.markdown("""
    <div class="footer">
        <p>PARKRUN PREDICTOR | POWERED BY <a href="mailto:deniotokiari@gmail.com">@DENIOTOKIARI</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
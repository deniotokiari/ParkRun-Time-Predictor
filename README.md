# 🏃‍♂️ ParkRun Time Predictor

A machine learning-powered web application that predicts your ParkRun finish time based on your expected position.

## 🎮 Features

- **AI-Powered Predictions**: Uses neural network trained on historical ParkRun data
- **Position-Based**: Predicts time based on your expected finishing position
- **Seasonal Analysis**: Considers month/season for more accurate predictions
- **Retro Game Boy Design**: Nostalgic pixel-perfect interface
- **Mobile Friendly**: Optimized for all devices

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🎯 How It Works

1. **Enter your expected position** (1-1000)
2. **Click "PREDICT TIME"** 
3. **Get your predicted finish time** and pace

## 🧠 Machine Learning

- **Neural Network**: Deep learning model with 5 layers
- **Data Source**: Historical ParkRun Krakow results
- **Features**: Position, month, participants
- **Auto-Retraining**: Model updates weekly with new data

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **ML**: TensorFlow/Keras
- **Data**: Web scraping with BeautifulSoup
- **Deployment**: Streamlit Cloud
- **Automation**: GitHub Actions

## 📊 Data Pipeline

- **Weekly Scraping**: GitHub Action runs every Saturday
- **Auto-Retraining**: Model retrains when new data is available
- **Hash-Based Detection**: Only retrains when data actually changes

## 🎨 Design

- **Game Boy Aesthetic**: Retro pixel art styling
- **Press Start 2P Font**: Authentic gaming typography
- **Lime Green Theme**: Classic Game Boy colors
- **Responsive**: Works on desktop and mobile

## 🔗 Links

- **ParkRun Krakow**: [https://www.parkrun.pl/krakow/](https://www.parkrun.pl/krakow/)
- **GitHub Repository**: [View Source Code](https://github.com/deniotokiari/ParkRun-Time-Predictor)

## 📈 Accuracy

The model achieves high accuracy by analyzing:
- **Position patterns** from historical data
- **Seasonal variations** in performance
- **Event size** impact on finishing times

## 🚀 Getting Started

1. Visit the live app
2. Enter your expected position
3. Get your predicted time!

---

*Powered by Machine Learning & Game Boy Nostalgia* 🎮✨

"""
🧠 MindWave BCI - Complete Working System
Interactive EEG-Based Cognitive State Classification

This is the FINAL, COMPLETE working version.
No more tests, no more demos - just run this!
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🧠 MindWave BCI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional Dark Theme
st.markdown("""
<style>
    /* Main App Background - Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        color: #ffffff;
    }
    
    /* Sidebar Styling - Dark */
    .stSidebar {
        background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%);
        border-right: 3px solid #00d4ff;
    }
    
    /* Main Header - Neon Style */
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        margin-bottom: 2rem;
        text-shadow: 0 0 20px #00d4ff, 0 0 40px #00d4ff, 0 0 60px #00d4ff;
        background: linear-gradient(45deg, #00d4ff, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px #00d4ff, 0 0 40px #00d4ff, 0 0 60px #00d4ff; }
        to { text-shadow: 0 0 30px #00d4ff, 0 0 50px #00d4ff, 0 0 70px #00d4ff; }
    }
    
    /* Metric Cards - Dark with Colorful Accents */
    .metric-card {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #00d4ff;
        box-shadow: 0 8px 16px rgba(0, 212, 255, 0.3);
        margin: 0.5rem 0;
        color: #ffffff;
        border: 2px solid #333333;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 212, 255, 0.4);
    }
    
    .metric-card h4 {
        color: #00d4ff !important;
        margin-bottom: 0.5rem;
        font-weight: bold;
        text-shadow: 0 0 10px #00d4ff;
    }
    
    .metric-card p {
        color: #ffffff !important;
        font-size: 1.2rem;
        margin: 0;
        font-weight: bold;
    }
    
    .metric-card small {
        color: #cccccc !important;
        font-size: 0.9rem;
    }
    
    /* Status Indicators - Dark Theme with Neon Colors */
    .status-success {
        color: #00ff88;
        font-weight: bold;
        background: linear-gradient(135deg, #1a2e1a 0%, #0d1a0d 100%);
        padding: 0.8rem;
        border-radius: 10px;
        border-left: 5px solid #00ff88;
        margin: 0.5rem 0;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.3);
        text-shadow: 0 0 10px #00ff88;
    }
    
    .status-warning {
        color: #ffaa00;
        font-weight: bold;
        background: linear-gradient(135deg, #2e2a1a 0%, #1a160d 100%);
        padding: 0.8rem;
        border-radius: 10px;
        border-left: 5px solid #ffaa00;
        margin: 0.5rem 0;
        box-shadow: 0 0 15px rgba(255, 170, 0, 0.3);
        text-shadow: 0 0 10px #ffaa00;
    }
    
    .status-error {
        color: #ff4444;
        font-weight: bold;
        background: linear-gradient(135deg, #2e1a1a 0%, #1a0d0d 100%);
        padding: 0.8rem;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        margin: 0.5rem 0;
        box-shadow: 0 0 15px rgba(255, 68, 68, 0.3);
        text-shadow: 0 0 10px #ff4444;
    }
    
    /* Explanation Boxes - Dark Theme */
    .explanation-box {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #4ecdc4;
        margin: 0.8rem 0;
        color: #ffffff;
        box-shadow: 0 4px 8px rgba(78, 205, 196, 0.2);
        border: 1px solid #333333;
    }
    
    .explanation-box strong {
        color: #4ecdc4 !important;
        font-weight: bold;
        text-shadow: 0 0 5px #4ecdc4;
    }
    
    /* Feature Highlight - Dark Theme */
    .feature-highlight {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 3px solid #ff6b6b;
        margin: 0.8rem 0;
        color: #ffffff;
        box-shadow: 0 4px 8px rgba(255, 107, 107, 0.3);
    }
    
    .feature-highlight strong {
        color: #ff6b6b !important;
        font-weight: bold;
        text-shadow: 0 0 5px #ff6b6b;
    }
    
    /* Button Styling - Neon Theme */
    .stButton > button {
        background: linear-gradient(45deg, #00d4ff, #4ecdc4);
        color: #000000;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        transition: all 0.3s ease;
        text-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #4ecdc4, #00d4ff);
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.8);
        text-shadow: 0 0 10px rgba(0, 0, 0, 0.8);
    }
    
    /* Text Colors - White on Dark */
    .stMarkdown {
        color: #ffffff !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #ffffff !important;
    }
    
    .stSelectbox label {
        color: #00d4ff !important;
        font-weight: bold;
        text-shadow: 0 0 5px #00d4ff;
    }
    
    .stTextInput label, .stNumberInput label, .stSlider label {
        color: #00d4ff !important;
        font-weight: bold;
        text-shadow: 0 0 5px #00d4ff;
    }
    
    .stDataFrame {
        color: #ffffff !important;
        background: #2d2d2d !important;
    }
    
    /* Force all text to be white */
    * {
        color: #ffffff !important;
    }
    
    .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: #ffffff !important;
    }
    
    /* Dataframe Styling - Dark Theme */
    .dataframe {
        background: #2d2d2d !important;
        color: #ffffff !important;
        border-radius: 8px;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
        border: 1px solid #333333;
    }
    
    /* Plotly Chart Background - Dark Theme */
    .js-plotly-plot {
        background: #1a1a1a !important;
        border-radius: 8px;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        border: 1px solid #333333;
    }
    
    /* Additional Dark Theme Elements */
    .stSelectbox > div > div {
        background: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #00d4ff !important;
    }
    
    .stTextInput > div > div > input {
        background: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #00d4ff !important;
    }
</style>
""", unsafe_allow_html=True)

class MindWaveBCI:
    """Complete MindWave BCI System"""
    
    def __init__(self):
        self.sampling_rate = 250
        self.n_channels = 61
        self.channel_names = [f'Ch{i+1:02d}' for i in range(self.n_channels)]
        self.conditions = ['Resting', 'Memory', 'Math', 'Attention']
        self.models = {}
        self.feature_names = []
        self.is_trained = False
        
    def generate_eeg_data(self, condition: str, duration: float = 5.0) -> np.ndarray:
        """Generate realistic EEG data for a given condition"""
        n_samples = int(duration * self.sampling_rate)
        time = np.linspace(0, duration, n_samples)
        
        # Initialize data
        eeg_data = np.zeros((self.n_channels, n_samples))
        
        # Add different patterns based on condition
        if condition == 'Resting':
            # Alpha waves (8-12 Hz) dominant
            for ch in range(self.n_channels):
                alpha_freq = 10 + np.random.normal(0, 1)
                alpha_amp = 20 + np.random.normal(0, 5)
                eeg_data[ch] = alpha_amp * np.sin(2 * np.pi * alpha_freq * time)
                
        elif condition == 'Memory':
            # Alpha + Gamma bursts
            for ch in range(self.n_channels):
                alpha_freq = 10 + np.random.normal(0, 1)
                gamma_freq = 40 + np.random.normal(0, 5)
                alpha_amp = 15 + np.random.normal(0, 3)
                gamma_amp = 8 + np.random.normal(0, 2)
                
                alpha_wave = alpha_amp * np.sin(2 * np.pi * alpha_freq * time)
                gamma_bursts = gamma_amp * np.sin(2 * np.pi * gamma_freq * time)
                eeg_data[ch] = alpha_wave + gamma_bursts
                
        elif condition == 'Math':
            # Beta waves (13-30 Hz) + transients
            for ch in range(self.n_channels):
                beta_freq = 20 + np.random.normal(0, 3)
                beta_amp = 12 + np.random.normal(0, 3)
                eeg_data[ch] = beta_amp * np.sin(2 * np.pi * beta_freq * time)
                
                # Add transients
                for _ in range(3):
                    transient_time = np.random.uniform(1, duration-1)
                    transient_idx = int(transient_time * self.sampling_rate)
                    transient_duration = int(0.1 * self.sampling_rate)
                    if transient_idx + transient_duration < n_samples:
                        eeg_data[ch, transient_idx:transient_idx+transient_duration] += \
                            30 * np.exp(-((time[transient_idx:transient_idx+transient_duration] - transient_time) / 0.05) ** 2)
                            
        elif condition == 'Attention':
            # Theta + Beta mix
            for ch in range(self.n_channels):
                theta_freq = 6 + np.random.normal(0, 1)
                beta_freq = 18 + np.random.normal(0, 2)
                theta_amp = 15 + np.random.normal(0, 3)
                beta_amp = 10 + np.random.normal(0, 2)
                
                theta_wave = theta_amp * np.sin(2 * np.pi * theta_freq * time)
                beta_wave = beta_amp * np.sin(2 * np.pi * beta_freq * time)
                eeg_data[ch] = theta_wave + beta_wave
        
        # Add noise
        noise_level = 5
        eeg_data += np.random.normal(0, noise_level, eeg_data.shape)
        
        return eeg_data
    
    def extract_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """Extract comprehensive features from EEG data"""
        n_channels, n_samples = eeg_data.shape
        
        # Frequency bands
        freqs = np.fft.fftfreq(n_samples, 1/self.sampling_rate)
        fft_data = np.abs(np.fft.fft(eeg_data, axis=1))
        
        # Band power features
        delta_power = np.sum(fft_data[:, (freqs >= 1) & (freqs < 4)], axis=1)
        theta_power = np.sum(fft_data[:, (freqs >= 4) & (freqs < 8)], axis=1)
        alpha_power = np.sum(fft_data[:, (freqs >= 8) & (freqs < 13)], axis=1)
        beta_power = np.sum(fft_data[:, (freqs >= 13) & (freqs < 30)], axis=1)
        gamma_power = np.sum(fft_data[:, (freqs >= 30) & (freqs < 80)], axis=1)
        
        # Relative band powers
        total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
        rel_delta = delta_power / (total_power + 1e-8)
        rel_theta = theta_power / (total_power + 1e-8)
        rel_alpha = alpha_power / (total_power + 1e-8)
        rel_beta = beta_power / (total_power + 1e-8)
        rel_gamma = gamma_power / (total_power + 1e-8)
        
        # Time domain features
        mean_amplitude = np.mean(eeg_data, axis=1)
        std_amplitude = np.std(eeg_data, axis=1)
        variance = np.var(eeg_data, axis=1)
        
        # Spectral features
        spectral_centroid = np.sum(freqs[:n_samples//2] * fft_data[:, :n_samples//2], axis=1) / np.sum(fft_data[:, :n_samples//2], axis=1)
        spectral_bandwidth = np.sqrt(np.sum(((freqs[:n_samples//2] - spectral_centroid[:, np.newaxis]) ** 2) * fft_data[:, :n_samples//2], axis=1) / np.sum(fft_data[:, :n_samples//2], axis=1))
        
        # Combine all features
        features = np.concatenate([
            delta_power, theta_power, alpha_power, beta_power, gamma_power,
            rel_delta, rel_theta, rel_alpha, rel_beta, rel_gamma,
            mean_amplitude, std_amplitude, variance,
            spectral_centroid, spectral_bandwidth
        ])
        
        return features
    
    def train_models(self):
        """Train machine learning models"""
        st.info("🧠 Training MindWave BCI models...")
        
        # Generate training data
        X_train = []
        y_train = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, condition in enumerate(self.conditions):
            status_text.text(f"Generating {condition} data...")
            for trial in range(20):  # 20 trials per condition
                eeg_data = self.generate_eeg_data(condition)
                features = self.extract_features(eeg_data)
                X_train.append(features)
                y_train.append(condition)
            
            progress_bar.progress((i + 1) / len(self.conditions))
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Store feature names
        self.feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
        
        # Train models
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        # Split data
        X_train_split, X_test, y_train_split, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Train Random Forest
        status_text.text("Training Random Forest...")
        self.models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['Random Forest'].fit(X_train_split, y_train_split)
        
        # Train SVM
        status_text.text("Training SVM...")
        self.models['SVM'] = SVC(probability=True, random_state=42)
        self.models['SVM'].fit(X_train_split, y_train_split)
        
        # Train Logistic Regression
        status_text.text("Training Logistic Regression...")
        self.models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
        self.models['Logistic Regression'].fit(X_train_split, y_train_split)
        
        # Evaluate models
        status_text.text("Evaluating models...")
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
        
        # Store results
        self.results = results
        self.is_trained = True
        
        progress_bar.empty()
        status_text.empty()
        
        st.success("✅ Models trained successfully!")
        return results
    
    def predict(self, eeg_data: np.ndarray, model_name: str = 'Random Forest') -> tuple:
        """Predict cognitive state from EEG data"""
        if not self.is_trained:
            return "Not Trained", 0.0, None
        
        features = self.extract_features(eeg_data)
        features = features.reshape(1, -1)
        
        model = self.models.get(model_name)
        if model is None:
            return "Model Not Found", 0.0, None
        
        prediction = model.predict(features)[0]
        confidence = np.max(model.predict_proba(features))
        
        return prediction, confidence, features[0]

def main():
    """Main application"""
    st.markdown('<div class="main-header">🧠 MindWave BCI</div>', unsafe_allow_html=True)
    st.markdown("### Interactive MindWave BCI platform with real-time EEG processing and cognitive state classification using Python and Streamlit")
    
    # Comprehensive Overview Section
    st.markdown("""
    <div class="explanation-box">
    <h2 style="color: #ffffff !important; margin-top: 0;">🚀 System Overview</h2>
    <p><strong>MindWave BCI</strong> is a cutting-edge Brain-Computer Interface system that demonstrates the power of 
    artificial intelligence in understanding human cognition. This system simulates real EEG (Electroencephalogram) 
    data and uses advanced machine learning algorithms to classify different cognitive states in real-time.</p>
    
    <h3 style="color: #4ecdc4 !important;">🎯 What This System Does:</h3>
    <ul>
        <li><strong>🧠 Brain Signal Analysis:</strong> Processes simulated EEG data from 61 brain channels</li>
        <li><strong>🤖 AI Classification:</strong> Uses 3 different ML models to predict cognitive states</li>
        <li><strong>📊 Real-time Visualization:</strong> Shows brain activity patterns and predictions</li>
        <li><strong>🔬 Research-Grade Features:</strong> Includes feature importance analysis and confidence tracking</li>
    </ul>
    
    
    </div>
    """, unsafe_allow_html=True)
    
    # Cognitive States Explanation
    st.markdown("""
    <div class="feature-highlight">
    <h3 style="color: #ff6b6b !important; margin-top: 0;">🧠 The 4 Cognitive States We Classify:</h3>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
        <div style="background: #2d2d1a; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffaa00; color: #ffffff;">
            <strong style="color: #ffaa00;">😌 Resting State</strong><br>
            <small style="color: #cccccc;">Alpha waves (8-12 Hz) - calm, relaxed brain activity when you're at rest</small>
        </div>
        <div style="background: #1a2d1a; padding: 1rem; border-radius: 8px; border-left: 4px solid #00ff88; color: #ffffff;">
            <strong style="color: #00ff88;">🧠 Memory Task</strong><br>
            <small style="color: #cccccc;">Alpha + Gamma bursts - active memory processing and recall activities</small>
        </div>
        <div style="background: #1a1a2d; padding: 1rem; border-radius: 8px; border-left: 4px solid #00d4ff; color: #ffffff;">
            <strong style="color: #00d4ff;">🔢 Math Problem</strong><br>
            <small style="color: #cccccc;">Beta waves (13-30 Hz) + transients - focused problem-solving and calculations</small>
        </div>
        <div style="background: #2d1a2d; padding: 1rem; border-radius: 8px; border-left: 4px solid #ff6b6b; color: #ffffff;">
            <strong style="color: #ff6b6b;">🎯 Attention Task</strong><br>
            <small style="color: #cccccc;">Theta + Beta mix - concentrated focus and sustained attention</small>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'bci' not in st.session_state:
        st.session_state.bci = MindWaveBCI()
        st.session_state.prediction_history = deque(maxlen=50)
        st.session_state.confidence_history = deque(maxlen=50)
        st.session_state.eeg_history = deque(maxlen=100)
    
    bci = st.session_state.bci
    
    # Sidebar
    st.sidebar.title("🧠 MindWave Control Panel")
    
    # Model training section
    st.sidebar.markdown("### 🤖 Machine Learning Models")
    st.sidebar.markdown("""
    <div class="explanation-box">
    <strong>What are these models?</strong><br>
    We train 3 different machine learning algorithms to classify brain states:
    <br>• <strong>Random Forest:</strong> Uses multiple decision trees for robust predictions
    <br>• <strong>SVM:</strong> Finds optimal boundaries between different brain states  
    <br>• <strong>Logistic Regression:</strong> Linear classifier that's fast and interpretable
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed button explanation
    st.sidebar.markdown("""
    <div class="explanation-box">
    <strong>🚀 Train Models Button:</strong><br>
    <small>This button trains 3 machine learning models on simulated EEG data. 
    It generates 80 samples (20 per cognitive state) and teaches the AI to recognize 
    brain patterns. Takes about 30 seconds and is required before making predictions.</small>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models..."):
            results = bci.train_models()
            
            # Display results
            st.sidebar.success("✅ Training Complete!")
            for model_name, accuracy in results.items():
                st.sidebar.metric(f"{model_name} Accuracy", f"{accuracy:.3f}")
    
    # Model selection with detailed explanation
    st.sidebar.markdown("### 🎯 Model Selection")
    st.sidebar.markdown("""
    <div class="explanation-box">
    <strong>🤖 Choose Your AI Model:</strong><br>
    <small>Each model uses different algorithms to analyze brain signals. Try different models to see which performs best for your data!</small>
    </div>
    """, unsafe_allow_html=True)
    
    if bci.is_trained:
        model_name = st.sidebar.selectbox(
            "Choose which AI model to use:",
            list(bci.models.keys()),
            index=0
        )
        st.sidebar.markdown(f"""
        <div class="explanation-box">
        <strong>✅ Selected Model:</strong> {model_name}<br>
        <small>This model will analyze your brain signals and predict your cognitive state. 
        Each model has different strengths - Random Forest is robust, SVM finds complex patterns, 
        and Logistic Regression is fast and interpretable.</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.warning("⚠️ Please train models first")
        model_name = "Random Forest"
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Real-time EEG Simulation")
        
        # Explain EEG simulation
        st.markdown("""
        <div class="explanation-box">
        <strong>🧠 What is EEG Simulation?</strong><br>
        We generate realistic brain wave patterns that mimic real EEG data. Each cognitive state has 
        unique brain wave characteristics:
        <br>• <strong>Resting:</strong> Alpha waves (8-12 Hz) - calm, relaxed state
        <br>• <strong>Memory:</strong> Alpha + Gamma bursts - active memory processing
        <br>• <strong>Math:</strong> Beta waves (13-30 Hz) + transients - focused problem solving
        <br>• <strong>Attention:</strong> Theta + Beta mix - concentrated focus
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed button explanation
        st.markdown("""
        <div class="explanation-box">
        <strong>🎯 Generate New EEG Sample Button:</strong><br>
        <small>This button creates a new simulated EEG recording (5 seconds, 61 channels) for a randomly 
        selected cognitive state. The AI then analyzes this brain data and predicts what cognitive 
        state it represents. You can compare the AI's prediction with the true state to see how well it performs!</small>
        </div>
        """, unsafe_allow_html=True)
        
        # EEG visualization
        if st.button("🎯 Generate New EEG Sample", type="primary"):
            # Select random condition
            condition = np.random.choice(bci.conditions)
            eeg_data = bci.generate_eeg_data(condition)
            
            # Store in history
            st.session_state.eeg_history.append((eeg_data, condition))
            
            # Make prediction
            if bci.is_trained:
                prediction, confidence, features = bci.predict(eeg_data, model_name)
                st.session_state.prediction_history.append(prediction)
                st.session_state.confidence_history.append(confidence)
            else:
                prediction, confidence = "Not Trained", 0.0
            
            # Display results with detailed explanations
            col_pred, col_conf = st.columns(2)
            with col_pred:
                st.markdown("""
                <div class="metric-card">
                <h4>🎯 Predicted Cognitive State</h4>
                <p style="font-size: 1.5rem; margin: 0; color: #ffffff !important;">{}</p>
                <small style="color: #cccccc !important;">This is what the AI thinks your brain is doing based on the EEG patterns it analyzed. 
                The AI uses machine learning to identify characteristic brain wave signatures for each cognitive state.</small>
                </div>
                """.format(prediction), unsafe_allow_html=True)
            with col_conf:
                st.markdown("""
                <div class="metric-card">
                <h4>📊 AI Confidence Score</h4>
                <p style="font-size: 1.5rem; margin: 0; color: #ffffff !important;">{:.1%}</p>
                <small style="color: #cccccc !important;">This percentage shows how certain the AI is about its prediction. 
                Higher values (closer to 100%) mean the AI is very confident, while lower values indicate uncertainty.</small>
                </div>
                """.format(confidence), unsafe_allow_html=True)
            
            # Show true condition
            st.markdown(f"""
            <div class="feature-highlight">
            <strong>🔍 True Condition:</strong> {condition}<br>
            <small>This is the actual cognitive state that was simulated. Compare this with the AI's prediction above!</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Plot EEG
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=['Frontal Channels', 'Central Channels', 'Occipital Channels'],
                vertical_spacing=0.1
            )
            
            # Select representative channels
            frontal_chs = [0, 1, 2, 3, 4]
            central_chs = [20, 21, 22, 23, 24]
            occipital_chs = [50, 51, 52, 53, 54]
            
            time_axis = np.linspace(0, 5, eeg_data.shape[1])
            
            for i, ch in enumerate(frontal_chs):
                fig.add_trace(
                    go.Scatter(x=time_axis, y=eeg_data[ch] + i*50, name=f'Ch{ch+1:02d}'),
                    row=1, col=1
                )
            
            for i, ch in enumerate(central_chs):
                fig.add_trace(
                    go.Scatter(x=time_axis, y=eeg_data[ch] + i*50, name=f'Ch{ch+1:02d}'),
                    row=2, col=1
                )
            
            for i, ch in enumerate(occipital_chs):
                fig.add_trace(
                    go.Scatter(x=time_axis, y=eeg_data[ch] + i*50, name=f'Ch{ch+1:02d}'),
                    row=3, col=1
                )
            
            fig.update_layout(
                height=600,
                title="🧠 EEG Signal Visualization - Brain Activity Over Time",
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.markdown("""
            <div class="explanation-box">
            <strong>📈 EEG Signal Visualization - What You're Seeing:</strong><br>
            <p>This interactive plot shows <strong>real-time brain activity</strong> from 15 representative EEG channels over 5 seconds:</p>
            <ul>
                <li><strong>Frontal Channels (Top):</strong> Show executive functions, attention, and decision-making</li>
                <li><strong>Central Channels (Middle):</strong> Display motor cortex activity and sensory processing</li>
                <li><strong>Occipital Channels (Bottom):</strong> Represent visual processing and alpha wave activity</li>
            </ul>
            <p><strong>How to Read:</strong> Each wavy line represents electrical activity from one brain region. 
            Different cognitive states create distinct patterns - alpha waves for resting, beta waves for math, 
            gamma bursts for memory tasks. The AI analyzes these patterns to make predictions!</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("📈 System Status")
        
        # Training status
        if bci.is_trained:
            st.markdown('<p class="status-success">✅ Models Trained</p>', unsafe_allow_html=True)
            st.markdown("""
            <div class="explanation-box">
            <strong>✅ AI Models Ready!</strong><br>
            The machine learning models have been trained and are ready to analyze brain signals. 
            You can now generate EEG samples and get predictions.
            </div>
            """, unsafe_allow_html=True)
            st.metric("Models Available", len(bci.models))
        else:
            st.markdown('<p class="status-warning">⚠️ Models Not Trained</p>', unsafe_allow_html=True)
            st.markdown("""
            <div class="explanation-box">
            <strong>⚠️ Training Required</strong><br>
            Click "🚀 Train Models" in the sidebar to train the AI models. This will take about 30 seconds 
            and will teach the AI to recognize different brain states.
            </div>
            """, unsafe_allow_html=True)
        
        # Prediction history
        if st.session_state.prediction_history:
            st.subheader("🎯 Recent Predictions")
            
            st.markdown("""
            <div class="explanation-box">
            <strong>📊 Prediction History</strong><br>
            This shows the last 10 predictions made by the AI, along with how confident it was 
            about each prediction. Higher confidence means the AI was more certain.
            </div>
            """, unsafe_allow_html=True)
            
            # Create prediction dataframe
            pred_df = pd.DataFrame({
                'Prediction': list(st.session_state.prediction_history),
                'Confidence': [f"{conf:.1%}" for conf in list(st.session_state.confidence_history)]
            })
            
            # Show recent predictions
            st.dataframe(pred_df.tail(10), width='stretch')
            
            # Confidence trend
            if len(st.session_state.confidence_history) > 1:
                st.markdown("""
                <div class="explanation-box">
                <strong>📈 Confidence Trend</strong><br>
                This graph shows how confident the AI has been over time. A rising trend means 
                the AI is getting more confident, while a falling trend might indicate difficult cases.
                </div>
                """, unsafe_allow_html=True)
                
                fig_conf = go.Figure()
                fig_conf.add_trace(go.Scatter(
                    y=list(st.session_state.confidence_history),
                    mode='lines+markers',
                    name='Confidence',
                    line=dict(color='#2E86AB', width=3),
                    marker=dict(size=6, color='#2E86AB')
                ))
                fig_conf.update_layout(
                    title="🤖 AI Confidence Over Time",
                    xaxis_title="Prediction Number",
                    yaxis_title="Confidence Level",
                    height=200,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_conf, width='stretch')
        
        # Feature importance (if available)
        if bci.is_trained and hasattr(bci.models['Random Forest'], 'feature_importances_'):
            st.subheader("🔍 Feature Importance")
            
            st.markdown("""
            <div class="explanation-box">
            <strong>🧠 What are Features?</strong><br>
            Features are measurements extracted from brain signals that help the AI make predictions. 
            This shows which features (like alpha waves, beta waves, etc.) are most important for 
            distinguishing between different cognitive states.
            </div>
            """, unsafe_allow_html=True)
            
            # Get top 10 features
            importances = bci.models['Random Forest'].feature_importances_
            top_features = np.argsort(importances)[-10:]
            
            # Create feature names
            feature_names = [
                'Delta Power', 'Theta Power', 'Alpha Power', 'Beta Power', 'Gamma Power',
                'Rel Delta', 'Rel Theta', 'Rel Alpha', 'Rel Beta', 'Rel Gamma',
                'Mean Amp', 'Std Amp', 'Variance', 'Spectral Centroid', 'Spectral Bandwidth'
            ]
            
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                x=[f'Feature {i+1}' for i in top_features],
                y=importances[top_features],
                orientation='v',
                marker=dict(color='#2E86AB', opacity=0.8)
            ))
            fig_imp.update_layout(
                title="🎯 Top 10 Most Important Brain Features",
                xaxis_title="Feature Number",
                yaxis_title="Importance Score",
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_imp, width='stretch')
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #ffffff !important; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 2rem; border-radius: 15px; margin-top: 2rem; border: 2px solid #00d4ff; box-shadow: 0 0 30px rgba(0, 212, 255, 0.3);'>
        <h3 style='color: #00d4ff !important; text-shadow: 0 0 10px #00d4ff;'>🧠 Interactive MindWave BCI platform with real-time EEG processing and cognitive state classification using Python and Streamlit</h3>
        <p style='margin: 0.5rem 0; font-size: 1.1rem; color: #ffffff !important;'>
            <strong style='color: #4ecdc4;'>Advanced Brain-Computer Interface System</strong><br>
            Built with Python, Streamlit, and Machine Learning<br>
            <em style='color: #ff6b6b;'>Demonstrating real-time EEG analysis and cognitive state classification</em>
        </p>
        <div style='margin-top: 1rem; font-size: 0.9rem; color: #cccccc !important;'>
            Perfect for research, education, and job applications! 🚀
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

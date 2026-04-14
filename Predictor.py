#!/usr/bin/env python
# coding: utf-8

# # Prediction of shear strength in RC columns

# In[8]:

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Predictor of RC Column Shear Strength",
    page_icon="🏗️",
    layout="wide"
)

# ==================== 字体大小设置 ====================
with st.sidebar:
    st.header("📌 Model Information")
    st.markdown("""
     - **Input**: 8 features  
     - **Output**: Shear strength $V_u$ (kN)  
     - **Algorithm**: XGBoost Regressor
    """)
    st.divider()
    st.caption("Model file: Prediction of shear strength in RC columns.pkl")
    
    st.subheader("⚙️ Appearance")
    font_size = st.slider(
        "Font Size (px)", 
        min_value=12, 
        max_value=24, 
        value=16, 
        step=1,
        help="Adjust the font size of the main content (including labels, descriptions, and results)"
    )
    
    # 全面控制字体大小的CSS（覆盖所有文本元素）
    st.markdown(f"""
    <style>
    /* 主容器基础字体 */
    .main, .stApp, .stMarkdown, .stAlert, .stException {{
        font-size: {font_size}px !important;
    }}
    /* 标题字体 */
    h1 {{
        font-size: {font_size + 8}px !important;
    }}
    h2, h3, .stMarkdown h2, .stMarkdown h3 {{
        font-size: {font_size + 4}px !important;
    }}
    /* 普通段落、列表、说明文字 */
    p, li, .stMarkdown p, .stMarkdown li, .stMarkdown div, .stMarkdown span {{
        font-size: {font_size}px !important;
    }}
    /* 输入框标签（包括帮助文本） */
    .stNumberInput label, .stTextInput label, .stSelectbox label, .stSlider label {{
        font-size: {font_size}px !important;
        font-weight: 500 !important;
    }}
    /* 按钮文字 */
    .stButton button, .stDownloadButton button {{
        font-size: {font_size}px !important;
    }}
    /* 指标卡片 - 标签字体 */
    .stMetric label {{
        font-size: {font_size}px !important;
    }}
    /* 指标卡片 - 数值字体（增大并加粗） */
    .stMetric .stMetricValue, .stMetric value {{
        font-size: {font_size + 6}px !important;
        font-weight: bold !important;
    }}
    /* 警告、成功、错误框内文字 */
    .stAlert div, .stAlert p {{
        font-size: {font_size}px !important;
    }}
    /* 侧边栏文字（独立控制，稍小一点更协调） */
    .css-1d391kg, .sidebar .stMarkdown, .sidebar .stCaption {{
        font-size: {font_size - 1}px !important;
    }}
    /* 扩展器标题和内容 */
    .streamlit-expanderHeader, .streamlit-expanderContent {{
        font-size: {font_size}px !important;
    }}
    /* 数据表格内文字 */
    .dataframe, .stTable td, .stTable th {{
        font-size: {font_size - 1}px !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# 检查 xgboost 是否可用
try:
    import xgboost
    st.sidebar.success("✅ XGBoost backend ready")
except ImportError:
    st.sidebar.error("❌ XGBoost not installed. Please add 'xgboost' to requirements.txt")
    st.stop()

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load('Prediction of shear strength in RC columns.pkl')

try:
    model = load_model()
    st.success("✅ Model loaded successfully!", icon="🎉")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}\nEnsure the model file exists and xgboost is installed.")
    st.stop()

st.title("🏗️ Predictor of RC Column Shear Strength")
st.markdown("""
This application is based on the Ph3-XGBR model to predict the shear strength $V_u$ (kN) of reinforced concrete (RC) columns 
expected to undergo shear or flexural-shear failure.
Please enter the following 8 feature values and click the predict button to obtain the result.
""")

# ==================== 定义模型训练参数范围 ====================
PARAM_RANGES = {
    'L': {'min': 80.0, 'max': 1600.0, 'name': 'L (column height)'},
    'fc': {'min': 4.0, 'max': 216.0, 'name': 'fc (concrete strength)'},
    'ρs': {'min': 0.0, 'max': 2.0, 'name': 'ρs (transverse reinforcement ratio)'},
    'P': {'min': 0.0, 'max': 18400.0, 'name': 'P (axial load)'},
    'Vc': {'min': 32.0, 'max': 2830.0, 'name': 'Vc (concrete contribution)'},
    'Vs': {'min': 0.0, 'max': 3580.0, 'name': 'Vs (transverse reinforcement contribution)'},
    'Vl': {'min': 25.0, 'max': 11400.0, 'name': 'Vl (longitudinal reinforcement contribution)'},
    'Vp': {'min': 0.0, 'max': 2520.0, 'name': 'Vp (axial force contribution)'}
}

def check_parameters_out_of_range(L, fc, ρs, P, Vc, Vs, Vl, Vp):
    out_of_range = []
    if L < PARAM_RANGES['L']['min'] or L > PARAM_RANGES['L']['max']:
        out_of_range.append(PARAM_RANGES['L']['name'])
    if fc < PARAM_RANGES['fc']['min'] or fc > PARAM_RANGES['fc']['max']:
        out_of_range.append(PARAM_RANGES['fc']['name'])
    if ρs < PARAM_RANGES['ρs']['min'] or ρs > PARAM_RANGES['ρs']['max']:
        out_of_range.append(PARAM_RANGES['ρs']['name'])
    if P < PARAM_RANGES['P']['min'] or P > PARAM_RANGES['P']['max']:
        out_of_range.append(PARAM_RANGES['P']['name'])
    if Vc < PARAM_RANGES['Vc']['min'] or Vc > PARAM_RANGES['Vc']['max']:
        out_of_range.append(PARAM_RANGES['Vc']['name'])
    if Vs < PARAM_RANGES['Vs']['min'] or Vs > PARAM_RANGES['Vs']['max']:
        out_of_range.append(PARAM_RANGES['Vs']['name'])
    if Vl < PARAM_RANGES['Vl']['min'] or Vl > PARAM_RANGES['Vl']['max']:
        out_of_range.append(PARAM_RANGES['Vl']['name'])
    if Vp < PARAM_RANGES['Vp']['min'] or Vp > PARAM_RANGES['Vp']['max']:
        out_of_range.append(PARAM_RANGES['Vp']['name'])
    return out_of_range

st.header("📊 Input features")
col1, col2 = st.columns(2)

with col1:
    L = st.number_input(
        "**L** - column height (distance from point of maximum moment to point of zero moment) / mm", 
        value=1000.0, step=10.0,
        help="Column height (distance from point of maximum moment to point of zero moment)"
    )
    fc = st.number_input(
        "**fc** - concrete compressive strength / MPa", 
        value=25.0, step=1.0,
        help="Concrete compressive strength in MPa"
    )
    ρs = st.number_input(
        "**ρs** - transverse reinforcement ratio (%)", 
        value=0.02, step=0.01, format="%.4f",
        help="Transverse reinforcement ratio as percentage"
    )
    P = st.number_input(
        "**P** - axial load / kN", 
        value=500.0, step=50.0,
        help="Applied axial compressive load in kN"
    )

with col2:
    Vc = st.number_input(
        "**Vc** - shear contribution of concrete: fc^0.5·Ag/(L/h) / kN", 
        value=200.0, step=10.0,
        help="Shear contribution from concrete based on fc^0.5·Ag/(L/h)"
    )
    Vs = st.number_input(
        "**Vs** - shear contribution of transverse reinforcement: As·fys·h/s / kN", 
        value=150.0, step=10.0,
        help="Shear contribution from transverse reinforcement based on As·fys·h/s"
    )
    Vl = st.number_input(
        "**Vl** - shear contribution of longitudinal reinforcement: Al·fyl / kN", 
        value=300.0, step=10.0,
        help="Shear contribution from longitudinal reinforcement based on Al·fyl"
    )
    Vp = st.number_input(
        "**Vp** - shear contribution of axial force: P·(h-c)/(2L) / kN", 
        value=0.0, step=10.0,
        help="Shear contribution from axial force based on P·(h-c)/(2L)"
    )

input_data = pd.DataFrame({
    'L': [L], 'fc': [fc], 'ρs': [ρs], 'P': [P],
    'Vc': [Vc], 'Vs': [Vs], 'Vl': [Vl], 'Vp': [Vp]
})

st.markdown("---")
col_btn, col_res = st.columns([1, 3])
with col_btn:
    predict_btn = st.button("🔮 Start predicting", type="primary", use_container_width=True)

if predict_btn:
    out_of_range_params = check_parameters_out_of_range(L, fc, ρs, P, Vc, Vs, Vl, Vp)
    
    with st.spinner("Prediction in progress, please wait..."):
        prediction = model.predict(input_data)[0]
    
    with col_res:
        st.success("### Prediction result")
        st.metric(label="📐 Shear strength $V_u$ (kN)", value=f"{prediction:.2f} kN")
    
    if out_of_range_params:
        st.warning(f"⚠️ **The input parameters are outside the model's range. Use the prediction results with caution**\n\nThe following parameters are outside the range of the model：\n- " + "\n- ".join(out_of_range_params))
        
        with st.expander("📊 Please check the range of model parameters."):
            range_df = pd.DataFrame([
                {"parameters": "L (mm) - column height (distance from point of maximum moment to point of zero moment)", "range": f"{PARAM_RANGES['L']['min']} – {PARAM_RANGES['L']['max']}"},
                {"parameters": "fc (MPa)", "range": f"{PARAM_RANGES['fc']['min']} – {PARAM_RANGES['fc']['max']}"},
                {"parameters": "ρs (%)", "range": f"{PARAM_RANGES['ρs']['min']} – {PARAM_RANGES['ρs']['max']}"},
                {"parameters": "P (kN)", "range": f"{PARAM_RANGES['P']['min']} – {PARAM_RANGES['P']['max']}"},
                {"parameters": "Vc (kN)", "range": f"{PARAM_RANGES['Vc']['min']} – {PARAM_RANGES['Vc']['max']}"},
                {"parameters": "Vs (kN)", "range": f"{PARAM_RANGES['Vs']['min']} – {PARAM_RANGES['Vs']['max']}"},
                {"parameters": "Vl (kN)", "range": f"{PARAM_RANGES['Vl']['min']} – {PARAM_RANGES['Vl']['max']}"},
                {"parameters": "Vp (kN)", "range": f"{PARAM_RANGES['Vp']['min']} – {PARAM_RANGES['Vp']['max']}"}
            ])
            st.table(range_df)
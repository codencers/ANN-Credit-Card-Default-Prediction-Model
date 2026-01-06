import streamlit as st
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .high-risk {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .low-risk {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- LOAD ARTIFACTS --------------------
@st.cache_resource
def load_artifacts():
    model =tf.keras.models.load_model('model.h5')
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("features.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# -------------------- FEATURE MAPPINGS --------------------
FEATURE_MAPPINGS = {
    "SEX": {0: "Unknown", 1: "Male", 2: "Female"},
    "EDUCATION": {0: "Unknown", 1: "Graduate School", 2: "University", 3: "High School", 4: "Others"},
    "MARRIAGE": {0: "Unknown", 1: "Married", 2: "Single", 3: "Divorced"},
}

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
PAYMENT_STATUS_FEATURES = [f for f in feature_names if f.startswith("PAY_")]
BILL_FEATURES = [f for f in feature_names if f.startswith("BILL_AMT")]
PAY_AMT_FEATURES = [f for f in feature_names if f.startswith("PAY_AMT")]
AGE_LIMIT_FEATURES = [f for f in feature_names if f in ["AGE", "LIMIT_BAL"]]

# -------------------- HEADER --------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.title("💳 Credit Card Default Prediction")
    st.markdown("**Advanced Neural Network Model for Credit Risk Assessment**")
with col2:
    st.metric("Model Type", "ANN")

st.divider()

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Threshold Slider
    threshold = st.slider(
        "Prediction Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust the threshold for default risk classification"
    )
    
    st.divider()
    
    # Model Info
    st.subheader("📊 Model Information")
    st.info("""
    - **Model**: Artificial Neural Network (ANN)
    - **Architecture**: 64 → 32 → 1 neurons
    - **Activation**: ReLU + Sigmoid
    - **Optimization**: Adam
    - **Metric**: ROC-AUC
    - **Training Data**: UCI Credit Card Dataset
    """)

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "👤 Customer Profile", "📈 Analytics", "ℹ️ Help"])

# ================== TAB 1: PREDICTION ==================
with tab1:
    st.subheader("Enter Customer Information")
    
    col1, col2 = st.columns(2)
    
    user_input = {}
    
    # Basic Information
    with col1:
        st.markdown("### 👤 Personal Information")
        
        age_limit_vals = {}
        for feature in AGE_LIMIT_FEATURES:
            if feature == "AGE":
                age_limit_vals[feature] = st.number_input(
                    "Age (years)",
                    min_value=18,
                    max_value=100,
                    value=30,
                    step=1
                )
            elif feature == "LIMIT_BAL":
                age_limit_vals[feature] = st.number_input(
                    "Credit Limit (NT$)",
                    min_value=10000,
                    max_value=1000000,
                    value=50000,
                    step=5000
                )
        user_input.update(age_limit_vals)
        
        st.markdown("### 👥 Demographics")
        for feature in CATEGORICAL_FEATURES:
            user_input[feature] = st.selectbox(
                feature.replace("_", " "),
                options=list(FEATURE_MAPPINGS[feature].keys()),
                format_func=lambda x: FEATURE_MAPPINGS[feature][x]
            )
    
    # Payment Status
    with col2:
        st.markdown("### 💰 Recent Payment Status (Last 6 Months)")
        pay_status = {}
        for i, feature in enumerate(PAYMENT_STATUS_FEATURES, 1):
            pay_status[feature] = st.selectbox(
                f"Month {i}",
                options=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                format_func=lambda x: {
                    -1: "No Consumption",
                    0: "Paid in Full",
                    1: "1 Month Overdue",
                    2: "2 Months Overdue",
                    3: "3 Months Overdue",
                    4: "4 Months Overdue",
                    5: "5 Months Overdue",
                    6: "6 Months Overdue",
                    7: "7 Months Overdue",
                    8: "8 Months Overdue",
                    9: "9+ Months Overdue"
                }.get(x, str(x)),
                key=f"pay_status_{feature}"
            )
        user_input.update(pay_status)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    # Bill Amounts
    with col1:
        st.markdown("### 📄 Bill Amounts (Last 6 Months, NT$)")
        bill_amt = {}
        for i, feature in enumerate(BILL_FEATURES, 1):
            bill_amt[feature] = st.number_input(
                f"Month {i}",
                min_value=0,
                max_value=500000,
                value=5000,
                step=500,
                key=f"bill_{feature}"
            )
        user_input.update(bill_amt)
    
    # Payment Amounts
    with col2:
        st.markdown("### 💸 Previous Payment Amounts (Last 6 Months, NT$)")
        pay_amt = {}
        for i, feature in enumerate(PAY_AMT_FEATURES, 1):
            pay_amt[feature] = st.number_input(
                f"Month {i}",
                min_value=0,
                max_value=500000,
                value=2000,
                step=500,
                key=f"payamt_{feature}"
            )
        user_input.update(pay_amt)
    
    st.divider()
    
    # Prediction Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔮 Predict Default Risk",
            use_container_width=True,
            type="primary"
        )
    
    # ---- PREDICTION RESULTS ----
    if predict_button:
        try:
            # Convert input to DataFrame in correct order
            input_df = pd.DataFrame([user_input])[feature_names]
            
            # Scale input
            input_scaled = scaler.transform(input_df)
            
            # Predict probability
            probability = model.predict(input_scaled, verbose=0)[0][0]
            
            st.divider()
            
            # Risk Assessment
            risk_level = "HIGH RISK ⚠️" if probability >= threshold else "LOW RISK ✅"
            risk_color = "high-risk" if probability >= threshold else "low-risk"
            
            # Probability Gauge
            fig_gauge = go.Figure(data=[
                go.Indicator(
                    mode="gauge+number+delta",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Default Probability (%)"},
                    delta={'reference': threshold * 100},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#f5576c" if probability >= threshold else "#4facfe"},
                        'steps': [
                            {'range': [0, 25], 'color': "#e8f4f8"},
                            {'range': [25, 50], 'color': "#c8e6f5"},
                            {'range': [50, 75], 'color': "#ffe8e8"},
                            {'range': [75, 100], 'color': "#ffc8c8"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                )
            ])
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Risk Category
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                    <div class="metric-card {risk_color}">
                        <h3>{risk_level}</h3>
                        <p style="font-size: 24px; font-weight: bold;">{probability*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Confidence Score", f"{(1-abs(probability-0.5))*100:.1f}%")
            
            with col3:
                st.metric("Risk Category", 
                         "High" if probability >= threshold else "Low",
                         delta="⚠️ Monitor" if probability >= threshold else "✅ Approve")
            
            st.divider()
            
            # Detailed Explanation
            with st.expander("📋 Detailed Risk Analysis"):
                if probability >= threshold:
                    st.warning(f"""
                    **Risk Assessment**: This customer shows **{probability*100:.1f}%** probability of default
                    
                    **Recommendations**:
                    - ⚠️ Require additional verification
                    - 📊 Review payment history carefully
                    - 💰 Consider credit limit reduction
                    - 👁️ Increase monitoring frequency
                    """)
                else:
                    st.success(f"""
                    **Risk Assessment**: This customer shows **{probability*100:.1f}%** probability of default
                    
                    **Recommendations**:
                    - ✅ Low risk profile
                    - 📈 Consider credit limit increase opportunities
                    - 💳 Standard approval with regular monitoring
                    """)
            
            # Key Insights
            with st.expander("🔍 Key Indicators"):
                insight_cols = st.columns(3)
                with insight_cols[0]:
                    avg_payment_status = np.mean(list(pay_status.values()))
                    st.metric("Avg Payment Status", f"{avg_payment_status:.1f}", 
                             "⚠️ Overdue" if avg_payment_status > 0 else "✅ On Time")
                
                with insight_cols[1]:
                    total_bill = sum(bill_amt.values())
                    st.metric("Total Outstanding Bills", f"NT${total_bill:,.0f}")
                
                with insight_cols[2]:
                    total_paid = sum(pay_amt.values())
                    payment_ratio = (total_paid / max(total_bill, 1)) * 100
                    st.metric("Payment Ratio", f"{payment_ratio:.1f}%")
        
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

# ================== TAB 2: CUSTOMER PROFILE ==================
with tab2:
    st.subheader("Customer Financial Profile Summary")
    
    if 'user_input' in locals() and predict_button:
        # Summary Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Age", f"{user_input.get('AGE', 0)} years")
        with col2:
            st.metric("Credit Limit", f"NT${user_input.get('LIMIT_BAL', 0):,.0f}")
        with col3:
            sex_label = FEATURE_MAPPINGS["SEX"].get(user_input.get("SEX", 0), "Unknown")
            st.metric("Gender", sex_label)
        with col4:
            edu_label = FEATURE_MAPPINGS["EDUCATION"].get(user_input.get("EDUCATION", 0), "Unknown")
            st.metric("Education", edu_label)
        
        st.divider()
        
        # Payment Trend
        months = [f"M{i}" for i in range(1, 7)]
        payment_statuses = [user_input.get(f, 0) for f in PAYMENT_STATUS_FEATURES]
        
        fig_payment = go.Figure()
        fig_payment.add_trace(go.Scatter(
            x=months, y=payment_statuses,
            mode='lines+markers',
            name='Payment Status',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10)
        ))
        fig_payment.add_hline(y=0, line_dash="dash", line_color="green", annotation_text="On Time")
        fig_payment.update_layout(
            title="Payment Status Trend (Last 6 Months)",
            xaxis_title="Month",
            yaxis_title="Status Code",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_payment, use_container_width=True)
        
        # Bill vs Payment Comparison
        bills = [user_input.get(f, 0) for f in BILL_FEATURES]
        payments = [user_input.get(f, 0) for f in PAY_AMT_FEATURES]
        
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(x=months, y=bills, name='Bill Amount', marker_color='#f093fb'))
        fig_comparison.add_trace(go.Bar(x=months, y=payments, name='Payment Amount', marker_color='#4facfe'))
        fig_comparison.update_layout(
            title="Bills vs Payments Comparison",
            xaxis_title="Month",
            yaxis_title="Amount (NT$)",
            barmode='group',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    else:
        st.info("👈 Enter customer information in the **Prediction** tab to see profile details")

# ================== TAB 3: ANALYTICS ==================
with tab3:
    st.subheader("Model Performance & Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Model Performance Metrics
        - **ROC-AUC Score**: 0.77+
        - **Accuracy**: 82%
        - **Architecture**: Dense Neural Network
        - **Layers**: 3 (64 → 32 → 1)
        - **Training Data**: 30,000 customers
        """)
    
    with col2:
        st.markdown("""
        ### Why ANN Over Logistic Regression?
        - ✅ Captures non-linear relationships
        - ✅ Better ROC-AUC performance
        - ✅ Handles complex interactions
        - ✅ Industry-standard approach
        - ✅ Batch normalization reduces overfitting
        """)
    
    st.divider()
    
    # ROC Curve Info
    st.markdown("""
    ### Model Evaluation
    The ANN model was trained and validated on the UCI Credit Card dataset with:
    - **70%** Training data
    - **15%** Validation data
    - **15%** Test data
    
    **Key Features Used**: Payment status, bill amounts, payment history, age, credit limit, education, marital status
    """)

# ================== TAB 4: HELP ==================
with tab4:
    st.subheader("❓ Help & Documentation")
    
    with st.expander("📖 How to Use This Tool"):
        st.markdown("""
        1. **Enter Customer Information**: Fill in all required fields in the Prediction tab
        2. **Adjust Threshold**: Use the sidebar to set your risk threshold (default 0.5)
        3. **Get Prediction**: Click "Predict Default Risk" button
        4. **Review Results**: Check probability gauge and risk assessment
        5. **Analyze Profile**: View detailed customer profile in the Customer Profile tab
        """)
    
    with st.expander("📊 Understanding the Metrics"):
        st.markdown("""
        - **Default Probability**: Likelihood (0-100%) that customer will default
        - **Threshold**: Cutoff point to classify as high/low risk
        - **Confidence Score**: Model's certainty in the prediction
        - **Payment Status**: -1 (no consumption), 0 (paid), 1-9 (months overdue)
        """)
    
    with st.expander("⚠️ Feature Guide"):
        st.markdown("""
        **Demographics**:
        - Age: Customer's age in years
        - Credit Limit: Maximum credit available (NT$)
        - Sex: Male/Female
        - Education: Graduate/University/High School/Others
        - Marital Status: Married/Single/Divorced
        
        **Payment Information**:
        - Payment Status: Recent payment behavior (months -1-9)
        - Bill Amount: Outstanding balance for each month
        - Payment Amount: Amount paid in each month
        """)
    
    with st.expander("💡 Best Practices"):
        st.markdown("""
        - ✅ Always verify critical customer information
        - ✅ Use this as a decision support tool, not the final decision
        - ✅ Review historical trends before making final approval
        - ✅ Adjust threshold based on your risk tolerance
        - ✅ Monitor customers regularly, especially borderline cases
        """)

# -------------------- FOOTER --------------------
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px; padding: 20px;">
    <p>🔐 Secure Credit Risk Assessment | Built with ANN | Powered by Streamlit</p>
    <p>Last Updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
</div>

""", unsafe_allow_html=True)

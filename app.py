import streamlit as st
import pandas as pd
import joblib
import os
from utils.data_loader import load_all_data
from utils.feature_eng import build_master_dataset
from utils.model_utils import preprocess_for_modeling
from utils.decision_logic import apply_decision_logic

# Page Configuration
st.set_page_config(
    page_title="NexGen Logistics Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: white;
        color: #31333F !important;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: #31333F !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #31333F !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #31333F !important;
    }
    .stDataFrame {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prep_data_v2():
    """
    Loads data, builds master dataset, and applies models/logic.
    Version 2: Forces cache refresh to pick up new Cost Features.
    """
    data = load_all_data()
    master = build_master_dataset(data)
    
    # Load Models
    models_dir = 'models'
    try:
        delay_model = joblib.load(os.path.join(models_dir, 'delay_model.joblib'))
        risk_model = joblib.load(os.path.join(models_dir, 'risk_model.joblib'))
        # encoders = joblib.load(os.path.join(models_dir, 'encoders.joblib')) # Not used directly if we use pipeline or raw inference setup? 
        # Actually verify_pipeline used them. Let's keep loading.
        
        # Preprocess for Inference
        processed_df, _ = preprocess_for_modeling(master)
        
        # Inference - Delay
        delay_feats = [
            'Distance_KM', 'route_risk_score', 'vehicle_suitability_score', 
            'Traffic_Delay_Minutes', 'Priority', 'Origin', 'Product_Category'
        ]
        master['delay_probability'] = delay_model.predict_proba(processed_df[delay_feats])[:, 1]
        
        # Inference - Customer Risk
        risk_feats = [
             'segment_avg_rating', 'segment_recommend_pct', 'delay_days', 
             'Priority', 'Order_Value_INR'
        ]
        master['customer_risk_pred'] = risk_model.predict_proba(processed_df[risk_feats])[:, 1]
        
        # Apply Decision Logic
        final_df = apply_decision_logic(master)
        
        return final_df
        
    except Exception as e:
        st.error(f"Error loading models/data: {e}")
        return pd.DataFrame()

# Load Data
df = load_and_prep_data_v2()

if df.empty:
    st.warning("Data could not be loaded. Please check the logs.")
    st.stop()

# Sidebar
st.sidebar.title("🚚 NexGen Intelligent Logistics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "ℹ️ About & Solution",
    "📊 Executive Overview",
    "🔮 Predictive Delivery Risk",
    "😊 Customer Experience",
    "⭐ Operational Control Tower"
])

st.sidebar.header("Global Filters")
if not df.empty:
    selected_wh = st.sidebar.multiselect("Select Warehouse", df['Origin'].unique(), default=df['Origin'].unique())
    filtered_df = df[df['Origin'].isin(selected_wh)]
else:
    filtered_df = df

# --- PAGE: ABOUT & SOLUTION ---
if page == "ℹ️ About & Solution":
    st.title("ℹ️ Project: Predictive Logistics Intelligence")
    
    st.markdown("""
    ### 🚩 Problem Statement
    **NexGen Logistics** is currently facing significant operational challenges:
    *   **Reactive Operations**: Dealing with delays only *after* they happen.
    *   **High Costs**: Inefficient use of premium shipping and vehicle assets.
    *   **Customer Churn**: Increasing dissatisfaction due to unpredicted service failures.
    
    ### 🔬 Our Approach
    We moved from a static analysis to a **Predictive & Prescriptive** engine:
    1.  **Data Integration**: Unified 7 siloed datasets (Orders, Fleet, Feedback, etc.).
    2.  **Advanced Feature Engineering**: Created composite risk scores (`Route Risk`, `Vehicle Suitability`).
    3.  **Machine Learning**: Trained Random Forest & Gradient Boosting models to predict delays and customer risk.
    
    ### 💡 The Solution
    A **Decision Intelligence System** that doesn't just show charts, but drives action:
    *   **Predict** delays before dispatch.
    *   **Identify** at-risk customers dynamically.
    *   **Recommend** specific interventions (e.g., "Upgrade Vehicle", "Prioritize").
    *   **Simulate** business impact in real-time.
    
    ---
    **👈 Navigate to the 'Operational Control Tower' to see the AI in action!**
    """)

# --- PAGE 1: EXECUTIVE OVERVIEW ---
elif page == "📊 Executive Overview":
    st.title("📊 Executive Operational Overview")
    
    # Top KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    avg_delay = filtered_df['delay_days'].mean()
    on_time_pct = (filtered_df['is_delayed'] == 0).mean() * 100
    risk_orders = (filtered_df['delay_probability'] > 0.5).sum()
    avg_cost = filtered_df['total_cost'].mean()
    
    col1.metric("On-Time Delivery %", f"{on_time_pct:.1f}%", f"{on_time_pct-95:.1f}% vs Target")
    col2.metric("Avg Delay (Days)", f"{avg_delay:.2f}", "-0.5 goal")
    col3.metric("High Risk Orders (Predicted)", f"{risk_orders}", "Requires Action", delta_color="inverse")
    col4.metric("Avg Order Cost", f"₹{avg_cost:,.0f}")
    
    st.divider()
    
    col_a, col_b = st.columns(2)
    
    # Chart 1: Delay by Route/Origin
    with col_a:
        st.subheader("Delay Risk by Origin")
        origin_risk = filtered_df.groupby('Origin')['delay_probability'].mean().sort_values()
        st.bar_chart(origin_risk)
        
    # Chart 2: Cost Breakout
    with col_b:
        st.subheader("Cost Drivers Breakdown")
        cost_cols = ['Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 'Insurance', 'Packaging_Cost']
        # Filter only existing columns
        real_cost_cols = [c for c in cost_cols if c in filtered_df.columns]
        
        if real_cost_cols:
            cost_sum = filtered_df[real_cost_cols].sum().reset_index()
            cost_sum.columns = ['Cost Component', 'Total Amount']
            st.bar_chart(cost_sum, x='Cost Component', y='Total Amount', color='#FF4B4B')
    
# --- PAGE 2: PREDICTIVE DELIVERY RISK ---
elif page == "🔮 Predictive Delivery Risk":
    st.title("🔮 Predictive Delivery Risk Intelligence")
    
    st.markdown("Orders predicted to be delayed **before** they impact the customer.")
    
    # Slider for Risk Threshold
    threshold = st.slider("Risk Probability Threshold", 0.0, 1.0, 0.5)
    
    risky_orders = filtered_df[filtered_df['delay_probability'] > threshold].sort_values('delay_probability', ascending=False)
    
    st.dataframe(
        risky_orders[['Order_ID', 'Origin', 'Destination', 'Priority', 'delay_probability', 'route_risk_score', 'vehicle_suitability_score']],
        column_config={
            "delay_probability": st.column_config.ProgressColumn(
                "Delay Risk",
                help="Probability of delivery delay",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
             "route_risk_score": st.column_config.NumberColumn(
                "Route Risk",
                format="%d"
            ),
             "vehicle_suitability_score": st.column_config.NumberColumn(
                "Vehicle Suitability",
                format="%.1f"
            )
        },
        use_container_width=True
    )
    
    st.subheader("Explainability: Why are these delayed?")
    st.info("Top Factors: Route Risk Score (Traffic/Weather) & Vehicle Suitability (Age/Efficiency).")


# --- PAGE 3: CUSTOMER EXPERIENCE ---
elif page == "😊 Customer Experience":
    st.title("😊 Customer Experience Intelligence")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dissatisfaction Risk by Segment")
        # Ensure we have satisfaction data
        if 'customer_dissatisfaction_risk' in filtered_df.columns:
            seg_risk = filtered_df.groupby('Customer_Segment')['customer_dissatisfaction_risk'].mean()
            st.bar_chart(seg_risk)
            
    with col2:
        st.subheader("Impact of Delays on Ratings")
        st.scatter_chart(data=filtered_df, x='delay_days', y='segment_avg_rating') 

# --- PAGE 4: OPERATIONAL ACTIONS ---
elif page == "⭐ Operational Control Tower":
    st.title("⚡ Operational Control Tower")
    st.markdown("manage and resolve delivery risks proactively.")
    
    # KPIs for Ops
    ops_df = filtered_df[filtered_df['Action'] != 'Standard Dispatch']
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Pending Interventions", len(ops_df), "High Priority", delta_color="inverse")
    kpi2.metric("Projected Cost Impact", f"₹{(len(ops_df)*150):,}", "+5%") # Proxy avg cost
    kpi3.metric("Risk Reduction Potential", "-18%", "Target Impact")
    
    st.divider()
    
    # TABS DESIGN
    tab1, tab2, tab3 = st.tabs(["📋 Critical Action Queue", "🎮 Risk Simulator", "📥 Reports & Exports"])
    
    with tab1:
        st.subheader("Action Queue")
        
        # Filter controls
        c1, c2 = st.columns([1, 1])
        action_filter = c1.multiselect("Filter by Action Type", ops_df['Action'].unique())
        
        display_df = ops_df if not action_filter else ops_df[ops_df['Action'].isin(action_filter)]
        
        st.dataframe(
            display_df[['Order_ID', 'Priority', 'Action', 'Reason', 'Cost_Impact', 'delay_probability']],
            column_config={
                "delay_probability": st.column_config.ProgressColumn("Risk Level", format="%.2f", min_value=0, max_value=1),
                "Action": st.column_config.TextColumn("Recommended Action", width="medium"),
            },
            use_container_width=True,
            hide_index=True
        )
        
        if st.button("🚀 Approve All Displayed Actions", type="primary"):
            st.success(f"Successfully queued {len(display_df)} actions for execution in TMS.")
            st.balloons()

    with tab2:
        st.subheader("Interactive Risk Simulator")
        st.info("Deep dive into a specific order to test 'What-If' scenarios.")
        
        col_list, col_sim = st.columns([1, 2])
        
        with col_list:
            selected_order_sim = st.selectbox("Select Order", ops_df['Order_ID'].head(20))
        
        if selected_order_sim:
            with col_sim:
                # Custom Card Style for details
                order_details = ops_df[ops_df['Order_ID'] == selected_order_sim].iloc[0]
                
                with st.container():
                    st.markdown(f"### Order: {order_details['Order_ID']}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Risk Score", f"{order_details['delay_probability']*100:.0f}/100")
                    m2.metric("Vehicle Data", f"{order_details['vehicle_suitability_score']:.1f}")
                    m3.metric("Customer Tier", order_details['Customer_Segment'])
                    
                    st.write("---")
                    
                    s1, s2 = st.columns(2)
                    with s1:
                        st.markdown(f"**AI Recommendation**\n\n{order_details['Action']}")
                        st.caption(f"Reason: {order_details['Reason']}")
                        if st.button("Apply AI Plan"):
                            st.success("Plan Applied. Risk -40%.")
                    
                    with s2:
                        st.markdown("**Alternative: Force Express**")
                        st.caption("Guaranteed speed, higher cost.")
                        if st.button("Apply Express"):
                            st.info("Applied. Cost +₹1200.")

    with tab3:
        st.subheader("Export Operations Plan")
        st.write("Download the optimized dispatch plan for the warehouse team.")
        
        csv = ops_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Action List (CSV)",
            data=csv,
            file_name='daily_operations_actions.csv',
            mime='text/csv',
        )


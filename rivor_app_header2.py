import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# --- PAGE CONFIG AND CUSTOM CSS ---
st.set_page_config(page_title="RIVOR Method App", page_icon="🌊", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    h1 { color: #004080; font-size: 2.5em; font-weight: 700; }
    h3 { color: #0059b3; margin-bottom: 10px; }
    .stButton>button {
        background-color: #0066cc; color: white; border-radius: 12px; padding: 0.6em 1.2em;
        font-weight: bold; border: none;
    }
    .stButton>button:hover { background-color: #005bb5; }
    .stSelectbox label, .stNumberInput label { font-weight: bold; color: #003366; }
    .stDownloadButton>button {
        background-color: #00802b; color: white; border-radius: 10px;
        font-weight: bold; padding: 0.5em 1.2em;
    }
    .stDownloadButton>button:hover { background-color: #006622; }
    </style>
""", unsafe_allow_html=True)

# --- LOGO + HEADER ---
st.image("rivor_logo.png", width=120)  # Use your own RIVOR logo image here
st.title("🌊 RIVOR Method: Revised Ideal-Based VIKOR Ordering Tool")
st.write("Upload your dataset and apply the RIVOR ranking method with customizable parameters.")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📤 Upload Excel File (.xlsx only)", type=["xlsx"])

# --- FUNCTIONS ---
def read_and_average_excel(file):
    xls = pd.ExcelFile(file)
    sheet_names = [s for s in xls.sheet_names if s.lower() != "weights"]
    matrices = []
    for sheet in sheet_names:
        df = xls.parse(sheet, header=0)
        matrix = df.iloc[:, 1:].astype(float)
        matrices.append(matrix)
    avg_matrix = sum(matrices) / len(matrices)
    alt_names = df.iloc[:, 0]
    criteria = df.columns[1:]
    return alt_names, criteria, pd.DataFrame(avg_matrix.values, columns=criteria, index=alt_names)

def read_weights(file, criteria):
    xls = pd.ExcelFile(file)
    if "Weights" in xls.sheet_names:
        weights_df = xls.parse("Weights")
        return weights_df["Weight"].values
    else:
        st.sidebar.markdown("### 🎯 Enter Weights (No sheet found)")
        weights = []
        for col in criteria:
            w = st.sidebar.number_input(f"Weight for {col}", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key=f"w_{col}")
            weights.append(w)
        weights = np.array(weights)
        return weights / weights.sum()

def rivor_normalization(data, criterion_types, target_values):
    norm_data = pd.DataFrame(index=data.index, columns=data.columns)
    for i, col in enumerate(data.columns):
        col_data = data[col].astype(float)
        if criterion_types[i] == "Benefit":
            norm_data[col] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
        elif criterion_types[i] == "Cost":
            norm_data[col] = (col_data.max() - col_data) / (col_data.max() - col_data.min())
        elif criterion_types[i] == "Target":
            target = target_values[i]
            norm_data[col] = 1 - abs(col_data - target) / abs(col_data - target).max()
    return norm_data.astype(float)

def compute_rivor_scores(norm_df, weights, v=0.5):
    weighted_matrix = norm_df.values * weights
    Si = weighted_matrix.sum(axis=1)
    Ri = weighted_matrix.max(axis=1)
    S_best, S_worst = Si.min(), Si.max()
    R_best, R_worst = Ri.min(), Ri.max()
    Qi = v * (1 - (Si - S_best) / (S_worst - S_best + 1e-9)) + (1 - v) * (1 - (Ri - R_best) / (R_worst - R_best + 1e-9))
    result_df = pd.DataFrame({"S": Si, "R": Ri, "Q": Qi}, index=norm_df.index)
    result_df["Rank"] = result_df["Q"].rank(ascending=False).astype(int)
    return result_df.sort_values("Rank")

def to_excel(df_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for name, df in df_dict.items():
            df.to_excel(writer, sheet_name=name, index=True)
    output.seek(0)
    return output

# --- MAIN APP ---
if uploaded_file:
    alt_names, criteria, df_avg = read_and_average_excel(uploaded_file)

    st.subheader("Step 1: Define Criteria Type and Targets")
    criterion_types = []
    target_values = []
    for col in criteria:
        ctype = st.selectbox(f"Type for {col}", ["Benefit", "Cost", "Target"], key=f"type_{col}")
        criterion_types.append(ctype)
        if ctype == "Target":
            target = st.number_input(f"Target value for {col}", value=float(df_avg[col].mean()), key=f"target_{col}")
            target_values.append(target)
        else:
            target_values.append(None)

    weights = read_weights(uploaded_file, criteria)
    df_norm = rivor_normalization(df_avg, criterion_types, target_values)

    st.subheader("Step 2: View Matrices")
    st.write("**Average Matrix:**")
    st.dataframe(df_avg)
    st.write("**Normalized Matrix:**")
    st.dataframe(df_norm)

    st.subheader("Step 3: Compute RIVOR Score")
    v_value = st.slider("Compromise coefficient (v)", 0.0, 1.0, 0.5, step=0.05)
    scores_df = compute_rivor_scores(df_norm, weights=weights, v=v_value)

    st.write("**RIVOR Scores:**")
    st.dataframe(scores_df)

    top_percent = st.slider("Top % alternatives for compromise set", 1, 50, 10)
    top_n = int(len(scores_df) * top_percent / 100)
    compromise_df = scores_df.head(top_n)

    # DOWNLOAD
    result_excel = to_excel({
        "Average_Matrix": df_avg,
        "Normalized_Matrix": df_norm,
        "RIVOR_Scores": scores_df,
        "Compromise_Set": compromise_df
    })
    st.download_button("📥 Download All Results (.xlsx)", data=result_excel, file_name="rivor_results.xlsx")

    st.success("✅ Analysis complete. Download your results above.")

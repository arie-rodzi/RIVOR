
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="RIVOR Method App", layout="wide")
st.title("RIVOR Method: Revised Ideal-Based VIKOR Ordering Tool")

def read_and_average_excel(file):
    xls = pd.ExcelFile(file)
    sheet_names = [s for s in xls.sheet_names if s.lower() != "weights"]
    matrices = []
    for sheet in sheet_names:
        df = xls.parse(sheet, header=1)
        matrix = df.iloc[:, 1:].astype(float)
        matrices.append(matrix)
    average_matrix = sum(matrices) / len(matrices)
    alternatives = df.iloc[:, 0]
    criteria = df.columns[1:]
    return alternatives, criteria, average_matrix

def read_weights(file):
    xls = pd.ExcelFile(file)
    if "Weights" in xls.sheet_names:
        weights_df = xls.parse("Weights")
        return weights_df["Weight"].values
    return None

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
    return norm_data

def compute_rivor_scores(norm_df, weights=None, v=0.5):
    if weights is None:
        weights = np.ones(norm_df.shape[1]) / norm_df.shape[1]
    weighted_matrix = norm_df.values * weights
    Si = weighted_matrix.sum(axis=1)
    Ri = weighted_matrix.max(axis=1)
    S_best, S_worst = Si.min(), Si.max()
    R_best, R_worst = Ri.min(), Ri.max()
    Qi = v * (1 - (Si - S_best) / (S_worst - S_best + 1e-9)) +          (1 - v) * (1 - (Ri - R_best) / (R_worst - R_best + 1e-9))
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

# --- Streamlit App Workflow ---
st.markdown("### Step 1: Upload File")
uploaded_file = st.file_uploader("Upload Excel file (headers in second row)", type=["xlsx"])

if uploaded_file:
    alt_names, criteria, avg_matrix = read_and_average_excel(uploaded_file)
    df_avg = pd.DataFrame(avg_matrix.values, columns=criteria, index=alt_names)

    st.markdown("### Step 2: Define Criteria Type")
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

    df_norm = rivor_normalization(df_avg, criterion_types, target_values)

    st.markdown("### Step 3: Display Results")
    st.write("**Average Matrix:**")
    st.dataframe(df_avg)
    st.write("**Normalized Matrix:**")
    st.dataframe(df_norm)

    st.download_button("Download Matrices", data=to_excel({"Average_Matrix": df_avg, "Normalized_Matrix": df_norm}),
                       file_name="rivor_matrices.xlsx")

    st.markdown("### Step 4: RIVOR Scoring")
    v_value = st.slider("Select compromise weight (v)", 0.0, 1.0, 0.5, step=0.05)

    weight_array = read_weights(uploaded_file)
    scores_df = compute_rivor_scores(df_norm, weights=weight_array, v=v_value)
    st.dataframe(scores_df)

    st.download_button("Download RIVOR Scores", data=to_excel({"RIVOR_Scores": scores_df}),
                       file_name="rivor_scores.xlsx")

    st.markdown("### Step 5: Compromise Set")
    top_percent = st.slider("Top % alternatives to keep", 1, 50, 10)
    top_n = int(len(scores_df) * top_percent / 100)
    compromise_df = scores_df.head(top_n)
    st.dataframe(compromise_df)

    st.download_button("Download Compromise Set", data=to_excel({"Compromise_Set": compromise_df}),
                       file_name="rivor_compromise.xlsx")

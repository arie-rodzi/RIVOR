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
        df = xls.parse(sheet, header=0)
        matrix = df.iloc[:, 1:].astype(float)
        matrices.append(matrix)
    average_matrix = sum(matrices) / len(matrices)
    alternatives = df.iloc[:, 0]
    criteria = df.columns[1:]
    return alternatives, criteria, average_matrix

def read_weights(file, criteria):
    xls = pd.ExcelFile(file)
    if "Weights" in xls.sheet_names:
        weights_df = xls.parse("Weights")
        return weights_df["Weight"].values
    else:
        st.sidebar.markdown("### 🎯 Enter Weights")
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
            norm_data[col] = 1 - abs(col_data - target) / max(abs(target - col_data.min()), abs(target - col_data.max()))
    return norm_data.astype(float)

def compute_rivor_scores(norm_df, weights, alpha=0.5):
    one_minus_matrix = (1 - norm_df.values) * weights
    S = one_minus_matrix.sum(axis=1)
    R = one_minus_matrix.max(axis=1)
    S_best, S_worst = S.min(), S.max()
    R_best, R_worst = R.min(), R.max()
    Q = ((S - S_best) / (S_worst - S_best + 1e-9)) * alpha + ((R - R_best) / (R_worst - R_best + 1e-9)) * (1 - alpha)
    Q = 1 - Q  # Flip so higher is better
    result_df = pd.DataFrame({"S": S, "R": R, "Q": Q}, index=norm_df.index)
    result_df["Rank"] = result_df["Q"].rank(ascending=False).astype(int)
    return result_df.sort_values("Rank")

def to_excel(df_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for name, df in df_dict.items():
            df.to_excel(writer, sheet_name=name, index=True)
    output.seek(0)
    return output

# --- Streamlit App Interface ---
st.markdown("### Step 1: Upload Excel File")
uploaded_file = st.file_uploader("Upload Excel file (first row = header)", type=["xlsx"])

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

    weights = read_weights(uploaded_file, criteria)
    df_norm = rivor_normalization(df_avg, criterion_types, target_values)

    st.markdown("### Step 3: Display Matrices")
    st.write("**Average Matrix:**")
    st.dataframe(df_avg.style.format(precision=4))
    st.write("**Normalized Matrix:**")
    st.dataframe(df_norm.style.format(precision=4))

    st.download_button("Download Matrices", data=to_excel({
        "Average_Matrix": df_avg,
        "Normalized_Matrix": df_norm
    }), file_name="rivor_matrices.xlsx")

    st.markdown("### Step 4: Compute RIVOR Scores")
    alpha = st.slider("Select compromise weight (α)", 0.0, 1.0, 0.5, step=0.05)
    scores_df = compute_rivor_scores(df_norm, weights, alpha=alpha)
    st.dataframe(scores_df.style.format(precision=4))

    st.download_button("Download RIVOR Scores", data=to_excel({"RIVOR_Scores": scores_df}),
                       file_name="rivor_scores.xlsx")

    st.markdown("### Step 5: Compromise Set")
    top_percent = st.slider("Top % alternatives to keep", 1, 50, 10)
    top_n = max(1, int(len(scores_df) * top_percent / 100))
    compromise_df = scores_df.head(top_n)
    st.dataframe(compromise_df)

    st.download_button("Download Compromise Set", data=to_excel({"Compromise_Set": compromise_df}),
                       file_name="rivor_compromise.xlsx")

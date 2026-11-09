
import streamlit as st
import pandas as pd
import joblib

# Memuat model yang sudah dilatih
model = joblib.load("model_joblib.pkl")

# Judul aplikasi
st.title("Prediksi Tagihan Listrik Jakarta")
st.write(
    "Aplikasi untuk memprediksi jumlah tagihan listrik "
    "berdasarkan parameter yang diberikan."
)

# Input pengguna
st.sidebar.header("Input Parameter")

def user_input_features():
    kwh = st.sidebar.slider("Konsumsi KWH (kWh)", 150.0, 600.0, 350.0)
    ac_units = st.sidebar.slider("Jumlah AC", 0, 3, 1)
    ac_hours_per_day = st.sidebar.slider("Jam AC per Hari", 0.0, 10.0, 5.0)
    family_size = st.sidebar.slider("Jumlah Anggota Keluarga", 2, 6, 4)

    month_name = st.sidebar.selectbox(
        "Bulan",
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )

    tariff_class = st.sidebar.selectbox(
        "Kelas Tarif", ["R1", "R2", "R3"]
    )

    data = {
        "kwh": kwh,
        "ac_units": ac_units,
        "ac_hours_per_day": ac_hours_per_day,
        "family_size": family_size,
        "month_name": month_name,
        "tariff_class": tariff_class
    }

    return pd.DataFrame(data, index=[0])


df_input = user_input_features()

st.subheader("Parameter Input Pengguna:")
st.write(df_input)

# Kolom yang digunakan saat pelatihan model
training_columns = [
    "kwh",
    "ac_units",
    "ac_hours_per_day",
    "family_size",
    "month_name_Aug",
    "month_name_Dec",
    "month_name_Feb",
    "month_name_Jan",
    "month_name_Jul",
    "month_name_Jun",
    "month_name_Mar",
    "month_name_May",
    "month_name_Nov",
    "month_name_Oct",
    "month_name_Sep",
    "tariff_class_R2",
    "tariff_class_R3"
]

# Membuat data input dengan kolom yang sesuai
final_input_df = pd.DataFrame(0, index=[0], columns=training_columns)

# Mengisi fitur numerik
final_input_df["kwh"] = df_input["kwh"]
final_input_df["ac_units"] = df_input["ac_units"]
final_input_df["ac_hours_per_day"] = df_input["ac_hours_per_day"]
final_input_df["family_size"] = df_input["family_size"]

# One-hot encoding bulan
selected_month_col = "month_name_" + df_input["month_name"].iloc[0]

if selected_month_col in final_input_df.columns:
    final_input_df[selected_month_col] = 1

# One-hot encoding kelas tarif
selected_tariff_col = "tariff_class_" + df_input["tariff_class"].iloc[0]

if selected_tariff_col in final_input_df.columns:
    final_input_df[selected_tariff_col] = 1

# Tombol prediksi
if st.sidebar.button("Prediksi Tagihan"):

    try:
        prediction = model.predict(final_input_df)

        st.subheader("Hasil Prediksi Tagihan Listrik:")
        st.success(f"Tagihan Diprediksi: Rp {prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

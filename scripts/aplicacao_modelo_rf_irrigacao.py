import streamlit as st
import pandas as pd
import joblib

# =============================================================================
# CARREGAMENTO
# =============================================================================

modelo = joblib.load("model/modelo_irrigacao.pkl")
features_treinamento = joblib.load("model/features.pkl")

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

st.set_page_config(
    page_title="Predição de Irrigação",
    page_icon="💦🌱",
    layout="centered"
)

st.markdown(
    """
    <style>
    /* Altera a largura máxima do container principal do Streamlit */
    .block-container {
        max-width: 1000px; /* Altere aqui para a largura desejada (ex: 950px, 1100px) */
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Predição da Necessidade de Irrigação")

st.markdown("""
Informe as condições ambientais e de manejo para estimar a necessidade de irrigação.
""")

# =============================================================================
# ENTRADAS (Organizadas em Colunas)
# =============================================================================

col1, col2 = st.columns(2)

with col1:
    soil_moisture = st.slider(
        "Umidade do Solo (%)",
        min_value=8,
        max_value=65,
        value=40
    )

    temperature = st.slider(
        "Temperatura (°C)",
        min_value=0.0,
        max_value=50.0,
        value=25.0
    )

    crop_stage = st.selectbox(
        "Estágio da Cultura",
        [
            "Planting",
            "Vegetative",
            "Flowering",
            "Harvesting"
        ]
    )

with col2:
    rainfall = st.slider(
        "Precipitação (mm)",
        min_value=0.0,
        max_value=300.0,
        value=20.0
    )

    wind_speed = st.slider(
        "Velocidade do Vento (km/h)",
        min_value=0.0,
        max_value=80.0,
        value=10.0
    )

    mulching = st.selectbox(
        "Uso de Mulching",
        ["No", "Yes"]
    )

# =============================================================================
# PREPARAÇÃO DOS DADOS
# =============================================================================

entrada = pd.DataFrame({
    "Soil_Moisture": [soil_moisture],
    "Temperature_C": [temperature],
    "Rainfall_mm": [rainfall],
    "Wind_Speed_kmh": [wind_speed]
})

crop_dummy = pd.get_dummies(
    pd.Series([crop_stage]),
    prefix="Crop_Stage"
)

mulching_dummy = pd.get_dummies(
    pd.Series([mulching]),
    prefix="Mulching",
    drop_first=True
)

entrada = pd.concat(
    [
        entrada,
        crop_dummy,
        mulching_dummy
    ],
    axis=1
)

# garante mesmas colunas do treinamento

entrada = entrada.reindex(
    columns=features_treinamento,
    fill_value=0
)

# =============================================================================
# PREDIÇÃO
# =============================================================================

if st.button("Realizar Predição"):

    predicao = modelo.predict(entrada)[0]

    probabilidades = modelo.predict_proba(entrada)[0]

    classes = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    st.success(
        f"Necessidade de Irrigação: {classes[predicao]}"
    )

    st.subheader("Probabilidades")

    prob_df = pd.DataFrame({
        "Classe": ["Low", "Medium", "High"],
        "Probabilidade": probabilidades
    })

    st.bar_chart(
        prob_df.set_index("Classe")
    )

    st.dataframe(
        prob_df.style.format({
            "Probabilidade": "{:.2%}"
        })
    )
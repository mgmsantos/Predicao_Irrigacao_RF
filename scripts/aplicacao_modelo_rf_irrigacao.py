import streamlit as st
import pandas as pd
import joblib

# =============================================================================
# CARREGAMENTO
# =============================================================================

modelo = joblib.load("model/modelo_irrigacao.pkl")
features_treinamento = joblib.load("model/features.pkl")

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Predição da Necessidade de Irrigação",
    page_icon="💦🌱",
    layout="centered"
)

st.markdown(
    """
    <style>

    .block-container{
        max-width:1000px;
        padding-top:2rem;
        padding-bottom:2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# TÍTULO
# =============================================================================

st.title("💦🌱 Predição da Necessidade de Irrigação")

st.markdown(
"""
Informe as condições ambientais e de manejo da cultura para estimar a necessidade de irrigação utilizando um modelo Random Forest.
"""
)

# =============================================================================
# ENTRADAS
# =============================================================================

col1, col2 = st.columns(2)

with col1:

    soil_moisture = st.slider(
        "Umidade do Solo (%)",
        min_value=8.0,
        max_value=65.0,
        value=40.0
    )

    temperature = st.slider(
        "Temperatura do Ar (°C)",
        min_value=0.0,
        max_value=50.0,
        value=25.0
    )

    estagios = {
        "Semeadura": "Sowing",
        "Vegetativo": "Vegetative",
        "Florescimento": "Flowering",
        "Colheita": "Harvest"
    }

    # Usuário escolhe em português
    crop_stage_pt = st.selectbox(
        "Estágio da Cultura",
        list(estagios.keys())
    )

    # Modelo recebe em inglês
    crop_stage = estagios[crop_stage_pt]

with col2:

    rainfall = st.slider(
        "Precipitação Acumulada (mm)",
        min_value=0.0,
        max_value=2500.0,
        value=1200.0,
        step=10.0
    )

    wind_speed = st.slider(
        "Velocidade do Vento (km/h)",
        min_value=0.0,
        max_value=20.0,
        value=5.0
    )

    mulching = st.selectbox(
        "Uso de Mulching",
        [
            "No",
            "Yes"
        ]
    )

# =============================================================================
# PREPARAÇÃO DOS DADOS
# =============================================================================

# Cria um DataFrame contendo exatamente as mesmas colunas
# utilizadas durante o treinamento

entrada = pd.DataFrame(
    0.0,
    index=[0],
    columns=features_treinamento
)

# -----------------------------
# Variáveis Numéricas
# -----------------------------

entrada.loc[0, "Soil_Moisture"] = soil_moisture
entrada.loc[0, "Temperature_C"] = temperature
entrada.loc[0, "Rainfall_mm"] = rainfall
entrada.loc[0, "Wind_Speed_kmh"] = wind_speed

# -----------------------------
# Estágio da Cultura
# -----------------------------

mapa_estagios = {
    "Sowing": "Crop_Stage_Sowing",
    "Vegetative": "Crop_Stage_Vegetative",
    "Flowering": "Crop_Stage_Flowering",
    "Harvest": "Crop_Stage_Harvest"
}

coluna_stage = mapa_estagios[crop_stage]

if coluna_stage in entrada.columns:
    entrada.loc[0, coluna_stage] = 1

# -----------------------------
# Mulching
# -----------------------------

entrada.loc[0, "Mulching_Yes"] = (
    1 if mulching == "Yes" else 0
)

# Garante exatamente a mesma ordem das colunas utilizadas no treinamento

entrada = entrada.reindex(
    columns=features_treinamento,
    fill_value=0
)

# =============================================================================
# PREDIÇÃO
# =============================================================================

if st.button("Realizar Predição", use_container_width=True):

    predicao = modelo.predict(entrada)[0]

    probabilidades = modelo.predict_proba(entrada)[0]

    classes = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    mensagens = {
        "Low": "🟢 Baixa necessidade de irrigação",
        "Medium": "🟡 Necessidade moderada de irrigação",
        "High": "🔴 Alta necessidade de irrigação"
    }

    st.success(mensagens[classes[predicao]])

    # ==========================================================
    # PROBABILIDADES
    # ==========================================================

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
        }),
        use_container_width=True
    )

    # ==========================================================
    # INTERPRETAÇÃO
    # ==========================================================

    st.subheader("Interpretação")

    if predicao == 0:

        st.info(
        """
        As condições informadas indicam elevada disponibilidade hídrica,
        reduzindo a necessidade de irrigação neste momento.
        """
        )

    elif predicao == 1:

        st.warning(
        """
        As condições ambientais sugerem uma necessidade intermediária
        de irrigação. Recomenda-se monitoramento da umidade do solo
        antes da tomada de decisão.
        """
        )

    else:

        st.error(
        """
        As condições ambientais indicam elevada demanda hídrica,
        sugerindo alta necessidade de irrigação.
        """
        )

    # ==========================================================
    # DADOS ENVIADOS AO MODELO
    # ==========================================================

    with st.expander("Visualizar dados enviados ao modelo"):

        st.dataframe(
            entrada.T.rename(columns={0: "Valor"}),
            use_container_width=True
        )
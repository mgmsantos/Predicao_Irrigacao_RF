# Predição da Necessidade de Irrigação com Random Forest


## Visão Geral
Este projeto prevê as necessidades de irrigação das culturas (baixa média e alta) utilizando variáveis (features) ​​ambientais e de manejo.

O objetivo foi desenvolver um modelo de aprendizado de máquina interpretável capaz de auxiliar na tomada de decisões de manejo da irrigação com base nas condições edafoclimáticas e no manejo da cultura.

## Conjunto de Dados
O dataset contêm 10000 observações e inclui as seguintes features:

- Soil_Type
- Soil_pH
- Soil_Moisture
- Organic_Carbon
- Electrical_Conductivity
- Temperature_C
- Humidity
- Rainfall_mm
- Sunlight_Hours
- Wind_Speed_kmh
- Crop_Type
- Crop_Growth_Stage
- Season
- Irrigation_Type
- Water_Source
- Field_Area_hectare
- Mulching_Used
- Previous_Irrigation_mm
- Region
- Irrigation_Need

## Metodologia

### 1. Seleção Inicial de Features
O processo de modelagem começou utilizando todas as variáveis disponíveis no conjunto de dados.

Um modelo inicial de Random Forest foi treinado usando o conjunto completo de features para avaliar sua importância individual.

A análise das importâncias de cada feature indicou que:

- A umidade do solo (Soil_Moisture) capturou grande parte da informação contida em outras variáveis do dataset, tornando-as redundantes;
- Features como: Soil_pH, Organic_Carbon, 'Region', 'Water_Source', entre outras, contribuiam muito pouco, e sua adição fazia o modelo criar regras muito específicas;
- Variáveis meteorológicas como Temperature_C, Rainfall_mm, Wind_Speed_kmh contribuiram significamente na construção do modelo, visto que fazem parte da definição de parâmetros agronômicos considerados na determinação da evapotranspiração de cultura e na decisão de irrigar;
- O estágio de desenvolvimento da cultura e o uso de mulching são features que influenciam bastante na previsão de irrigação, tornando-as peça fundamental na construção do modelo.

Com base nisso, foram selecionadas as características preditoras listadas abaixo:

- Soil_Moisture
- Rainfall_mm
- Temperature_C
- Wind_Speed_kmh
- Mulching_Used
- Crop_Growth_Stage

### 2. Preparação dos dados
A variável-resposta Irrigation_Need foi transformadas em nível de irrigação, correspondendo a 0 (Low), 1 (Medium) e High (2).

As variáveis categóricas Mulching_Used e Crop_Growth_Stage foram transformadas usando One-Hot-Encoding, sendo que para Mulching_Used foi usado o argumento ```drop_first = True``` para remover a redundância. Crop_Growth_Stage utilizou ```drop_first = False``` visto que não correspondia a uma variável categórica binária, e sim categórica nominal.

### 3. Desenvolvimento do modelo
Foram destinados 70% do dataset para o treinamento do modelo, com estratificação da variável-resposta visto que não havia balanceamento dos dados para cada categoria de necessidade de irrigação (linhas com necessidade 'High' representavam apenas 3.36% do conjunto de dados, enquanto 'Medium' 38% e 'Low', 58.64%).

Uma Random Forest foi inicialmente treinada para obter o melhor valor de ```max_depth```, o qual correspondeu a 6, em que oferencia um bom balanceamento do modelo, evitando o underfitting e overfitting.

### 4. Avaliação do Modelo
O modelo foi avaliado considerando as seguintes análises:

- Matriz de Confusão (Confusion Matrix)
- Relatório de Classificação (Classification Report)
- Curva ROC-AUC (Macro)
- Validação Cruzada Estratificada (Stratified Cross Validation)
- Curva de Aprendizado do Modelo (Model Learning Curve)

### 5. Explicabilidade
A interpretabilidade do modelo foi avaliada por meio das seguintes análises:
- Permutation Importance
- SHAP Beeswarm Plot
- SHAP Waterfall Plot

### 6. Tecnologias
Em todo o desenvolvimento do modelo foram utilizadas as seguinte tecnologias:
- Python 3.14.5
- Sckit-Learn
- Pandas
- Numpy
- Matplotlib
- SHAP

## Resultados
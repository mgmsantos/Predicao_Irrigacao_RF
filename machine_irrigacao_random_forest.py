# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.inspection import permutation_importance
import shap

# %%
dados_irrigacao = pd.read_csv(r"C:\Users\migue\Downloads\archive\irrigation_prediction.csv")
dados_irrigacao.shape # 10 mil linhas e 20 colunas

# %%
# --- SELEÇÃO DE FEATURES E PREPARAÇÃO ---
# Excelente decisão agronômica: umidade e temperatura já absorvem os efeitos das variáveis removidas
dados_irrigacao = dados_irrigacao[[
    'Soil_Moisture',
    'Temperature_C',
    'Rainfall_mm',
    'Wind_Speed_kmh',
    'Crop_Growth_Stage',
    'Mulching_Used',
    'Irrigation_Need'
]].copy()

niveis_irrigacao = {
    'Low': 0,
    'Medium': 1,
    'High': 2
}
dados_irrigacao['Irrigation_Need_Class'] = dados_irrigacao['Irrigation_Need'].map(niveis_irrigacao)

# Features numéricas
col_num = dados_irrigacao[['Soil_Moisture', 'Temperature_C', 'Rainfall_mm', 'Wind_Speed_kmh']].copy()

# Processamento Crop_Growth_Stage
col_crop_stage_dummy = pd.get_dummies(dados_irrigacao['Crop_Growth_Stage'], prefix='Crop_Stage', drop_first=False)

# Processamento Mulching
col_mulching_dummy = pd.get_dummies(dados_irrigacao['Mulching_Used'], prefix='Mulching', drop_first=True)

# Concatenação das features
dados_processado = pd.concat([col_num, col_crop_stage_dummy, col_mulching_dummy], axis=1)

# Garantia de transformação numérica
dados_processado = dados_processado.astype(float)

dados_processado['Irrigation_Need'] = dados_irrigacao['Irrigation_Need']
dados_processado['Irrigation_Need_Class'] = dados_irrigacao['Irrigation_Need_Class']

# Separação de features e variável resposta
y = dados_processado['Irrigation_Need_Class']
X = dados_processado.drop(columns = ['Irrigation_Need', 'Irrigation_Need_Class'])

# Separação de treino e teste (30% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.3,
    stratify = y,
    random_state = 42
)

# --- BUSCA DO MAX_DEPTH IDEAL ---
notas_treino = []
notas_teste = []
profundidades = range(1, 20)

for depth in profundidades:
    model = RandomForestClassifier(
        n_estimators = 100,
        max_depth = depth,
        class_weight = 'balanced',
        random_state = 42,
        n_jobs = -1
    )
    model.fit(X_train, y_train)
    notas_treino.append(model.score(X_train, y_train))
    notas_teste.append(model.score(X_test, y_test))

plt.figure(figsize = (10, 6))
plt.plot(profundidades, notas_treino, label = 'Treino', marker = 'o')
plt.plot(profundidades, notas_teste, label = 'Teste', marker = 'o')
plt.xlabel('Profundidade Máxima (max_depth)')
plt.ylabel('Acurácia')
plt.xlim(0, 20)
plt.xticks(range(0, 20, 2))
plt.title('Obtendo max_depth ideal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- MODELO FINAL RANDOM FOREST ---
# Configurado com o max_depth=7 escolhido com base no platô do gráfico anterior
random_forest = RandomForestClassifier(
    n_estimators = 100,
    max_depth = 6,
    class_weight = 'balanced',
    random_state = 42,
    n_jobs = -1 # OTIMIZAÇÃO: Usa todos os núcleos do seu processador para treinar mais rápido
)
random_forest.fit(X_train, y_train)

# %%
# --- AVALIAÇÃO CONVENCIONAL ---
previsoes_test = random_forest.predict(X_test)
print("--- Matriz de Confusão Test ---")
print(confusion_matrix(y_test, previsoes_test))

print("\n--- Relatório de Classificação Test ---")
print(classification_report(y_test, previsoes_test, target_names = ['Low', 'Medium', 'High']))

# %%
previsoes_train = random_forest.predict(X_train)
print("\n--- Matriz de Confusão Train ---")
print(confusion_matrix(y_train, previsoes_train))

print("\n--- Relatório de Classificação Train ---")
print(classification_report(y_train, previsoes_train, target_names = ['Low', 'Medium', 'High']))

# %%
# --- VALIDAÇÃO CRUZADA ---
cv = StratifiedKFold(
    n_splits = 5,
    shuffle = True,
    random_state = 42
)

scores = cross_val_score(
    random_forest,
    X,
    y,
    cv = cv,
    scoring = 'f1_macro',
    n_jobs = -1
)
print("\n--- Validação Cruzada (F1 Macro) ---")
print(f"Scores: {scores}")
print(f"Média: {scores.mean():.4f}")
print(f"Desvio Padrão: {scores.std():.4f}")

# --- CURVA DE APRENDIZADO ---
train_sizes, train_scores, test_scores = learning_curve(
    random_forest,
    X,
    y,
    cv = cv, # CORREÇÃO 2: Passando a estratégia de validação cruzada estratificada aqui também
    scoring = 'f1_macro',
    n_jobs = -1,
    train_sizes = np.linspace(0.1, 1.0, 10)
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_mean, marker='o', label='Treino')
plt.plot(train_sizes, test_mean, marker='o', label='Validação')
plt.xlabel('Número de amostras de treinamento')
plt.ylabel('F1 Macro')
plt.title('Curva de Aprendizado')
plt.legend()
plt.grid(True)
plt.show()

# --- PERMUTATION IMPORTANCE ---
resultado = permutation_importance(
    random_forest,
    X_test,
    y_test,
    n_repeats = 10,
    random_state = 42,
    n_jobs = -1
)

df_importancia = pd.DataFrame({
    'Variavel': X.columns,
    'Importancia': resultado.importances_mean
}).sort_values('Importancia', ascending=False)

print("\n--- Permutation Importance ---")
print(df_importancia)

# --- ROC-AUC ---
y_prob = random_forest.predict_proba(X_test)
roc_auc = roc_auc_score(
    y_test,
    y_prob,
    multi_class = 'ovr',
    average = 'macro'
)
print(f"\nROC-AUC Macro: {roc_auc:.4f}")

# --- SHAP EXPLAINER ---
explicador = shap.TreeExplainer(random_forest)

# CORREÇÃO 3: Selecionando uma amostra de 300 linhas para o SHAP não travar o computador
X_amostra = X_test.sample(n=300, random_state=42) if len(X_test) > 300 else X_test
valores_shap = explicador(X_amostra)

# Gráfico Global (Beeswarm) para a classe High (índice 2)
plt.figure(figsize = (10, 6))
shap.plots.beeswarm(valores_shap[:, :, 2])
plt.show()

# Gráfico Local (Waterfall)
# Como usamos sample(), vamos pegar o primeiro elemento (índice 0) dessa amostra para plotar a cascata
plt.figure(figsize=(10, 4))
shap.plots.waterfall(valores_shap[0, :, 2])
plt.show()
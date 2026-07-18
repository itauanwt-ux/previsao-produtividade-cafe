# Previsão de Produtividade do Café com Machine Learning

Projeto de regressão para prever a produtividade de lavouras de café
(sacas/hectare) a partir de variáveis agronômicas: chuva, temperatura,
adubação, altitude e idade da lavoura.

> **Nota sobre os dados:** base **sintética**, gerada em Python a partir de
> relações agronômicas conhecidas (ex.: café se beneficia de maior altitude e
> chuva regular, mas sofre com temperaturas muito altas). O objetivo é
> demonstrar o fluxo completo de um projeto de ML aplicado ao agro — dos dados
> ao modelo — podendo ser adaptado facilmente para dados reais de cooperativas.

## Pergunta de negócio
É possível prever a produtividade de uma lavoura de café com base em variáveis
climáticas e de manejo? Quais fatores mais influenciam o resultado?

## O que foi feito
1. Geração da base `produtividade_cafe.csv` (500 lavouras simuladas).
2. Divisão treino/teste (80/20).
3. Comparação de dois modelos:
   - Regressão Linear (baseline)
   - Random Forest Regressor
4. Avaliação com **MAE** (erro médio absoluto) e **R²**.
5. Análise de importância das variáveis.

## Resultados
| Modelo | MAE (sacas/ha) | R² |
|---|---|---|
| Regressão Linear | 4,28 | 0,493 |
| Random Forest | 4,26 | 0,515 |

**Importância das variáveis (Random Forest):**
1. Chuva anual (48,5%)
2. Adubação (19,2%)
3. Temperatura média (17,6%)
4. Altitude (11,3%)
5. Idade da lavoura (3,4%)

## Principal insight
A **chuva anual** é, de longe, o fator mais determinante da produtividade,
respondendo por quase metade da capacidade preditiva do modelo — à frente até
da adubação. Isso reforça a importância de monitoramento climático e irrigação
de apoio em anos de seca para cooperativas que dependem da produção de café.

## Como reproduzir
```bash
pip install pandas scikit-learn matplotlib
python gerar_dados.py   # gera a base
python modelo.py        # treina os modelos e gera os gráficos
```

## Arquivos
- `gerar_dados.py` — geração da base sintética
- `modelo.py` — treinamento, avaliação e importância de variáveis
- `produtividade_cafe.csv` — base de dados
- `importancia_variaveis.png`, `real_vs_previsto.png`

## Próximos passos
- Testar com dados reais de produtividade da cooperativa (anonimizados).
- Incluir variáveis de solo e histórico de pragas.
- Testar modelos como XGBoost para comparação.

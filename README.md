# Fenotipado y Predicción de Severidad en Datos de Alta Dimensionalidad

Este proyecto implementa un flujo completo de análisis en Python para datos de alta dimensionalidad, combinando técnicas de aprendizaje no supervisado y supervisado con el objetivo de identificar subgrupos latentes (fenotipos) y predecir severidad clínica.

En la práctica clínica, muchos síndromes crónicos presentan una alta heterogeneidad biológica y clínica, lo que dificulta tanto la predicción de severidad como la respuesta al tratamiento. En este proyecto se simula un escenario clínico donde se plantea que un síndrome crónico complejo puede dividirse en distintos fenotipos, los cuales podrían aportar información adicional para la predicción de severidad.
El objetivo principal es integrar métodos no supervisados y supervisados para:
Identificar fenotipos clínicos latentes.
Evaluar si dichos fenotipos mejoran la predicción de severidad.
Aplicar modelos modernos de machine learning en un contexto clínico simulado.

---

## Preparación del dataset

Se simula un conjunto de datos con:

- 600 observaciones
- 15 variables continuas (Feature_1 a Feature_15), representando biomarcadores y síntomas.
- Un outcome binario:
  - Severidad = 1 → Alta
  - Severidad = 0 → Baja
El dataset fue diseñado con separación moderada entre clases para representar un escenario clínico realista, donde la predicción no es trivial.
---

## Preparación de los datos

### Escalado y Reducción de Dimensionalidad (PCA)
Las 15 variables predictoras son escaladas mediante `StandardScaler`, paso obligatorio para métodos basados en distancia y varianza.
Posteriormente se aplico analisis de componentes principales PCA reduciendo la dimensionalidad a dos componentes con fines exploratorios y de visulización

---

## Fenotipado no supervisado

### Análisis de Componentes Principales (PCA)
Se aplica PCA reduciendo las 15 variables originales a dos componentes principales (PC1 y PC2) para visualización de la estructura del espacio de datos.

Varianza explicada:
- PC1: 0.212  
- PC2: 0.156  
- PC1 + PC2: 0.368
- 
Esto indica que el plano PC1–PC2 captura cerca del 37% de la variabilidad total, lo cual es adecuado para visualización, aunque no sustituye el uso de todas las variables originales para modelado.


### PCA según Severidad
Visualización del espacio PC1–PC2 coloreado según el outcome real de severidad.

![PCA según Severidad](img/pca_severidad.png)

### Clustering con K-Means
Se aplicó K-Means con 3 clusters sobre los datos escalados, identificando tres fenotipos clínicos potenciales.
La distribución de los fenotipos fue razonablemente balanceada, lo que sugiere que el algoritmo logró segmentar la población sin colapsar en clusters triviales.
Estos fenotipos representan posibles subgrupos clínicos con características latentes distintas, no definidas explícitamente por la severidad.

Distribución de observaciones:
- Fenotipo 0: 136  
- Fenotipo 1: 293  
- Fenotipo 2: 171  

### PCA según Fenotipo
Proyección de los tres fenotipos identificados en el plano PC1–PC2.

![PCA según Fenotipo](img/pca_fenotipos_kmeans.png)

---

## Modelado supervisado

### Preparación
Se construye un modelo predictivo incorporando:
- Las 15 features originales
- La variable `Fenotipo`

Los datos se dividen en:
- 70% entrenamiento
- 30% prueba

### Random Forest
Se entrena un modelo `RandomForestClassifier` con 300 árboles.
El modelo Random Forest Classifier para predecir severidad utilizando:
Las 15 variables originales
El fenotipo asignado por K-Means como predictor adicional
División de datos:
70% entrenamiento
30% prueba
Estratificación por severidad

### Evaluación
El rendimiento del modelo se evalúa mediante el Área Bajo la Curva ROC (AUC) en el conjunto de prueba.

AUC final: **0.999**

Este desempeño indica una capacidad discriminativa excelente, esperable en un contexto de datos simulados donde se combinan múltiples variables informativas y fenotipado previo.

![Curva ROC Random Forest](img/roc_random_forest.png)

### Importancia de variables

El análisis de importancia de variables del Random Forest mostró que:
- Varias features originales aportan información relevante.
- El fenotipo aparece entre las variables más importantes, indicando que el fenotipado no supervisado agrega valor predictivo más allá de las variables individuales.
Esto sugiere que la combinación de enfoques no supervisados y supervisados es útil para mejorar la predicción clínica.

---
## Preguntas Tarea: 

## 1. Interpretación del Fenotipado
A partir del análisis de Componentes Principales (PCA), se observó que las dos primeras componentes explican aproximadamente el 36.8% de la varianza total del dataset (PC1 = 21.2%, PC2 = 15.6%). Esto indica que, si bien el plano PC1–PC2 no captura la totalidad de la información contenida en las 15 variables originales, sí resume una proporción relevante de la estructura latente del síndrome, permitiendo una visualización exploratoria de patrones globales.

La aplicación de K-Means (k = 3) permitió identificar tres fenotipos clínicos que no coinciden necesariamente con la clasificación binaria de severidad. En la planificación de un ensayo clínico, la existencia de estos fenotipos es relevante porque permite:
- Estratificar la aleatorización por fenotipo, reduciendo heterogeneidad basal.
- Evaluar modificación de efecto (interacción tratamiento × fenotipo).
- Diseñar análisis de subgrupos con foco en medicina personalizada, donde distintos fenotipos podrían beneficiarse de estrategias terapéuticas diferenciadas.

## 2. El rol de MLP (redes neuronales)

En este proyecto, el Random Forest obtuvo un AUC superior en la predicción de severidad (AUC ≈ 0.999), lo cual es esperable en datos tabulares y estructurados. Sin embargo, un bioestadístico aún consideraría Redes Neuronales (MLP) en proyectos de salud de alta complejidad por razones clave:
- Datos no tabulares: Las redes neuronales son especialmente adecuadas para imágenes médicas (RX, TAC, RM), texto clínico (epicrisis, notas) y señales (ECG/EEG), donde los modelos basados en árboles no suelen ser la primera opción.
- Escalado obligatorio: Los MLP requieren estandarización/normalización de variables para un entrenamiento estable, lo que favorece representaciones numéricas homogéneas y el aprendizaje de relaciones no lineales complejas.
- Modelos multimodales: Permiten integrar simultáneamente datos clínicos estructurados + imagen + texto, una necesidad frecuente en sistemas de apoyo a la decisión clínica.

En primer lugar, las redes neuronales son especialmente adecuadas para datos no tabulares, como imágenes médicas (radiografías, TAC, resonancia magnética), texto clínico (epicrisis, notas de evolución) o señales fisiológicas (ECG, EEG), donde los modelos basados en árboles no pueden aplicarse directamente. En segundo lugar, los MLP requieren un escalado explícito de las variables, lo que favorece una representación numérica homogénea y permite el aprendizaje de relaciones no lineales complejas en espacios de alta dimensionalidad.
Por último, las arquitecturas neuronales permiten extender el análisis hacia modelos multimodales, integrando simultáneamente datos clínicos estructurados, imágenes y texto, algo indispensable en sistemas modernos de apoyo a la decisión clínica. Por estas razones, aunque el Random Forest sea superior en este escenario específico, las redes neuronales siguen siendo herramientas clave en salud.

## 3. Desafio ético y sesgo algoritmico ## 

Si una de las 15 features estuviera altamente correlacionada con el nivel socioeconómico (NSE), existiría riesgo de sesgo algorítmico: el modelo podría aprender asociaciones que perjudiquen sistemáticamente a ciertos grupos (por ejemplo, sobreestimar severidad en NSE bajo), perpetuando inequidades en decisiones de priorización o acceso.
Antes de desplegar el modelo, tomaría al menos estas precauciones:
- Auditoría de equidad: evaluar desempeño por subgrupos (por NSE u otros determinantes), comparando sensibilidad/especificidad/AUC estratificados.
- Interpretabilidad y contribución: cuantificar el aporte de esa variable (importancia, y idealmente métodos como SHAP) y verificar si su exclusión cambia sustantivamente el rendimiento.
- Gobernanza y monitoreo: documentar el riesgo, definir criterios de uso (apoyo, no reemplazo), y monitorear drift y sesgos en producción.

El objetivo no es solo maximizar AUC, sino asegurar un uso justo, transparente y responsable del modelo.

## 4. Reflexión y conclusión

Este proyecto evidencia el valor de pasar desde la Regresión Logística (enfoque inferencial) hacia un enfoque moderno de fenotipado y predicción con ensamblajes. El fenotipado no supervisado (K-Means) permite identificar subgrupos clínicos latentes que capturan heterogeneidad no visible en la severidad binaria, mejorando la estratificación y la planificación de ensayos clínicos (incluyendo evaluación de interacción tratamiento×fenotipo). Al incorporar el fenotipo en un modelo de Random Forest, se obtuvo una capacidad discriminativa sobresaliente (AUC ≈ 0.999) en un escenario simulado. En conjunto, este enfoque fortalece la toma de decisiones en epidemiología al integrar exploración de estructura latente, predicción robusta y consideraciones éticas para un despliegue responsable


# ENTREGA 4 - SUBE-G17
## Ciencia de Datos - 5k9 - 2025
## Integrantes:
* Franco Veggiani
* Juan Ignacio Diaz

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de predicción de pasajeros para líneas de colectivo utilizando datos históricos de transporte público, condiciones climáticas y características temporales. La aplicación incluye visualizaciones interactivas con Altair y una interfaz para realizar predicciones en tiempo real.

## 🚀 Características Implementadas

### Visualizaciones Interactivas (Altair)
1. **Evolución Temporal**: Gráfico de línea mostrando la evolución del promedio mensual de pasajeros a lo largo del tiempo
2. **Patrones por Día de la Semana**: Análisis de demanda promedio por día, mostrando diferencias entre días laborales y fines de semana
3. **Impacto del Clima**: Visualización de la relación entre condiciones climáticas (temperatura y precipitación) y demanda de pasajeros
4. **Top 10 Líneas**: Comparación de las líneas con mayor promedio de pasajeros transportados

### Funcionalidades de la App
- **Exploración de Datos**: Sección dedicada con visualizaciones interactivas y resumen estadístico
- **Predicción de Pasajeros**: Interfaz para ingresar datos y obtener predicciones del modelo entrenado
- **Navegación por Tabs**: Organización clara entre exploración y predicción

## 📁 Estructura del Proyecto

```
sube-G17/
├── app.py                          # Aplicación Streamlit principal
├── requirements.txt                 # Dependencias del proyecto
├── README.md                       # Este archivo
├── .gitignore                      # Archivos ignorados por Git
├── final_2024-11-04.csv            # Dataset utilizado
├── IGNA_Entrega3_DiazVeggiani.ipynb # Notebook de entrenamiento
└── artifacts/                      # Artefactos del modelo entrenado
    ├── fe_pipeline.joblib          # Pipeline de feature engineering
    ├── preprocessor.joblib         # Pipeline de preprocesamiento
    ├── model.joblib                # Modelo entrenado
    └── metadata.json               # Metadatos del modelo
```

## 🛠️ Instalación y Uso Local

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio** (o descargar los archivos):
```bash
git clone <url-del-repositorio>
cd sube-G17
```

2. **Crear un entorno virtual** (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Ejecutar la aplicación**:
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## ☁️ Despliegue en Streamlit Cloud

### Pasos para Desplegar

1. **Asegurarse de que el repositorio esté en GitHub**:
   - Todos los archivos necesarios deben estar en el repositorio
   - El archivo `requirements.txt` debe estar presente y actualizado
   - Los artefactos del modelo deben estar en la carpeta `artifacts/`

2. **Conectar con Streamlit Cloud**:
   - Ir a [share.streamlit.io](https://share.streamlit.io)
   - Iniciar sesión con tu cuenta de GitHub
   - Seleccionar "New app"
   - Elegir el repositorio `sube-G17`
   - Especificar:
     - **Main file path**: `app.py`
     - **Python version**: 3.8 o superior
   - Click en "Deploy"

3. **Verificar el despliegue**:
   - Streamlit Cloud instalará automáticamente las dependencias
   - La app estará disponible en una URL pública

### Notas Importantes para Streamlit Cloud

- ✅ Todos los archivos necesarios están incluidos en el repositorio
- ✅ El archivo `requirements.txt` está actualizado con todas las dependencias
- ✅ El archivo `.gitignore` está configurado correctamente
- ✅ Los artefactos del modelo están en `artifacts/` y deben estar en el repositorio

## 📊 Datos y Modelo

- **Dataset**: `final_2024-11-04.csv` - Datos históricos de transporte público
- **Modelo**: Linear Regression entrenado con scikit-learn
- **Features**: Temporales, climáticas, y características del servicio

## 🔧 Tecnologías Utilizadas

- **Streamlit**: Framework para la aplicación web
- **Altair**: Visualizaciones interactivas
- **Pandas**: Manipulación de datos
- **Scikit-learn**: Modelo de machine learning
- **Joblib**: Serialización del modelo
- **NumPy**: Operaciones numéricas

## 📝 Notas

- El modelo NO utiliza features de lag (datos históricos previos), por lo que las predicciones se basan únicamente en patrones temporales y contextuales
- Las visualizaciones son interactivas y permiten explorar los datos mediante zoom, pan y tooltips
- La aplicación está optimizada para funcionar tanto localmente como en Streamlit Cloud

## 📚 Referencias

- Enunciado de la Cuarta Entrega - Ciencia de Datos 5k9 - 2025
- Notebook de entrenamiento: `IGNA_Entrega3_DiazVeggiani.ipynb`

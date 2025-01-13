# Proyecto de Clasificación con Keras y Python

Este proyecto implementa una red neuronal convolucional (CNN) utilizando Keras para clasificar imágenes en un conjunto de datos. 

## Contenido del Proyecto

- **Preprocesamiento de imágenes**: Se utiliza `ImageDataGenerator` para la normalización y la aumentación de datos.
- **Entrenamiento del modelo**: Una CNN diseñada con capas de convolución, max-pooling y dropout.
- **Visualización de resultados**: Gráficas de precisión y pérdida.
- **Guardado y evaluación del modelo**: Modelo guardado en formato `.keras` y `.h5`.

---

## Requisitos

- Python 3.7 o superior
- TensorFlow y Keras
- Matplotlib
- Un conjunto de datos estructurado en carpetas de entrenamiento y prueba.

---

## Estructura de Carpetas

El proyecto asume la siguiente estructura para el conjunto de datos:

datasetFO/ ├── train/ │ ├── clase1/ │ ├── clase2/ ├── test/ ├── clase1/ ├── clase2/

## Endpoints Clave del Código

### Construcción del Modelo
El modelo CNN incluye:

- Tres capas de convolución y max-pooling.
- Capa de aplanamiento (Flatten).
- Regularización mediante dropout.
- Salida con activación softmax para clasificación multiclase.

### Aumentación de Datos
Se aplica una variedad de transformaciones, incluyendo:

- Rotaciones
- Desplazamientos
- Volteos horizontales

### Guardado del Modelo
- **Modelo con mejor desempeño**: `models/best_cnn_model.keras`
- **Modelo final**: `models/final_cnn_model.h5`

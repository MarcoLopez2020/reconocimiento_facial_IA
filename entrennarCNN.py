import os
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Definir directorios para los datos de entrenamiento y prueba
base_dir = r'C:\Users\USUARIO\Desktop\datasetFO'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Crear carpeta para guardar modelos si no existe
models_dir = './models'
os.makedirs(models_dir, exist_ok=True)

# Generadores de datos con aumentación para entrenamiento y normalización para prueba
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(56, 56),
    batch_size=32,
    class_mode='categorical',
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(56, 56),
    batch_size=32,
    class_mode='categorical',
)

# Construcción del modelo CNN
model = Sequential([
    Conv2D(200, (3, 3), input_shape=(56, 56, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(100, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(50, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax'),
])

# Compilar el modelo
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

# Configuración de callbacks
checkpoint = ModelCheckpoint(
    filepath=os.path.join(models_dir, 'best_cnn_model.keras'),  # Cambiado a .keras
    monitor='val_loss',
    save_best_only=True,
    mode='min',
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Número de épocas sin mejora antes de detener
    restore_best_weights=True,
)

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=test_generator,
    callbacks=[checkpoint, early_stopping],
)

# Evaluación del modelo guardado en el conjunto de prueba
best_model = load_model(os.path.join(models_dir, 'best_cnn_model.keras'))
test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f"\nPrecisión en el conjunto de prueba: {test_accuracy * 100:.2f}%")

# Guardar el modelo final
final_model_path = os.path.join(models_dir, 'final_cnn_model.h5')
model.save(final_model_path)
print(f"Modelo final guardado como '{final_model_path}'")

# Visualización de resultados de entrenamiento
def plot_training_results(history):
    plt.figure(figsize=(14, 6))

    # Gráfica de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validación', marker='o')
    plt.title('Precisión del Modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid()

    # Gráfica de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento', marker='o')
    plt.plot(history.history['val_loss'], label='Validación', marker='o')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Mostrar resultados de entrenamiento
plot_training_results(history)

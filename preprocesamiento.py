import cv2
import os
import numpy as np

def preprocess_images(input_dir, output_dir, img_size=(56, 56), augment=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            # Convertir a escala de grises y detectar la cara
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Recortar y redimensionar la cara detectada
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, img_size)

                # Generar diferentes versiones de la imagen
                preprocessed_images = []

                # Imagen binaria (Otsu)
                _, binary = cv2.threshold(face_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                preprocessed_images.append(binary)

                # Filtro pasa bajo (suavizado)
                low_pass = cv2.GaussianBlur(face_resized, (5, 5), 0)
                preprocessed_images.append(low_pass)

                # Filtro pasa alto (acento de bordes)
                kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
                high_pass = cv2.filter2D(face_resized, -1, kernel)
                preprocessed_images.append(high_pass)

                # Guardar versiones preprocesadas
                for i, proc_img in enumerate(preprocessed_images):
                    output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_preproc_{i}.png")
                    cv2.imwrite(output_path, proc_img)

                # Aumento de datos (si está habilitado)
                if augment:
                    augmentations = [
                        cv2.rotate(face_resized, cv2.ROTATE_90_CLOCKWISE),
                        cv2.rotate(face_resized, cv2.ROTATE_90_COUNTERCLOCKWISE),
                        cv2.rotate(face_resized, cv2.ROTATE_180),
                        cv2.flip(face_resized, 1),  # Espejo horizontal
                        cv2.resize(face_resized, (int(img_size[0] * 1.2), int(img_size[1] * 1.2))),  # Escalado
                    ]
                    for j, aug_img in enumerate(augmentations):
                        aug_resized = cv2.resize(aug_img, img_size) if aug_img.shape[:2] != img_size else aug_img
                        aug_output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_aug_{j}.png")
                        cv2.imwrite(aug_output_path, aug_resized)

                break  # Procesa solo la primera cara encontrada

# Llama a la función con tu camino especificado
preprocess_images(
    input_dir=r'C:\Users\USUARIO\Desktop\datasetFO\Otros', 
    output_dir=r'C:\Users\USUARIO\Desktop\datasetFO\Otros\processed_images'
)

from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
import shutil
import cv2

# === Load model and history ===
model = load_model('weather_classifier_model_final.h5')

# Plot training history 
with open('training_history.json', 'r') as f:
    history = json.load(f)

train_acc = history['accuracy']
val_acc = history['val_accuracy']
epochs = len(train_acc)

# Draw accuracy plot
def draw_accuracy_plot(train_acc, val_acc, save_path='accuracy_plot.png'):
    width, height, margin = 800, 600, 60
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.line(img, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 2)
    cv2.line(img, (margin, margin), (margin, height - margin), (0, 0, 0), 2)

    def to_px_x(epoch): return int(margin + epoch * (width - 2 * margin) / (epochs - 1))
    def to_px_y(val): return int(height - margin - val * (height - 2 * margin))

    for i in range(1, epochs):
        cv2.line(img, (to_px_x(i - 1), to_px_y(train_acc[i - 1])), (to_px_x(i), to_px_y(train_acc[i])), (255, 0, 0), 2)
        cv2.line(img, (to_px_x(i - 1), to_px_y(val_acc[i - 1])), (to_px_x(i), to_px_y(val_acc[i])), (0, 0, 255), 2)

    cv2.putText(img, 'Train Accuracy', (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, 'Val Accuracy', (width - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, 'Epochs', (width // 2 - 40, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, 'Accuracy', (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imwrite(save_path, img)
    print(f"Saved plot as {save_path}")

draw_accuracy_plot(train_acc, val_acc)

# === Prediction and filtering logic ===
def classify_and_filter_images(folder_path, output_file, output_folder, threshold=0.6):
    os.makedirs(output_folder, exist_ok=True)
    count = 0

    with open(output_file, 'w') as f:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)

                # Load and preprocess image
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Predict
                prediction = model.predict(img_array)[0][0]
                predicted_label = 'rainy 🌧️' if prediction > threshold else 'clear ☀️'
                score = f"{prediction:.4f}"
                result_line = f"{filename}: {predicted_label} ({score})"
                print(result_line)
                
                # score = f"{prediction:.4f}"
                # result_line = f"{filename}: {predicted_label} ({score})"
                # print(result_line)
                f.write(result_line + "\n")

                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{predicted_label}_{score}{ext}"
                dst_path = os.path.join(output_folder, new_filename)
                shutil.copy(img_path, dst_path)
                count += 1

    print(f"\nFiltered {count} images to '{output_folder}' and logged in '{output_file}'")


classify_and_filter_images(
    
    folder_path='dataset2/clear_and_rainy_drops_images',
    output_file='clear_and_rainy_drops_images.txt',
    output_folder='clear_and_rainy_drops_images_check_folder',

    threshold=0.5  
)


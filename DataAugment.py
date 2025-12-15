import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img, img_to_array, save_img

INPUT_DIR = r"C:\Users\Sudarshan\Desktop\Mini Project DL\archive (2)\dataset-resized"    # your original dataset root (class subfolders)
OUTPUT_DIR = r"C:\Users\Sudarshan\Desktop\Mini Project DL\archive (2)\Data-augmented" # where augmented images will be saved
IMG_SIZE = (224, 224)
NUM_AUG_PER_IMAGE = 5
AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
    layers.Lambda(lambda x: tf.image.random_brightness(x, 0.2)),
])

os.makedirs(OUTPUT_DIR, exist_ok=True)
classes = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
for cls in classes:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

for cls in classes:
    in_cls_dir = os.path.join(INPUT_DIR, cls)
    out_cls_dir = os.path.join(OUTPUT_DIR, cls)
    filenames = [f for f in os.listdir(in_cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    print(f"Processing class '{cls}' -> {len(filenames)} files")

    for fname in filenames:
        in_path = os.path.join(in_cls_dir, fname)
        # load image as float32 [0,255]
        pil = load_img(in_path, target_size=IMG_SIZE)
        arr = img_to_array(pil)



        for i in range(NUM_AUG_PER_IMAGE):
            img_tensor = tf.expand_dims(arr, 0)
            aug = data_augmentation(img_tensor, training=True)
            aug = tf.cast(aug[0], tf.uint8).numpy()
            base, ext = os.path.splitext(fname)
            out_name = f"{base}_aug{i}{ext if ext else '.jpg'}"
            out_path = os.path.join(out_cls_dir, out_name)
            save_img(out_path, aug)
    print(f"Saved augmented images for class '{cls}' to {out_cls_dir}")
print("Done.")

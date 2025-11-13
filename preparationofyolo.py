import os
import shutil

# ----- Adjust these paths if needed -----
base_path = r"E:\pathpath\training\data\food101"
images_path = os.path.join(base_path, "images")
meta_path = os.path.join(base_path, "meta")

# Output dataset for YOLOv8
output_base = r"E:\pathpath\training\data\food101"
train_dir = os.path.join(output_base, "train")
val_dir = os.path.join(output_base, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Function to copy images safely
def copy_images(list_file, destination_dir):
    with open(list_file, "r") as f:
        lines = f.read().splitlines()

    for i, line in enumerate(lines):
        class_name, img_name = line.split("/")
        src = os.path.join(images_path, class_name, img_name + ".jpg")
        dest_folder = os.path.join(destination_dir, class_name)
        os.makedirs(dest_folder, exist_ok=True)
        shutil.copy(src, dest_folder)

        if (i + 1) % 500 == 0:
            print(f"Copied {i+1}/{len(lines)} images to {destination_dir}")

# --- Copy training and validation images ---
print("📦 Copying training images...")
copy_images(os.path.join(meta_path, "train.txt"), train_dir)
print("✅ Training set ready!")

print("📦 Copying validation images...")
copy_images(os.path.join(meta_path, "test.txt"), val_dir)
print("✅ Validation set ready!")

print("🎉 Food-101 is now ready for YOLOv8 classification!")


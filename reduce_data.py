import os
import shutil

def limit_images_per_class(dataset_path, max_images=200):
    for class_dir in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_dir)

        if os.path.isdir(class_path):
            images = sorted(os.listdir(class_path))  # sort to keep consistency
            excess_images = images[max_images:]

            for img in excess_images:
                os.remove(os.path.join(class_path, img))
            print(f"{class_dir}: reduced to {max_images} images.")

# Example usage
limit_images_per_class('data/raw/PlantVillage', max_images=200)

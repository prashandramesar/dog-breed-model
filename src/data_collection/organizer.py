# data_preparation.py
import os
import pathlib
import shutil

from sklearn.model_selection import train_test_split


def organize_dataset(
    source_dir: str | pathlib.Path,
    target_dir: str | pathlib.Path,
    test_split: float = 0.15,
    val_split: float = 0.15,
) -> None:
    """
    Organizes images into train/val/test directories.

    Args:
        source_dir: Directory containing breed folders with images
        target_dir: Target directory for the organized dataset
        test_split: Fraction of images to use for testing
        val_split: Fraction of images to use for validation
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Create train, validation, and test directories
    train_dir = os.path.join(target_dir, "train")
    val_dir = os.path.join(target_dir, "validation")
    test_dir = os.path.join(target_dir, "test")

    for directory in [train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Process each breed directory
    for breed_folder in os.listdir(source_dir):
        breed_path = os.path.join(source_dir, breed_folder)

        if os.path.isdir(breed_path):
            # Create breed folder in each split directory
            train_breed_dir = os.path.join(train_dir, breed_folder)
            val_breed_dir = os.path.join(val_dir, breed_folder)
            test_breed_dir = os.path.join(test_dir, breed_folder)

            for directory in [train_breed_dir, val_breed_dir, test_breed_dir]:
                if not os.path.exists(directory):
                    os.makedirs(directory)

            # Get all images for this breed
            images: list[str] = [
                img
                for img in os.listdir(breed_path)
                if img.endswith((".jpg", ".jpeg", ".png"))
            ]

            # Split into train, validation, and test sets
            train_images, test_images = train_test_split(
                images, test_size=test_split, random_state=42
            )

            train_images, val_images = train_test_split(
                train_images, test_size=val_split / (1 - test_split), random_state=42
            )

            # Copy images to their respective directories
            for img_list, target_breed_dir in [
                (train_images, train_breed_dir),
                (val_images, val_breed_dir),
                (test_images, test_breed_dir),
            ]:
                for img in img_list:
                    src_path = os.path.join(breed_path, img)
                    dst_path = os.path.join(target_breed_dir, img)
                    shutil.copy(src_path, dst_path)

            print(f"Processed {breed_folder}: {len(train_images)} train")
            print(f"{len(val_images)}: validation, {len(test_images)} test")


if __name__ == "__main__":
    # Adjust these paths to your environment
    source_directory = pathlib.Path().resolve() / "data" / "raw"
    target_directory = pathlib.Path().resolve() / "data" / "processed"

    organize_dataset(source_directory, target_directory)

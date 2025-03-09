import pathlib
import time

import requests


def test_prediction(image_path: str | pathlib.Path) -> None:
    """
    Test the prediction API endpoint with an image file.

    Args:
        image_path: Path to the image file to test
    """
    url = "http://127.0.0.1:8000/predict/"

    # Convert the path to string for the filename in the files parameter
    image_path_str = str(image_path)

    with open(image_path, "rb") as image_file:
        # Use string version of the path for the filename
        files = {"file": (image_path_str, image_file, "image/jpeg")}

        # Measure response time
        start_time = time.time()
        response = requests.post(url, files=files)
        end_time = time.time()

        # Print results
        print(f"\nTesting: {image_path_str}")
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")

        if response.status_code == 200:
            result = response.json()
            print("Predictions:")
            for prediction in result["predictions"]:
                breed = prediction["breed"]
                confidence = prediction["confidence"] * 100
                print(f"  {breed}: {confidence:.2f}%")
        else:
            print(f"Error: {response.text}")


if __name__ == "__main__":
    # Test with different dog images
    p = (pathlib.Path().resolve() / "src" / "backend" / "test_images").glob("**/*")
    file_paths: list[pathlib.Path] = [x for x in p if x.is_file()]
    for test_image in file_paths:
        test_prediction(test_image)

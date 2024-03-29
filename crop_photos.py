import os
import cv2

# Define paths to the base images and their corresponding binary masks
base_dir = "PATH_TO_IMAGES"
mask_dir = "PATH_TO_MASKS_OF_IMAGES"

# Define output directory for the cropped images
output_dir = "OUTPUT_PATH_FOLDER"

# Loop over the files in the base image directory
for file in os.listdir(base_dir):
    if file.endswith(".jpg"):
        # Load the base image
        base_img_path = os.path.join(base_dir, file)
        base_img = cv2.imread(base_img_path)

        # Load the corresponding binary mask
        mask_path = os.path.join(mask_dir, file[:-4] + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Find the bounding box of the mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i,cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)

            # Crop the base image based on the bounding box
            cropped_img = base_img[y:y + h, x:x + w]

            # Save the cropped image
            output_path = os.path.join(output_dir, f"{i}_{file}")
            print(output_path)
            cv2.imwrite(output_path, cropped_img)

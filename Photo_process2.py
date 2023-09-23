import cv2
import numpy as np
import os
import sys
import shutil

def extract_tooth_from_image(img_path):
    """Extract the tooth part from the image."""
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def process_and_save_images(folder_name, input_base, output_base):
    """Process the images from the specified folder and save to the output directory."""
    current_folder_path = os.path.join(input_base, folder_name)
    output_subfolder = os.path.join(output_base, folder_name)
    os.makedirs(output_subfolder, exist_ok=True)
    
    for img_name in os.listdir(current_folder_path):
        img_path = os.path.join(current_folder_path, img_name)
        processed_img = extract_tooth_from_image(img_path)
        output_img_path = os.path.join(output_subfolder, img_name)
        cv2.imwrite(output_img_path, processed_img)

def main():
    folders = ['13', '14', '15', '16']
    input_base = 'data/0920'
    output_base = 'data/process2/0920'
    
    for folder in folders:
        process_and_save_images(folder, input_base, output_base)
    
    print("All images processed successfully!")

if __name__ == '__main__':
    main()

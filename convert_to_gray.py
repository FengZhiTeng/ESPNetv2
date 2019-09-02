import cv2
import os

image_dir = "./vision_datasets/sample_dataset/annotations/"
image_files = os.listdir(image_dir)

for im_file in image_files:
    old_filename = image_dir + im_file
    print(old_filename)
    image = cv2.imread(old_filename)
    try:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        print("Source is empty")
    new_filename = old_filename.replace(".png", "_gray.png")
    cv2.imwrite(new_filename, gray_img)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2


def open_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found.")
        exit()
    else:
        print_info(image)

    return image


def open_usb_cam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not found")
        exit()

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("フレームを取得できませんでした")
        exit()

    return frame


def print_info(image):
    if len(image.shape) == 3: # Color image
        height, width, channels = image.shape
        print(f"Height: {height}, Width: {width}, Channels: {channels}")
    elif len(image.shape) == 2: # Gray scale image
        height, width = image.shape
        print(f"Height: {height}, Width: {width}, Gray Scale")


def preprocess_image(image):
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ノイズ除去（ガウシアンフィルタ）
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 二値化（大津の二値化を使用）
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary



def save_image(image, output_path):
    ret = cv2.imwrite(output_path, image)
    if ret:
        print(f"Image saved successfully: {output_path}")
    else:
        print("Failed to save the image.")


def main():
    image_path = "./data/lenna.png"
    image = open_image(image_path)
    # image = open_usb_cam()

    if image is not None:
        cv2.imshow("Original Image", image)

        processed_image = preprocess_image(image)
        cv2.imshow("Processed Image", processed_image)

        # Press any key to close
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        save_image(processed_image, "./output/processed_lenna.png")


if __name__ == "__main__":
    main()

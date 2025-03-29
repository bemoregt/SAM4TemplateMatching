import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import torch

MASK_COLOR = (255, 0, 0)  # Red mask color

# Load the image
image_path = "/Users/user1/Desktop/music.png"
image = cv2.imread(image_path)
original_image = image.copy()  # Keep a copy of the original image

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "/Users/user1/Downloads/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)

template = None  # Initialize the template

def make_mask_2_img(mask):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(MASK_COLOR).reshape(1, 1, -1)
    mask_image = mask_image.astype(np.uint8)
    return mask_image

def on_mouse_move(event):
    global image, template
    input_point = np.array([[event.x, event.y]])
    input_label = np.array([1])
    
    mask, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    
    mask_img = make_mask_2_img(mask)
    gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    _, binaryimg = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binaryimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        template = image[y:y+h, x:x+w]  # Create the template from the original image within the bounding rectangle
    
    image_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(image_rgb)
    img = ImageTk.PhotoImage(im)
    img_label_proc.img = img
    img_label_proc.config(image=img)

def on_key_press(event):
    global image, template, original_image
    
    if event.char == 'm' and template is not None:
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.8)
        
        for pt in zip(*loc[::-1]):
            cv2.rectangle(image, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0,255,0), 2)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(image_rgb)
        img = ImageTk.PhotoImage(im)
        img_label_proc.img = img
        img_label_proc.config(image=img)
    
    if event.char == 'r':  # Reload the original image
        image = original_image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(image_rgb)
        img = ImageTk.PhotoImage(im)
        img_label_orig.img = img
        img_label_orig.config(image=img)
        img_label_proc.img = img
        img_label_proc.config(image=img)

root = tk.Tk()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
im = Image.fromarray(image_rgb)
img = ImageTk.PhotoImage(im)
img_label_orig = tk.Label(root, image=img)
img_label_orig.grid(row=0, column=0)
img_label_proc = tk.Label(root)
img_label_proc.grid(row=0, column=1)

root.bind("<Button-1>", on_mouse_move)
root.bind("<Key>", on_key_press)
root.mainloop()
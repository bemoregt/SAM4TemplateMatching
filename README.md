# SAM4TemplateMatching

An interactive application that combines Meta AI's Segment Anything Model (SAM) with OpenCV's template matching for object detection and segmentation.

## Overview

This application allows users to:

1. Click on objects in an image to segment them using SAM
2. Automatically extract the segmented object as a template
3. Find all similar objects in the image through template matching

The tool demonstrates how powerful foundation models like SAM can be combined with traditional computer vision techniques to create interactive and practical applications.

## Features

- Interactive object segmentation with a single click
- Automatic template extraction from segmented objects
- Template matching to find similar objects in the image
- Visual display of results in a simple GUI
- Keyboard controls to perform actions and reset the application

## Requirements

- Python 3.6+
- OpenCV
- PyTorch
- PIL (Pillow)
- tkinter
- Meta AI's Segment Anything Model

## Installation

1. Clone this repository:
```bash
git clone https://github.com/bemoregt/SAM4TemplateMatching.git
cd SAM4TemplateMatching
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the SAM model checkpoint from [Meta AI's repository](https://github.com/facebookresearch/segment-anything) and update the `sam_checkpoint` path in the code.

4. Update the `image_path` variable in the code to point to your own image.

## Usage

Run the application:
```bash
python sam_template_matching.py
```

### Controls:
- **Left-click** on an object to segment it with SAM
- Press **'m'** to perform template matching and find similar objects
- Press **'r'** to reset the image

## How It Works

1. When you click on an object, SAM generates a mask for that object
2. The application extracts the bounding box of the mask and creates a template
3. When you press 'm', OpenCV's template matching is applied to find similar objects
4. Matches are highlighted with green rectangles on the image

## Limitations

- The current implementation runs SAM on CPU, which can be slow. For better performance, consider using GPU if available.
- Template matching works best when objects have distinct features and consistent appearance.
- The threshold for template matching is set to 0.8, which may need adjustment for different images.

## License

MIT

## Credits

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- OpenCV for image processing and template matching

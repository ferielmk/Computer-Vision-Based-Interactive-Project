# Welcome to the Computer Vision Project! 🚀
This repository demonstrates the potential of Computer Vision by implementing multiple image processing techniques and interactive real-time applications. The project is designed to combine theoretical concepts with practical tools, offering an engaging way to learn and explore. Here’s what you’ll find:

##  🖼️  Image Processing Filters
Dive into various filters that manipulate images to achieve specific effects:

2D Filter: Applies kernel operations to enhance sharpness, detect edges, or reduce noise.
Laplacian Filter: Highlights intensity variations to emphasize edges and fine details.
Gaussian Filter: Blurs the image to reduce noise and smooth out high-frequency details.
Sobel Filter: Detects edges by calculating horizontal and vertical gradients, useful for segmentation.
Embossing Filter: Creates a 3D effect by highlighting intensity differences, perfect for artistic effects.
Morphological Operations:
Erosion & Dilation: Modifies object boundaries in an image.
Opening & Closing: Removes noise or closes small gaps in contours.

## 🖍️ Object Detection by Color

This feature detects objects in an image based on their color using the HSV color space.
It’s implemented through:

Real-Time Object Detection: Tracks objects dynamically using a live camera feed.
Invisibility Cloak: Renders an object invisible by replacing its pixels with a pre-recorded background. 🧙‍♂️
Green Screen Effect: Replaces the detected object’s background with a custom image or video. 🎥

## 🏎️ Brick Racing Game

An innovative game combining computer vision with real-time interaction:

Gameplay:
Guide a virtual car using a physical object detected by the camera or use the keyboard as an alternative control.
Avoid falling obstacles to keep your score rising!
Dynamic Features:
Obstacles move at increasing speeds, and scores are updated in real-time.
Collisions are detected using bounding boxes, ending the game with a "Game Over" message.

## 🕹️ Technologies Used

Programming Language: Python
Libraries:
OpenCV for image processing and vision algorithms.
NumPy for numerical operations and array manipulations.

In order to play this game, execute the firstgui.py script.

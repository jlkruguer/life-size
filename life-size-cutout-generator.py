# %%
import cv2
import numpy as np
import math
from PIL import Image

def remove_background(image):
    """
    Remove background using GrabCut algorithm
    """
    # Create initial mask
    mask = np.zeros(image.shape[:2], np.uint8)
    
    # Rectangular region containing the object of interest
    rect = (50, 50, image.shape[1]-100, image.shape[0]-100)
    
    # Background and foreground models
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create mask where definite and probable foreground are 1, background is 0
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    
    # Create transparent image
    transparent = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    transparent[:, :, 3] = mask2 * 255
    
    return transparent

def process_image(image_path, real_height_cm=170):
    """
    Process image to create life-size printable template
    Args:
        image_path: Path to input image
        real_height_cm: Actual height of person in centimeters (default 170cm)
    """
    # Load and remove background
    input_image = cv2.imread(image_path)
    output = remove_background(input_image)
    
    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(output, cv2.COLOR_BGRA2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get largest contour (assumed to be the person)
    person_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(person_contour)
    person = output[y:y+h, x:x+w]
    
    # Calculate scaling factor (A4 paper is 21.0 x 29.7 cm)
    PIXELS_PER_CM = 37.795275591  # 96 DPI
    target_height_pixels = real_height_cm * PIXELS_PER_CM
    scale_factor = target_height_pixels / h
    
    # Resize image to life size
    life_size = cv2.resize(person, None, fx=scale_factor, fy=scale_factor)
    
    # Calculate grid dimensions for A4 paper (21.0 x 29.7 cm)
    A4_WIDTH_PX = int(21.0 * PIXELS_PER_CM)
    A4_HEIGHT_PX = int(29.7 * PIXELS_PER_CM)
    
    grid_width = math.ceil(life_size.shape[1] / A4_WIDTH_PX)
    grid_height = math.ceil(life_size.shape[0] / A4_HEIGHT_PX)
    
    # Create individual page images
    pages = []
    for i in range(grid_height):
        row = []
        for j in range(grid_width):
            # Calculate coordinates for this section
            x1 = j * A4_WIDTH_PX
            y1 = i * A4_HEIGHT_PX
            x2 = min((j + 1) * A4_WIDTH_PX, life_size.shape[1])
            y2 = min((i + 1) * A4_HEIGHT_PX, life_size.shape[0])
            
            # Extract section
            section = life_size[y1:y2, x1:x2]
            
            # Create blank A4 page
            page = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX, 4), dtype=np.uint8) * 255
            
            # Place section on page
            page[0:section.shape[0], 0:section.shape[1]] = section
            
            # Add alignment marks
            cv2.line(page, (0, 0), (20, 0), (0,0,0,255), 2)
            cv2.line(page, (0, 0), (0, 20), (0,0,0,255), 2)
            
            # Add page number and grid position
            #text = f"Page {i+1}-{j+1} of {grid_height}x{grid_width}"
            #cv2.putText(page, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0,255), 2)
            
            # Save page
            filename = f'page_{i+1}_{j+1}.png'
            cv2.imwrite(filename, page)
            row.append(filename)
        pages.append(row)
    
    return pages

def main():
    # Example usage
    image_path = './andycohen.jpg'
    height_cm = 60  # Adjust to actual person's height
    pages = process_image(image_path, height_cm)
    
    print(f"Created {len(pages) * len(pages[0])} pages")
    print("Print all pages and align using the corner marks")
    print("Grid layout:")
    for row in pages:
        print(" ".join(row))

if __name__ == "__main__":
    main()



# %%

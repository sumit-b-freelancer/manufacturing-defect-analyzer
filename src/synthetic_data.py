
import os
import random
import numpy as np
from PIL import Image, ImageDraw

def generate_dataset(output_dir='data', num_train=20, num_val=5):
    """
    Generates a synthetic dataset of 'good' (clean) and 'defect' (scratched) images.
    Structure:
    data/train/good
    data/train/defect
    data/val/good
    data/val/defect
    """
    print(f"Generating synthetic dataset in {output_dir}...")
    
    dirs = ['train', 'val']
    classes = ['good', 'defect']
    
    for d in dirs:
        for c in classes:
            os.makedirs(os.path.join(output_dir, d, c), exist_ok=True)
            
    # Helper to generate image
    def create_image(is_defect=False):
        # Create a noisy gray background (simulating metal/surface)
        width, height = 224, 224
        # Base gray
        color = random.randint(100, 200)
        img_array = np.random.normal(color, 10, (height, width, 3)).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        if is_defect:
            draw = ImageDraw.Draw(img)
            # Draw random red scratches
            num_scratches = random.randint(1, 3)
            for _ in range(num_scratches):
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
                x2 = random.randint(0, width)
                y2 = random.randint(0, height)
                width_line = random.randint(2, 5)
                # Red color for defect
                draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=width_line)
        
        return img

    # Generate Train
    for i in range(num_train):
        # Good
        img = create_image(is_defect=False)
        img.save(os.path.join(output_dir, 'train', 'good', f"good_{i}.png"))
        
        # Defect
        img = create_image(is_defect=True)
        img.save(os.path.join(output_dir, 'train', 'defect', f"defect_{i}.png"))

    # Generate Val
    for i in range(num_val):
        # Good
        img = create_image(is_defect=False)
        img.save(os.path.join(output_dir, 'val', 'good', f"good_{i}.png"))
        
        # Defect
        img = create_image(is_defect=True)
        img.save(os.path.join(output_dir, 'val', 'defect', f"defect_{i}.png"))

    print("Dataset generation complete.")
    print(f"Train: {num_train} per class")
    print(f"Val:   {num_val} per class")

if __name__ == "__main__":
    generate_dataset()

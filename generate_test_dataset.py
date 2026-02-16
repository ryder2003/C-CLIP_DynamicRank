"""
Generate a small random test dataset for independent testing.
"""

import os
import csv
import random
from PIL import Image
import numpy as np

# Set random seed for reproducibility
random.seed(2026)
np.random.seed(2026)

# Configuration
NUM_SAMPLES = 20
OUTPUT_DIR = "data/test_independent"
IMAGE_SIZE = 224

# Create directories
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

# Possible subjects, actions, and locations for captions
subjects = ["dog", "cat", "bird", "car", "tree", "house", "person", "flower", "horse", "bicycle"]
actions = ["running", "jumping", "flying", "standing", "sleeping", "eating", "playing", "resting"]
locations = ["in the park", "by the river", "on the beach", "in the garden", "near the mountain", 
             "in the city", "on the road", "in the field"]

# Generate random images and captions
print(f"Generating {NUM_SAMPLES} test samples...")
data = []

for i in range(NUM_SAMPLES):
    # Generate random image (different colors for variety)
    r = random.randint(50, 255)
    g = random.randint(50, 255)
    b = random.randint(50, 255)
    
    # Create base color
    img_array = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    img_array[:, :, 0] = r
    img_array[:, :, 1] = g
    img_array[:, :, 2] = b
    
    # Add some random noise for texture
    noise = np.random.randint(-30, 30, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add random shapes for variety
    num_shapes = random.randint(1, 3)
    for _ in range(num_shapes):
        x = random.randint(0, IMAGE_SIZE - 50)
        y = random.randint(0, IMAGE_SIZE - 50)
        size = random.randint(20, 60)
        shape_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Draw circle or rectangle
        if random.random() > 0.5:
            # Circle
            for dx in range(-size//2, size//2):
                for dy in range(-size//2, size//2):
                    if dx*dx + dy*dy < (size//2)**2:
                        px, py = x + dx, y + dy
                        if 0 <= px < IMAGE_SIZE and 0 <= py < IMAGE_SIZE:
                            img_array[py, px] = shape_color
        else:
            # Rectangle
            img_array[y:min(y+size, IMAGE_SIZE), x:min(x+size, IMAGE_SIZE)] = shape_color
    
    # Save image
    img = Image.fromarray(img_array)
    img_path = f"images/test_img_{i:04d}.jpg"
    full_path = os.path.join(OUTPUT_DIR, img_path)
    img.save(full_path)
    
    # Generate caption
    subject = random.choice(subjects)
    action = random.choice(actions)
    location = random.choice(locations)
    caption = f"A {subject} {action} {location}"
    
    # Add to data
    data.append([img_path, caption])
    
    print(f"  Created: {img_path} - '{caption}'")

# Save CSV
csv_path = os.path.join(OUTPUT_DIR, "test.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'caption'])
    writer.writerows(data)

print(f"\n✅ Test dataset created!")
print(f"   Location: {OUTPUT_DIR}/")
print(f"   Samples: {NUM_SAMPLES}")
print(f"   CSV: {csv_path}")
print(f"   Images: {OUTPUT_DIR}/images/")
print(f"\nThis dataset is completely independent from training and validation data.")

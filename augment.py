import os
from PIL import Image
from torchvision import transforms
import torch

# Path to your dataset folder
data_path = r"D:\Bhavya\Paper on inter class simialrities\Interclasssimi-Outputs\Datasets\Group1\betel"

# Create an output folder for augmented images
output_path = os.path.join(data_path, "augmented")
os.makedirs(output_path, exist_ok=True)

# Define augmentations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),   # Flip image
    transforms.RandomRotation(degrees=20),    # Rotate within [-20, +20]
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), # Color change
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Crop and resize
    transforms.ToTensor()
])

# Loop through images and apply augmentations
for img_name in os.listdir(data_path):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(data_path, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Generate multiple augmentations for each image
        for i in range(5):  # create 5 augmented versions
            augmented = transform(image)
            
            # Convert tensor back to image for saving
            save_img = transforms.ToPILImage()(augmented)
            save_name = f"{os.path.splitext(img_name)[0]}_aug{i+1}.jpg"
            save_img.save(os.path.join(output_path, save_name))

print("Data augmentation completed. Augmented images saved in:", output_path)

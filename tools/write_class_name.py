from torchvision import datasets
import os


data_dir = r'datasets\THFOOD50-v1'
train_ds = datasets.ImageFolder(
    root=os.path.join(data_dir, "train")
)

# Save class names to a text file
with open("class_names.txt", "w", encoding="utf-8") as f:
    for name in train_ds.classes:
        f.write(f"{name}=\n")

print("✅ Saved class names to class_names.txt")

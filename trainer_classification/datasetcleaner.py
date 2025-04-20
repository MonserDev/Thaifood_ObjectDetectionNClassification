import os
from PIL import Image

root_dir = 'data\THFOOD50-v1'

# delete bad images (set to True to auto-delete)
delete_corrupted = True

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])

bad_files = []

for root, _, files in os.walk(root_dir):
    for file in files:
        if is_image_file(file):
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.load()       # force decode image
                    img.convert('RGB')  # simulate what PyTorch will do

            except Exception as e:
                print(f"Corrupted: {file_path} ({e})")
                bad_files.append(file_path)
                if delete_corrupted:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

print(f"\n Scan complete. Found {len(bad_files)} bad image(s).")

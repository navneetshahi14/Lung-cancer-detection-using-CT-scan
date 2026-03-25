import os
import hashlib

def get_image_hashes(data_dir):
    dataset_hashes = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                # Image content ka hash nikalna (Content-based detection)
                with open(file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                # Split identify karna (train/val/test)
                split = root.split(os.sep)[-2] # folder structure ke hisaab se adjust karein
                dataset_hashes[file_path] = (file_hash, file)
    return dataset_hashes

def detect_leakage(data_dir):
    splits = ['train', 'val', 'test']
    split_data = {split: {} for split in splits}

    for split in splits:
        path = os.path.join(data_dir, split)
        if not os.path.exists(path):
            continue
        
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    f_hash = hashlib.md5(f.read()).hexdigest()
                split_data[split][f_hash] = file

    # Compare splits
    print("🔍 Checking for Leakage between splits...")
    
    combinations = [('train', 'val'), ('train', 'test'), ('val', 'test')]
    for s1, s2 in combinations:
        common = set(split_data[s1].keys()) & set(split_data[s2].keys())
        if len(common) > 0:
            print(f"❌ CRITICAL LEAKAGE: Found {len(common)} identical images between {s1} and {s2}!")
        else:
            print(f"✅ Clean: No identical images between {s1} and {s2}.")

# Run it
DATA_DIR = "./lung_ct_split_no_dup"
get_image_hashes(DATA_DIR)
detect_leakage(DATA_DIR)

# import matplotlib.pyplot as plt

# def show_augmented_images(dataset, idx=0, n=5):
#     plt.figure(figsize=(15, 3))
    
#     for i in range(n):
#         img, label = dataset[idx]   # same index multiple times
        
#         plt.subplot(1, n, i+1)
#         plt.imshow(img.permute(1, 2, 0))  # tensor → HWC
#         plt.title(f"Label: {label}")
#         plt.axis('off')
    
#     plt.show()

def get_labels(dataset):
    if hasattr(dataset, "indices"):  # Subset case
        labels = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        labels = dataset.targets
    return labels
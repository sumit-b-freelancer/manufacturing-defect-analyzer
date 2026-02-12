
import os
import shutil
import urllib.request
import tarfile
from tqdm import tqdm

def download_mvtec_subset(category='screw', output_dir='data'):
    """
    Downloads MVTec AD dataset (screw) directly from the official source.
    """
    # Alternative direct link found via search
    url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937691-1629952097/screw.tar.xz"
    tar_path = f"{category}.tar.xz"
    
    print(f"Downloading {url}...")
    
    # Download with progress bar
    # MVTec server might require User-Agent
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)

    with tqdm(unit='B', unit_scale=True, miniters=1, desc=tar_path) as t:
        def reporthook(blocknum, blocksize, totalsize):
            t.total = totalsize
            t.update(blocknum * blocksize - t.n)

        try:
            urllib.request.urlretrieve(url, tar_path, reporthook=reporthook)
        except Exception as e:
            print(f"Download failed: {e}")
            # Clean exit
            return

    print("Extracting...")
    extract_dir = 'mvtec_raw'
    os.makedirs(extract_dir, exist_ok=True)
    
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_dir)
        
    print("Organizing data...")
    
    # Structure needed for ImageFolder:
    # data/train/good, data/train/defect
    # data/val/good, data/val/defect
    
    dirs = ['train', 'val']
    classes = ['good', 'defect']
    
    for d in dirs:
        for c in classes:
            os.makedirs(os.path.join(output_dir, d, c), exist_ok=True)
    
    # Organization Logic
    # MVTec Raw Structure:
    # screw/train/good/
    # screw/test/good/
    # screw/test/defect_type_1/ ...
    
    base_raw = os.path.join(extract_dir, category)
    
    # Source paths
    train_good_dir = os.path.join(base_raw, 'train', 'good')
    test_dir = os.path.join(base_raw, 'test')
    
    # Gather good images
    all_good = []
    if os.path.exists(train_good_dir):
        for f in os.listdir(train_good_dir):
            if f.lower().endswith('.png'):
                all_good.append(os.path.join(train_good_dir, f))
            
    # From test/good
    test_good_dir = os.path.join(test_dir, 'good')
    if os.path.exists(test_good_dir):
        for f in os.listdir(test_good_dir):
            if f.lower().endswith('.png'):
                all_good.append(os.path.join(test_good_dir, f))

    # Gather defect images
    all_defects = []
    # All subfolders in 'test' except 'good' are defects
    if os.path.exists(test_dir):
        for d in os.listdir(test_dir):
            if d == 'good': continue
            sub_dir = os.path.join(test_dir, d)
            if os.path.isdir(sub_dir):
                for f in os.listdir(sub_dir):
                    if f.lower().endswith('.png'):
                        all_defects.append(os.path.join(sub_dir, f))

    # Split and Copy
    def split_and_copy(files, class_name):
        n = len(files)
        # Use smaller subset if too many, to speed up demo training
        # But MVTec screw is small (~300 images), using all is fine.
        n_train = int(n * 0.8)
        train_files = files[:n_train]
        val_files = files[n_train:]
        
        for i, src in enumerate(train_files):
            dst = os.path.join(output_dir, 'train', class_name, f"{class_name}_{i:04d}.png")
            shutil.copy(src, dst)
            
        for i, src in enumerate(val_files):
            dst = os.path.join(output_dir, 'val', class_name, f"{class_name}_{i:04d}.png")
            shutil.copy(src, dst)
            
    split_and_copy(all_good, 'good')
    split_and_copy(all_defects, 'defect')

    print(f"Data ready in {output_dir}/")
    print(f"  Train Good: {int(len(all_good)*0.8)}, Defect: {int(len(all_defects)*0.8)}")
    print(f"  Val   Good: {len(all_good) - int(len(all_good)*0.8)}, Defect: {len(all_defects) - int(len(all_defects)*0.8)}")
    
    # Cleanup
    try:
        if os.path.exists(tar_path):
            os.remove(tar_path)
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
    except:
        pass

if __name__ == "__main__":
    download_mvtec_subset()

if __name__ == "__main__":
    download_mvtec_subset()

if __name__ == "__main__":
    download_mvtec_subset()

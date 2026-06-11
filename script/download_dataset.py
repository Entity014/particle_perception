import os
import urllib.request
import zipfile
import shutil
import time

def download_file_with_progress(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    start_time = time.time()
    
    # Custom headers to act like a browser / standard client
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    
    with urllib.request.urlopen(req) as response:
        total_size = int(response.info().get('Content-Length', 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1MB blocks
        
        last_reported = 0
        with open(dest_path, 'wb') as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                f.write(buffer)
                downloaded += len(buffer)
                
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    # Report progress every 5%
                    if percent - last_reported >= 5 or downloaded == total_size:
                        elapsed = time.time() - start_time
                        speed = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                        print(f"Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB) - Speed: {speed:.2f} MB/s")
                        last_reported = percent
                else:
                    print(f"Downloaded {downloaded / (1024*1024):.1f} MB (unknown total size)")
                    
    print(f"Finished downloading {dest_path} in {time.time() - start_time:.1f}s")

def extract_archive(file_path, extract_to):
    print(f"Extracting {file_path} to {extract_to}...")
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted zip: {file_path}")
        return True
    elif file_path.endswith(".tar.gz") or file_path.endswith(".tgz"):
        import tarfile
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
        print(f"Extracted tar.gz: {file_path}")
        return True
    return False

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(workspace_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    zip_url = "https://github.com/shengzesnail/PIV_dataset/archive/refs/heads/master.zip"
    zip_path = os.path.join(data_dir, "PIV_dataset_master.zip")
    
    # Step 1: Download repository zip if not already downloaded/extracted
    target_dir = os.path.join(data_dir, "PIV_dataset")
    if not os.path.exists(target_dir):
        if not os.path.exists(zip_path):
            download_file_with_progress(zip_url, zip_path)
            
        print(f"Extracting repository zip to {data_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
        extracted_dir = os.path.join(data_dir, "PIV_dataset-master")
        if os.path.exists(extracted_dir):
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.rename(extracted_dir, target_dir)
            print(f"Renamed {extracted_dir} to {target_dir}")
            
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
    # Step 2: Download PTV-Dataset (Laminar Jet video frames)
    ptv_url = "https://github.com/DingShizhe/PTV-Dataset/archive/refs/heads/main.zip"
    ptv_zip_path = os.path.join(data_dir, "PTV_dataset_main.zip")
    ptv_target_dir = os.path.join(data_dir, "PTV_dataset")
    
    if not os.path.exists(ptv_target_dir):
        if not os.path.exists(ptv_zip_path):
            download_file_with_progress(ptv_url, ptv_zip_path)
            
        print(f"Extracting PTV dataset zip to {data_dir}...")
        with zipfile.ZipFile(ptv_zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
        extracted_ptv_dir = os.path.join(data_dir, "PTV-Dataset-main")
        if os.path.exists(extracted_ptv_dir):
            if os.path.exists(ptv_target_dir):
                shutil.rmtree(ptv_target_dir)
            os.rename(extracted_ptv_dir, ptv_target_dir)
            print(f"Renamed {extracted_ptv_dir} to {ptv_target_dir}")
            
        if os.path.exists(ptv_zip_path):
            os.remove(ptv_zip_path)

    # Step 3: Recursively find and resolve Git LFS pointer files across both datasets
    print("Scanning for Git LFS pointers in the datasets...")
    lfs_pointers = []
    
    # We scan both PIV_dataset and PTV_dataset
    for current_target_dir in [target_dir, ptv_target_dir]:
        if not os.path.exists(current_target_dir):
            continue
        for root, dirs, files in os.walk(current_target_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        first_line = f.readline().strip()
                        if first_line == "version https://git-lfs.github.com/spec/v1":
                            lfs_pointers.append((file_path, current_target_dir))
                except Exception:
                    pass
                
    if not lfs_pointers:
        print("No Git LFS pointers found.")
    else:
        print(f"Found {len(lfs_pointers)} Git LFS pointer files to download.")
        for path, parent_dir in lfs_pointers:
            rel_path = os.path.relpath(path, parent_dir)
            url_rel_path = urllib.parse.quote(rel_path)
            
            # Determine correct URL based on which repo the file is in
            if parent_dir == target_dir:
                raw_url = f"https://github.com/shengzesnail/PIV_dataset/raw/master/{url_rel_path}"
            else:
                raw_url = f"https://github.com/DingShizhe/PTV-Dataset/raw/main/{url_rel_path}"
            
            # Download the actual large file over the pointer
            temp_download_path = path + ".tmp"
            try:
                download_file_with_progress(raw_url, temp_download_path)
                os.replace(temp_download_path, path)
                print(f"Successfully replaced LFS pointer with actual file: {rel_path}")
            except Exception as e:
                print(f"Error downloading LFS file {rel_path}: {e}")
                if os.path.exists(temp_download_path):
                    os.remove(temp_download_path)
                continue
                
            # If the downloaded file is a zip/tar, extract it!
            if path.endswith((".zip", ".tar.gz", ".tgz")):
                if path.endswith(".zip"):
                    extract_to = path[:-4]
                elif path.endswith(".tar.gz"):
                    extract_to = path[:-7]
                elif path.endswith(".tgz"):
                    extract_to = path[:-4]
                    
                os.makedirs(extract_to, exist_ok=True)
                try:
                    if extract_archive(path, extract_to):
                        os.remove(path)
                        print(f"Removed archive file after extraction: {rel_path}")
                except Exception as e:
                    print(f"Failed to extract {rel_path}: {e}")

    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()


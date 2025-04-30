import os
import urllib.request
import sys

def download_file(url, destination):
    if os.path.exists(destination):
        print(f"File {destination} already exists, skipping download.")
        return
    
    print(f"Downloading {url} to {destination}...")
    urllib.request.urlretrieve(url, destination)
    print(f"Downloaded {destination} successfully!")

def main():
    # Create directory if it doesn't exist
    os.makedirs("fasttext_models", exist_ok=True)
    
    # Download English model
    en_model_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
    en_model_gz = "fasttext_models/cc.en.300.bin.gz"
    en_model = "fasttext_models/cc.en.300.bin"
    
    # Download Japanese model
    ja_model_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.bin.gz"
    ja_model_gz = "fasttext_models/cc.ja.300.bin.gz"
    ja_model = "fasttext_models/cc.ja.300.bin"
    
    # Download models
    download_file(en_model_url, en_model_gz)
    download_file(ja_model_url, ja_model_gz)
    
    # Extract gz files if needed
    if not os.path.exists(en_model) and os.path.exists(en_model_gz):
        print(f"Extracting {en_model_gz}...")
        os.system(f"gunzip -k {en_model_gz}")
        print(f"Extracted to {en_model}")
    
    if not os.path.exists(ja_model) and os.path.exists(ja_model_gz):
        print(f"Extracting {ja_model_gz}...")
        os.system(f"gunzip -k {ja_model_gz}")
        print(f"Extracted to {ja_model}")
    
    print("All FastText models downloaded and extracted successfully!")

if __name__ == "__main__":
    main()
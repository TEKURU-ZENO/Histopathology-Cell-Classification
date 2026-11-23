import urllib.request
import os
import tarfile

url = 'https://huggingface.co/datasets/MaybeRichard/GenSegDataset/resolve/main/consep.tar'
dest_dir = r'c:\Users\Dev Mehta\Desktop\HISPATOLOGY EXPERIMENT\data\raw'
tar_path = os.path.join(dest_dir, 'consep.tar')

if not os.path.exists(tar_path):
    print('Downloading consep.tar (85.73 MB)...')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(tar_path, 'wb') as out_f:
        chunk_size = 1024 * 1024
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            print(f'  Downloaded {downloaded / (1024*1024):.1f} MB', end='\r')
    print('\nDownload complete!')

print('Extracting...')
extract_dir = os.path.join(dest_dir, 'consep')
os.makedirs(extract_dir, exist_ok=True)
with tarfile.open(tar_path) as tar:
    tar.extractall(extract_dir)

print('Extracted files overview:')
for root, dirs, files in os.walk(extract_dir):
    rel = os.path.relpath(root, extract_dir)
    if files:
        print(f'  {rel}: {len(files)} files (e.g. {files[:3]})')

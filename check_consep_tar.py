import urllib.request

url = 'https://huggingface.co/datasets/MaybeRichard/GenSegDataset/resolve/main/consep.tar'
req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        size = int(resp.headers.get('Content-Length', 0))
        print(f"consep.tar size: {size / (1024*1024):.2f} MB")
except Exception as e:
    print('Error:', e)

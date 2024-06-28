import hashlib

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
            return hash_md5.hexdigest()
       

 # Usage
file_path = "Sentiment_Model/best_model.pth"
print("MD5 checksum:", calculate_md5(file_path))
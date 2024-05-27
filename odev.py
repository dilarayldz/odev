import numpy as np
import cv2
import fiftyone as fo
import fiftyone.zoo as foz
from cryptography.fernet import Fernet
import matplotlib.pyplot as plt

# COCO-2017 validation veri setini yükle
dataset = foz.load_zoo_dataset("coco-2017", split="validation")

session = fo.launch_app(dataset)

# Veri kümesinden bir örnek al
sample = dataset.first()

# Görüntüyü al
image_path = sample.filepath
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Görüntüyü göster
plt.imshow(image)
plt.axis('off')
plt.show()

# Açıklamayı al
caption = sample.ground_truth.detections[0].label

# Gizli anahtar oluştur
key = Fernet.generate_key()
cipher = Fernet(key)

# Açıklamayı şifrele
encrypted_caption = cipher.encrypt(caption.encode())

# Görüntüye gizli anahtarı göm
def embed_key_to_image(image, key):
    # Gizli anahtarı uygun boyutta bir diziye dönüştür
    encoded_key = key.decode()
    key_array = np.frombuffer(encoded_key.encode(), dtype=np.uint8)
    key_length = len(key_array)
    # Anahtarı görüntünün her pikseline eşit şekilde dağıt
    for i in range(3):  # Her bir RGB kanalı için
        image[..., i] ^= key_array[i % key_length]  # Anahtarı her kanala dağıt
    return image

# Görüntüyü kopyalayın
image_with_key = image.copy()

# Anahtarı gömün
image_with_key = embed_key_to_image(image_with_key, key)

# Anahtarı görüntüden çıkar
def extract_key_from_image(image, key_length):
    extracted_key = [image[0, 0, i] for i in range(3)]
    return bytes(extracted_key)

extracted_key = extract_key_from_image(image_with_key, len(key))

# Şifreleme anahtarıyla açıklamayı çöz
decrypted_caption = cipher.decrypt(encrypted_caption).decode()

print("Orjinal Açıklama:", caption)
print("Şifreli Açıklama:", encrypted_caption)
print("Çözülen Açıklama:", decrypted_caption)
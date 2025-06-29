import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
import cv2
import numpy as np

input_folder = "images_src"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

model = ocr_predictor(
    det_arch='db_resnet50',
    reco_arch='crnn_vgg16_bn',
    pretrained=True
)

def smart_preprocess(image):
    """Prétraitement intelligent préservant la structure du texte"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12,12))
    enhanced = clahe.apply(gray)
    
    denoised = cv2.fastNlMeansDenoising(enhanced, h=7, templateWindowSize=7, searchWindowSize=21)
    
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)

def handle_image(image, output_folder):
    """Traitement précis avec contrôle qualité"""
    processed = smart_preprocess(image)
    
    scale_factor = 1.8 
    scaled = cv2.resize(processed, None, fx=scale_factor, fy=scale_factor, 
                       interpolation=cv2.INTER_CUBIC)
    
    debug_path = os.path.join(output_folder, "debug_preprocessed.png")
    cv2.imwrite(debug_path, scaled)
    
    doc = DocumentFile.from_images(debug_path)
    result = model(doc)
    
    raw_text = result.render()
    clean_text = clean_ocr_output(raw_text)
    
    synth_img = result.synthesize(font_family="DejaVuSans.ttf")[0]
    annotated = Image.fromarray(synth_img)
    
    os.remove(debug_path)
    
    return clean_text, annotated

def clean_ocr_output(text):
    """Correction des erreurs courantes de l'OCR"""
    corrections = {
        "Elèvei": "Élève",
        "TINSA": "INSA",
        "BACSSVTS": "BAC S SVT",
        "pdle": "pôle",
        "tournoii": "tournoi"
    }
    
    for error, correction in corrections.items():
        text = text.replace(error, correction)
    
    return text

for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    print(f"\nTraitement de {filename}...")
    input_path = os.path.join(input_folder, filename)
    
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Image {filename} illisible")
        
        h, w = img.shape[:2]
        if h < 500 or w < 500:
            print(f"Attention: résolution faible ({w}x{h})")
        
        text, annotated_img = handle_image(img, output_folder)
        
        base_name = os.path.splitext(filename)[0]
        
        text_path = os.path.join(output_folder, f"{base_name}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        img_path = os.path.join(output_folder, f"{base_name}_annotated.png")
        annotated_img.save(img_path)
        
        print(f"✔ Résultats sauvegardés pour {filename}")
        print(f"   Texte: {text_path}")
        print(f"   Image: {img_path}")
        
    except Exception as e:
        print(f"❌ Échec sur {filename}: {str(e)}")

print("\n✅ Traitement terminé avec contrôle qualité")

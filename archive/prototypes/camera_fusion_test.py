import cv2
import numpy as np
import ai_edge_litert.interpreter as litert  # Resmi Google TFLite Motoru (Python 3.14 Uyumlu)

# =================================================================
# 1. MODEL VE SINIF TANIMLAMALARI
# =================================================================

class_names_model1 = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
class_names_model2 = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

model1_path = 'garbage_resnet50.tflite'
model2_path = 'garbage_efficientnet.tflite'

print("Modeller LiteRT motoruyla yükleniyor, lütfen bekleyin...")
try:
    # Model 1 (ResNet50) Kurulumu
    interpreter1 = litert.Interpreter(model_path=model1_path)
    interpreter1.allocate_tensors()
    input_details1 = interpreter1.get_input_details()
    output_details1 = interpreter1.get_output_details()
    
    # Model 2 (EfficientNetB0) Kurulumu - Artik SHAPE hatasi vermez!
    interpreter2 = litert.Interpreter(model_path=model2_path)
    interpreter2.allocate_tensors()
    input_details2 = interpreter2.get_input_details()
    output_details2 = interpreter2.get_output_details()
    
    print("[BAŞARILI] Her iki model de LiteRT motoruyla yüklendi!")
except Exception as e:
    print(f"HATA: Modeller yüklenirken bir sorun oluştu! {e}")
    exit()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("HATA: Kamera açılamadı!")
    exit()

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
last_printed_class = ""

print("\n--- HİBRİT SİSTEM (LiteRT) AKTİF (METAL / PLASTIK / BIO) ---")
print("Kapatmak için kamera penceresi açıkken 'q' tuşuna basın.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Varsayılan odaklanma alanı
    default_size = int(min(h, w) * 0.7)
    x, y, width, height = (w - default_size) // 2, (h - default_size) // 2, default_size, default_size

    # Kontur Analizi ile Çöp Takibi
    fgmask = fgbg.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours if cv2.contourArea(c) > 8000]
    
    if large_contours:
        best_contour = max(large_contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(best_contour)
        box_size = max(width, height)
        x = max(0, min(x, w - box_size))
        y = max(0, min(y, h - box_size))
        width = height = box_size

    crop_frame = frame[y:y+height, x:x+width]
    
    if crop_frame.size > 0:
        input_image = cv2.resize(crop_frame, (224, 224))
        
        # -----------------------------------------------------------------
        # MODEL 1 (ResNet50) TAHMİN SÜRECİ (Mean Subtraction ve Float32)
        # -----------------------------------------------------------------
        # Görüntüyü RGB'ye çevirip float32 yapıyoruz ve ResNet mean değerlerini çıkarıyoruz
        img_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_rgb[:,:,0] -= 103.939
        img_rgb[:,:,1] -= 116.779
        img_rgb[:,:,2] -= 123.68
        input_data1 = np.expand_dims(img_rgb, axis=0)
        
        interpreter1.set_tensor(input_details1[0]['index'], input_data1)
        interpreter1.invoke()
        output1 = interpreter1.get_tensor(output_details1[0]['index']).flatten()
        
        idx1 = np.argmax(output1)
        class1 = class_names_model1[idx1]
        conf1 = 100 * output1[idx1]
        
        # -----------------------------------------------------------------
        # MODEL 2 (EfficientNetB0) TAHMİN SÜRECİ (Ham Görüntü, Mean Yok)
        # -----------------------------------------------------------------
        img_rgb2 = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        input_data2 = np.expand_dims(img_rgb2, axis=0)
        
        interpreter2.set_tensor(input_details2[0]['index'], input_data2)
        interpreter2.invoke()
        output2 = interpreter2.get_tensor(output_details2[0]['index']).flatten()
        
        idx2 = np.argmax(output2)
        class2 = class_names_model2[idx2]
        conf2 = 100 * output2[idx2]
        
        # =================================================================
        # 2. HİBRİT KARAR VE OYLAMA MEKANIZMASI (FUSION)
        # =================================================================
        final_class = None
        final_conf = 0.0
        
        if class2 == 'biological' and conf2 >= 85.0:
            final_class = "BIO"
            final_conf = conf2
        elif class1 == 'metal' and conf1 >= 70.0:
            final_class = "METAL"
            final_conf = conf1
        elif class1 == 'plastic' and conf1 >= 85.0:
            final_class = "PLASTIC"
            final_conf = conf1
        elif class2 == 'plastic' and conf2 >= 85.0:
            final_class = "PLASTIC"
            final_conf = conf2
        elif class2 == 'metal' and conf2 >= 70.0:
            final_class = "METAL"
            final_conf = conf2

        # =================================================================
        # 3. ÇIKTI VE TERMİNALE LOGLAMA
        # =================================================================
        if final_class is not None:
            text = f"SINIF: {final_class} ({final_conf:.1f}%)"
            box_color = (0, 255, 0)
            
            if final_class != last_printed_class:
                print(f"[FUSION TESPİT] -> SINIF: {final_class} ({final_conf:.1f}%)")
                last_printed_class = final_class
        else:
            text = "ARANIYOR... (METAL/PLASTIK/BIO BEKLENIYOR)"
            box_color = (0, 165, 255)
            last_printed_class = ""
    else:
        text = "NESNE BULUNAMADI"
        box_color = (0, 0, 255)

    # Çizim İşlemleri
    cv2.rectangle(frame, (x, y), (x+width, y+height), box_color, 2)
    cv2.rectangle(frame, (10, 10), (550, 45), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    cv2.imshow('Garbage LiteRT Ensemble Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nSistem kapatıldı.")
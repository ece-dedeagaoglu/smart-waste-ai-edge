# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import tensorflow as tf  # Sizin sisteminizde kurulu olan kÃ¼tÃ¼phaneye cekildi
from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory

# =================================================================
# 1. DONANIM VE SERVO MOTOR AYARLARI (GPIO 18 VE 19)
# =================================================================
try:
    pi_5_factory = LGPIOFactory()
    kapak_egme = Servo(18, pin_factory=pi_5_factory)
    kapak_dondurme = Servo(19, pin_factory=pi_5_factory)
    print("[BASARILI] Servo motorlar Pi 5 surucusuyle baslatildi.")
except Exception as e:
    print(f"HATA: Servo motorlar baslatilamadi! {e}")
    pi_5_factory = None

def servolari_sifirla():
    if pi_5_factory:
        kapak_egme.mid()      # Kapak duz pozisyonda bekler
        kapak_dondurme.min()  # Baslangic noktasi (0 derece - PLASTIC uzeri)
        time.sleep(0.5)

# Sistem acilirken servolari baslangic konumuna getir
servolari_sifirla()

# =================================================================
# 2. MEKANIK DOKME FONKSIYONU (120 DERECE AYRACLARA GORE)
# =================================================================
def copu_dok_ve_sirala(kategori):
    if not pi_5_factory:
        print(f"Servo baglantisi olmadigi icin {kategori} dokulemiyor.")
        return

    print(f"\n>>> MEKANIZMA TETIKLENDI! Hedef Bolme: {kategori}")
    
    # ADIM A: Bolmeye gore ilgili aciya don (120'ser derece aralikla)
    if kategori == "PLASTIC":
        print("Kapak zaten PLASTIC bolmesinde (0 derece).")
        kapak_dondurme.min() # En sol baslangic (0 derece)
    elif kategori == "METAL":
        print("Kapak METAL bolmesine donuyor (120 derece)...")
        kapak_dondurme.mid() # Orta nokta (120 derece)
    elif kategori == "BIO":
        print("Kapak ORGANIC bolmesine donuyor (240 derece)...")
        kapak_dondurme.max() # En sag nokta (240 derece)
    
    time.sleep(1.5) # Donus hareketinin tamamlanmasi icin bekleme suresi
    
    # ADIM B: Kapagi egerek copu hazneye dusur
    print("Kapak egiliyor, cop dokuluyor...")
    kapak_egme.min()    # Egilme acisi
    time.sleep(2.0)     # Copun kayip dusmesi icin bekleme suresi
    
    # ADIM C: Kapagi eski duz haline getir
    print("Kapak toplaniyor...")
    kapak_egme.mid()
    time.sleep(1.0)
    
    # ADIM D: Sistemi baslangic (bekleme) pozisyonuna geri cek
    print("Sistem sifirlaniyor, baslangic noktasina donuluyor...")
    servolari_sifirla()
    print("Sistem hazir, yeni cop bekleniyor...\n")

# =================================================================
# 3. MODEL VE SINIF TANIMLAMALARI
# =================================================================
class_names_model1 = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
class_names_model2 = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

# Sizin klasor yapiniza gore model yollari dad/ icine esitlendi
model1_path = 'models/garbage_resnet50.tflite'
model2_path = 'models/garbage_efficientnet.tflite'

print("Modeller TensorFlow Lite motoruyla yukleniyor, lutfen bekleyin...")
try:
    interpreter1 = tf.lite.Interpreter(model_path=model1_path)
    interpreter1.allocate_tensors()
    input_details1 = interpreter1.get_input_details()
    output_details1 = interpreter1.get_output_details()
    
    interpreter2 = tf.lite.Interpreter(model_path=model2_path)
    interpreter2.allocate_tensors()
    input_details2 = interpreter2.get_input_details()
    output_details2 = interpreter2.get_output_details()
    
    print("[BASARILI] Her iki yapay zeka modeli de yuklendi!")
except Exception as e:
    print(f"HATA: Modeller yuklenirken sorun olustu! {e}")
    exit()

# Raspberry Pi 5 kamera ayarlari
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("HATA: Kamera acilamadi!")
    exit()

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
last_printed_class = ""

# Kararlilik mekanizmasi degiskenleri
ardisik_tespit_sayisi = 0
son_tespit = None
KARARLILIK_ESIGI = 4  # Ayni nesne ust uste 4 kare gorulurse kapak tetiklenir

print("\n--- AKILLI COP KUTUSU ENTEGRE SISTEMI AKTIF ---")
print("Kapatmak icin kamera penceresi acikkan 'q' tusuna basin.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    default_size = int(min(h, w) * 0.7)
    x, y, width, height = (w - default_size) // 2, (h - default_size) // 2, default_size, default_size

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
        
        # MODEL 1 (ResNet50)
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
        
        # MODEL 2 (EfficientNetB0)
        img_rgb2 = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        input_data2 = np.expand_dims(img_rgb2, axis=0)
        
        interpreter2.set_tensor(input_details2[0]['index'], input_data2)
        interpreter2.invoke()
        output2 = interpreter2.get_tensor(output_details2[0]['index']).flatten()
        
        idx2 = np.argmax(output2)
        class2 = class_names_model2[idx2]
        conf2 = 100 * output2[idx2]
        
        # FUSION OYLAMA MEKANIZMASI
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

        # KARARLILIK VE MOTOR TETIKLEME FILTRESI
        if final_class is not None:
            text = f"SINIF: {final_class} ({final_conf:.1f}%)"
            box_color = (0, 255, 0)
            
            if final_class == son_tespit:
                ardisik_tespit_sayisi += 1
            else:
                son_tespit = final_class
                ardisik_tespit_sayisi = 1
                
            if final_class != last_printed_class:
                print(f"[FUSION TESPIT] -> SINIF: {final_class} ({final_conf:.1f}%)")
                last_printed_class = final_class
        else:
            text = "ARANIYOR... (METAL/PLASTIK/BIO BEKLENIYOR)"
            box_color = (0, 165, 255)
            last_printed_class = ""
            ardisik_tespit_sayisi = 0
            son_tespit = None
    else:
        text = "NESNE BULUNAMADI"
        box_color = (0, 0, 255)
        ardisik_tespit_sayisi = 0
        son_tespit = None

    # Eger tespitten kesin emin olunduysa mekanizmayi calistir
    if ardisik_tespit_sayisi >= KARARLILIK_ESIGI and son_tespit is not None:
        cv2.putText(frame, "KAPAK ACILIYOR...", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('Garbage LiteRT Ensemble Test', frame)
        cv2.waitKey(1)
        
        # Mekanik siralamayi tetikle (Plastik, Metal veya Bio)
        copu_dok_ve_sirala(son_tespit)
        
        # Sayaclari sifirla ve kameradaki eski kare yigini bosalt
        ardisik_tespit_sayisi = 0
        son_tespit = None
        for _ in range(15): cap.read()

    # Ekrana Bilgileri Ciz
    cv2.rectangle(frame, (x, y), (x+width, y+height), box_color, 2)
    cv2.rectangle(frame, (10, 10), (550, 45), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    cv2.imshow('Garbage LiteRT Ensemble Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if pi_5_factory:
    kapak_egme.detach()
    kapak_dondurme.detach()
print("\nSistem kapatildi.")

# -*- coding: utf-8 -*-
import torch
import torchvision
from torchvision.transforms import v2 as T
from PIL import Image, UnidentifiedImageError
import json
import time
import os
import shutil
import sys

# --- 1. PFADE ROBUST MACHEN ---
# Wir ermitteln den Ordner, in dem DIESES Skript liegt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Option A: Alles liegt im selben Ordner wie das Skript (Code-Ordner)
BASE_DIR = SCRIPT_DIR

# Option B: Falls du willst, dass die Ordner im "echten" Projekt-Stammverzeichnis liegen (eins drüber)
# BASE_DIR = os.path.dirname(SCRIPT_DIR) 

print(f"📂 Arbeitsverzeichnis ist: {BASE_DIR}")

# Pfade relativ zum Arbeitsverzeichnis definieren
INPUT_FOLDER = os.path.join(BASE_DIR, "Eingang")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "Ausgang")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "Erledigt")

MODEL_PATH = os.path.join(BASE_DIR, "mein_reh_masken_modell.pth")
JSON_FILENAME = os.path.join(BASE_DIR, "reh_karte.json")
FONT_PATH = os.path.join(BASE_DIR, "arial.ttf")

# Ordner erstellen, falls sie nicht existieren
for folder in [INPUT_FOLDER, OUTPUT_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)


# --- 2. MODELL FUNKTIONEN ---
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

# --- 3. MODELL LADEN ---
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"🚀 Starte Überwachung auf Gerät: {device}")

model = get_model_instance_segmentation(2) # 2 Klassen (Hintergrund + Reh)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ Modell geladen: {MODEL_PATH}")
else:
    print(f"❌ FEHLER: Modelldatei nicht gefunden!")
    print(f"   Erwarteter Pfad: {MODEL_PATH}")
    print("   Bitte kopiere 'mein_reh_masken_modell.pth' in den Ordner.")
    time.sleep(10)
    sys.exit()

model.eval()
model.to(device)

transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

print(f"👀 Überwache Ordner '{INPUT_FOLDER}'... (Drücke Strg+C zum Beenden)")

# --- 4. ENDLOSSCHLEIFE ---
try:
    while True:
        # Liste alle Dateien im Eingangsordner
        try:
            files = os.listdir(INPUT_FOLDER)
        except FileNotFoundError:
            print(f"⚠️ Ordner {INPUT_FOLDER} nicht gefunden. Erstelle neu...")
            os.makedirs(INPUT_FOLDER, exist_ok=True)
            files = []
        
        # Filtere nur Bilder heraus
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if not image_files:
            time.sleep(1)
            continue

        for img_name in image_files:
            full_input_path = os.path.join(INPUT_FOLDER, img_name)
            
            print(f"\n⚡ Neues Bild entdeckt: {img_name}")
            
            # Warten, falls Datei noch kopiert wird
            time.sleep(0.5) 

            try:
                # Bild laden
                pil_image = Image.open(full_input_path).convert("RGB")
                image_tensor = transform(pil_image).to(device)

                # Vorhersage
                with torch.no_grad():
                    prediction = model([image_tensor])

                # Filtern
                pred_scores = prediction[0]['scores']
                pred_boxes = prediction[0]['boxes']
                
                keep = pred_scores > 0.7
                filtered_boxes = pred_boxes[keep]
                filtered_scores = pred_scores[keep]

                found_reh = len(filtered_boxes) > 0

                if found_reh:
                    print(f"   -> 🦌 {len(filtered_boxes)} Reh(e) erkannt!")
                    
                    # --- A. JSON UPDATE ---
                    current_timestamp = time.time()
                    neue_marker = []
                    for box in filtered_boxes:
                        xmin, ymin, xmax, ymax = box.tolist()
                        center_x = (xmin + xmax) / 2
                        center_y = (ymin + ymax) / 2
                        neue_marker.append({
                            "x": int(center_x),
                            "y": int(center_y),
                            "timestamp": current_timestamp,
                            "source_image": img_name
                        })

                    # JSON lesen/schreiben
                    json_data = {"markers": []}
                    if os.path.exists(JSON_FILENAME):
                        try:
                            with open(JSON_FILENAME, "r", encoding='utf-8') as f:
                                content = json.load(f)
                                if "markers" in content: json_data = content
                        except: pass
                    
                    json_data["markers"].extend(neue_marker)
                    with open(JSON_FILENAME, "w", encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2)

                    # --- B. BILD SPEICHERN ---
                    image_to_draw = (image_tensor.cpu() * 255).to(torch.uint8)
                    labels = [f"Reh: {s:.2f}" for s in filtered_scores]
                    
                    # Schriftart laden (Fallback)
                    font = FONT_PATH if os.path.exists(FONT_PATH) else None
                    
                    result_image = torchvision.utils.draw_bounding_boxes(
                        image=image_to_draw,
                        boxes=filtered_boxes,
                        labels=labels,
                        colors="#4287f5",
                        width=8,
                        font=font,
                        font_size=40
                    )
                    
                    output_filename = os.path.join(OUTPUT_FOLDER, "analysed_" + img_name)
                    T.ToPILImage()(result_image).save(output_filename)
                    print(f"   -> Ergebnis gespeichert: {output_filename}")

                else:
                    print("   -> Kein Reh erkannt.")

                # --- C. AUFRÄUMEN ---
                pil_image.close()
                
                destination = os.path.join(PROCESSED_FOLDER, img_name)
                if os.path.exists(destination):
                    os.remove(destination)
                
                shutil.move(full_input_path, destination)
                print(f"   -> Original verschoben nach: {destination}")

            except PermissionError:
                print(f"   ⚠️ Zugriff verweigert. Datei wird evtl. noch benutzt.")
            except UnidentifiedImageError:
                print(f"   ⚠️ Keine Bilddatei/Defekt. Verschiebe nach Erledigt.")
                pil_image.close() if 'pil_image' in locals() else None
                shutil.move(full_input_path, os.path.join(PROCESSED_FOLDER, "broken_" + img_name))
            except Exception as e:
                print(f"   ❌ Fehler: {e}")

        time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 Überwachung beendet.")
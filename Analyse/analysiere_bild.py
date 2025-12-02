import torch
import torchvision
from torchvision.transforms import v2 as T
from PIL import Image
import matplotlib.pyplot as plt
import json
import time
import os

# --- 1. Modell-Architektur definieren ---
# Diese Funktion muss exakt so aussehen wie im Trainings-Skript (train_reh_mask.py)
def get_model_instance_segmentation(num_classes):
    # Lade das vor-trainierte Mask R-CNN Modell
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # 1. Den Box-Predictor (Bounding Box) anpassen
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    # 2. Den Mask-Predictor (Segmentierungsmaske) anpassen
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

# --- 2. Modell laden und vorbereiten ---
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Benutze Gerät: {device}")

# 2 Klassen: Hintergrund (0) + Reh (1)
num_classes = 2

# Modell initialisieren
model = get_model_instance_segmentation(num_classes)

# Trainierte Gewichte laden
# Stelle sicher, dass "mein_reh_masken_modell.pth" im selben Ordner liegt
# oder gib den vollen Pfad an.
model_path = "mein_reh_masken_modell.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Modell erfolgreich geladen.")
else:
    print(f"❌ FEHLER: Modelldatei '{model_path}' nicht gefunden!")
    exit()

model.eval()
model.to(device)


# --- 3. Bild laden und vorbereiten ---
# HIER BITTE DEN NAMEN DEINES TEST-BILDES EINTRAGEN:
image_path = "./reh_test_bild.jpg" 

if not os.path.exists(image_path):
    print(f"❌ FEHLER: Bilddatei '{image_path}' nicht gefunden! Bitte Pfad anpassen.")
    # Wir erstellen ein Dummy-Bild, damit das Skript nicht abstürzt, falls du es direkt testest
    exit()

pil_image = Image.open(image_path).convert("RGB")

# Transformationen wie im Training
transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
image_tensor = transform(pil_image).to(device)


# --- 4. Vorhersage durchführen ---
print("Analysiere Bild...")
with torch.no_grad():
    prediction = model([image_tensor])

# Ergebnisse extrahieren
pred_boxes = prediction[0]['boxes']
pred_labels = prediction[0]['labels']
pred_scores = prediction[0]['scores']
# Falls du die Masken später auch visualisieren willst:
# pred_masks = prediction[0]['masks']

# --- 5. Ergebnisse filtern ---
# Nur Ergebnisse mit hoher Wahrscheinlichkeit (z.B. > 70%) behalten
threshold = 0.7
keep = pred_scores > threshold

filtered_boxes = pred_boxes[keep]
filtered_scores = pred_scores[keep]
# filtered_labels = pred_labels[keep] 


# --- 6. JSON Export für die Karte ---
json_filename = "reh_karte.json"
current_timestamp = time.time()
neue_marker = []

# Berechnung der Mittelpunkte für die JSON-Datei
for box in filtered_boxes:
    xmin, ymin, xmax, ymax = box.tolist()
    
    # Mittelpunkt berechnen
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    
    marker = {
        "x": int(center_x),
        "y": int(center_y),
        "timestamp": current_timestamp
    }
    neue_marker.append(marker)

# In JSON Datei schreiben
if neue_marker:
    data = {"markers": []}
    
    # Alte Daten laden, falls vorhanden
    if os.path.exists(json_filename):
        try:
            with open(json_filename, "r") as f:
                content = json.load(f)
                if "markers" in content:
                    data = content
        except json.JSONDecodeError:
            pass # Datei kaputt oder leer, wir überschreiben sie
            
    # Neue Marker hinzufügen
    data["markers"].extend(neue_marker)
    
    # Speichern
    with open(json_filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ {len(neue_marker)} Rehe in '{json_filename}' gespeichert.")
else:
    print("Keine Rehe mit ausreichender Sicherheit gefunden.")


# --- 7. Visualisierung (Bild mit Boxen speichern) ---
# Bild für Anzeige vorbereiten (uint8 Format, CPU)
image_to_draw = (image_tensor.cpu() * 255).to(torch.uint8)

labels_with_scores = [
    f"Reh: {score:.2f}" for score in filtered_scores
]

if len(filtered_boxes) > 0:
    result_image = torchvision.utils.draw_bounding_boxes(
        image=image_to_draw,
        boxes=filtered_boxes,
        labels=labels_with_scores,
        colors="lime", # Helles Grün für Rehe
        width=4
    )
else:
    result_image = image_to_draw # Einfach das Originalbild nehmen

result_pil = T.ToPILImage()(result_image)

output_filename = "ergebnis_bild.jpg"
result_pil.save(output_filename)
print(f"Bild mit Boxen gespeichert als '{output_filename}'.")

# Optional: Bild direkt anzeigen (wenn du eine grafische Oberfläche hast)
# plt.figure(figsize=(12, 8))
# plt.imshow(result_pil)
# plt.axis('off')
# plt.show()
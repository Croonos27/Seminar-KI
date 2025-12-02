import os
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import transforms as T
import utils
from engine import train_one_epoch, evaluate
from reh_mask_dataset import RehMaskDataset

def get_model_instance_segmentation(num_classes):
    # 1. Lade ein vor-trainiertes Modell (Mask R-CNN)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # 2. Ersetze den Bounding-Box-Kopf für unsere Anzahl an Klassen
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 3. Ersetze den Masken-Kopf für unsere Anzahl an Klassen
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def get_transform(train):
    transforms = []
    # Wichtig: Bild in Tensor konvertieren
    transforms.append(T.PILToTensor())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    if train:
        # Zufälliges Spiegeln (Data Augmentation) während des Trainings
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    # Gerät auswählen (GPU bevorzugt)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training läuft auf: {device}")

    # Unser Dataset hat 2 Klassen: Hintergrund und Reh
    num_classes = 2

    # Dataset laden
    # HINWEIS: Passe '../RehDaten' an, falls dein Datenordner woanders liegt.
    # Wenn 'RehDaten' im selben Ordner wie der Code liegt, nutze 'RehDaten'.
    data_path = 'RehDaten' 
    if not os.path.exists(data_path):
        # Fallback: Versuche einen Ordner höher zu suchen, falls Code in Unterordner liegt
        data_path = '../RehDaten'
    
    print(f"Suche Daten in: {data_path}")
    
    dataset = RehMaskDataset(data_path, get_transform(train=True))
    dataset_test = RehMaskDataset(data_path, get_transform(train=False))

    # Split: Training und Test
    indices = torch.randperm(len(dataset)).tolist()
    
    # Nimm die letzten 5 Bilder zum Testen, den Rest zum Trainieren
    test_size = 5
    if len(dataset) <= test_size:
        test_size = 1 # Falls du sehr wenige Bilder hast
        
    dataset = torch.utils.data.Subset(dataset, indices[:-test_size])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_size:])

    # DataLoaders erstellen
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, # num_workers=0 ist unter Windows sicherer
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # Modell holen
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # Optimizer (SGD)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Learning Rate Scheduler (verringert die Lernrate alle 3 Epochen)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Trainings-Loop
    num_epochs = 10

    for epoch in range(num_epochs):
        # Trainieren (eine Epoche) - Funktion aus engine.py
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        
        # Lernrate anpassen
        lr_scheduler.step()
        
        # Evaluieren (Testen) - Funktion aus engine.py
        evaluate(model, data_loader_test, device=device)

    # Modell speichern
    save_filename = "mein_reh_masken_modell.pth"
    torch.save(model.state_dict(), save_filename)
    print(f"✅ Training beendet. Modell gespeichert als '{save_filename}'")

if __name__ == "__main__":
    main()
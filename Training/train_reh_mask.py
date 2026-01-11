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
import matplotlib.pyplot as plt # Für die Grafik

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
    transforms.append(T.PILToTensor())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Hilfsfunktion, um den Loss auf dem Test-Set zu berechnen
def validate_loss(model, data_loader, device):
    # Wichtig: Modell muss im Train-Modus sein, um Losses zu berechnen,
    # aber wir deaktivieren den Gradienten, um nicht zu trainieren.
    model.train()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            # Summe aller Losses (Box, Maske, Klasse etc.)
            losses = sum(loss for loss in loss_dict.values())
            
            total_loss += losses.item()
            count += 1
            
    return total_loss / count

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training läuft auf: {device}")

    num_classes = 2
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_path = os.path.join(parent_dir, 'RehDaten')
    
    if not os.path.exists(data_path):
        print(f"❌ FEHLER: Der Ordner '{data_path}' existiert nicht!")
        return
    
    dataset = RehMaskDataset(data_path, get_transform(train=True))
    dataset_test = RehMaskDataset(data_path, get_transform(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    
    test_size = 5
    if len(dataset) <= test_size:
        test_size = 1
        
    dataset = torch.utils.data.Subset(dataset, indices[:-test_size])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_size:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # --- EINSTELLUNGEN FÜR ERWEITERTES TRAINING ---
    num_epochs = 100         # Maximale Anzahl Epochen
    patience = 10            # Wie viele Epochen warten wir ohne Verbesserung?
    best_val_loss = float('inf')
    trigger_times = 0        # Zähler für Early Stopping
    
    train_losses = []        # Zum Speichern für die Grafik
    val_losses = []          # Zum Speichern für die Grafik

    print("Starte Training mit Early Stopping...")

    for epoch in range(num_epochs):
        # 1. Trainieren (eine Epoche)
        # train_one_epoch gibt einen MetricLogger zurück, aus dem wir den Loss lesen können
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        
        # Durchschnittlichen Trainings-Loss dieser Epoche holen
        avg_train_loss = metric_logger.meters['loss'].global_avg
        train_losses.append(avg_train_loss)
        
        # Lernrate anpassen
        lr_scheduler.step()
        
        # 2. Validierungs-Loss berechnen (Testen, wie gut es generalisiert)
        avg_val_loss = validate_loss(model, data_loader_test, device)
        val_losses.append(avg_val_loss)
        
        print(f"Epoche {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 3. Early Stopping Logik
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0 # Counter zurücksetzen
            # Speichere das BESTE Modell (nicht das letzte)
            save_filename = "best_reh_model.pth"
            torch.save(model.state_dict(), save_filename)
            print(f"   --> Neuer Bestwert! Modell gespeichert als '{save_filename}'")
        else:
            trigger_times += 1
            print(f"   --> Keine Verbesserung seit {trigger_times} Epochen.")
            
            if trigger_times >= patience:
                print(f"STOPP: Early Stopping aktiviert nach {epoch+1} Epochen.")
                break

        # Optional: Standard-Evaluierung (mAP) trotzdem ausführen, um Präzision zu sehen
        # evaluate(model, data_loader_test, device=device) 

    # --- GRAFIK ERSTELLEN ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss (Lernen)')
    plt.plot(val_losses, label='Validation Loss (Prüfung)')
    plt.xlabel('Epochen')
    plt.ylabel('Loss (Fehler)')
    plt.title('Trainingsverlauf: Training vs Validierung')
    plt.legend()
    plt.grid(True)
    plot_filename = 'training_loss_plot.png'
    plt.savefig(plot_filename)
    print(f"✅ Grafik gespeichert als '{plot_filename}'")
    print("Fertig.")

if __name__ == "__main__":
    main()
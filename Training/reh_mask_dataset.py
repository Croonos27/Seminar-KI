import os
import torch
import torch.utils.data
import numpy as np
from PIL import Image

class RehMaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        # Wir laden nur die Liste der Bilder. 
        # Der Ordnername "JPEGImages" ist hier wichtig (oder "Images", je nachdem wie er bei dir heißt).
        # HIER BITTE PRÜFEN: Heißt dein Ordner "JPEGImages" oder "Images"?
        # Ich nutze hier "JPEGImages" basierend auf unserer letzten Lösung.
        img_folder = os.path.join(root, "JPEGImages")
        if not os.path.exists(img_folder):
             # Fallback, falls er doch Images heißt
             img_folder = os.path.join(root, "Images")
             
        self.imgs = list(sorted(os.listdir(img_folder)))
        self.img_folder = img_folder # Speichern für später

    def __getitem__(self, idx):
        # 1. Bildpfad laden
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_folder, img_name)
        
        # 2. Maskenpfad konstruieren
        file_name_no_ext = os.path.splitext(img_name)[0]
        mask_name = file_name_no_ext + "_mask.png"
        mask_path = os.path.join(self.root, "Masks", mask_name)

        # Fallback für .jpg Masken
        if not os.path.exists(mask_path):
            mask_name_jpg = file_name_no_ext + "_mask.jpg"
            mask_path_jpg = os.path.join(self.root, "Masks", mask_name_jpg)
            if os.path.exists(mask_path_jpg):
                mask_path = mask_path_jpg
            # Falls immer noch nicht gefunden, könnte es auch nur der Dateiname ohne _mask sein?
            # (Optionaler Fallback, falls du das Schema änderst)

        img = Image.open(img_path).convert("RGB")
        
        # Maske laden (L = Grayscale)
        # WICHTIG: Hier arbeiten wir noch mit NumPy!
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)

        # Instanzen trennen (Hintergrund 0 entfernen)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:] # 0 ist Hintergrund

        # Binäre Masken erstellen (NumPy Boolean Array)
        masks = mask == obj_ids[:, None, None]

        # Fallback: Wenn keine IDs gefunden wurden (nur Weiß/Schwarz ohne ID-Unterscheidung)
        if len(obj_ids) == 0:
             # Alles was nicht schwarz ist, ist das Reh
             masks = mask > 0
             # Dimension hinzufügen (N, H, W) -> (1, H, W)
             masks = masks[None, :, :]

        num_objs = len(masks)
        boxes = []
        
        # --- HIER WAR DAS PROBLEM ---
        # Wir iterieren jetzt über NumPy Arrays, nicht Tensoren.
        for i in range(num_objs):
            pos = np.where(masks[i])
            # Sicherheitscheck: Ist das Objekt überhaupt sichtbar?
            if len(pos[0]) == 0:
                # Falls eine leere Maske dabei ist, fügen wir eine Dummy-Box hinzu oder überspringen
                # Hier: Dummy-Box [0,0,1,1], wird später meist eh gefiltert
                boxes.append([0, 0, 1, 1])
                continue
                
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # Erst JETZT wandeln wir alles in PyTorch Tensoren um
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = int(idx)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
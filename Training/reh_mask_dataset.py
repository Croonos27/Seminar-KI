import os
import torch
import torch.utils.data
import numpy as np
from PIL import Image

class RehMaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        # Pfade sortieren, damit Bild und Maske immer zusammenpassen
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Masks"))))

    def __getitem__(self, idx):
        # 1. Bild und Maske laden
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "Masks", self.masks[idx])
        
        img = Image.open(img_path).convert("RGB")
        # Maske als Graustufen laden (0=Schwarz, 255=Weiß)
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)

        # 2. Instanzen trennen
        # Wir nehmen an, alles was nicht schwarz (0) ist, ist ein Reh.
        # Wir vergeben eindeutige IDs für jedes separate Reh im Bild.
        obj_ids = np.unique(mask)
        # Die 0 ist der Hintergrund, den entfernen wir
        obj_ids = obj_ids[1:]

        # Wenn du mehrere Rehe hast, die sich nicht berühren, ist es besser,
        # sie als separate Objekte zu behandeln.
        # Für den Anfang nehmen wir an, alle weißen Pixel gehören zur Klasse "Reh".
        # Falls du fortgeschritten bist: Nutze hier `skimage.measure.label`, um getrennte Blobs zu finden.
        
        # Hier vereinfacht: Wir machen aus der Maske binäre Masken (0 und 1)
        # Wenn nur ein Reh drauf ist oder alle weiß sind:
        masks = mask > 0
        
        # ACHTUNG: Das Modell erwartet für JEDES Reh eine eigene Maske.
        # Wenn du mehrere Rehe hast, die alle weiß sind, werden sie hier als EIN Objekt betrachtet.
        # Das ist für den Anfang okay.
        
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # Shape muss sein (N, H, W), auch wenn N=1 ist
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)

        # 3. Bounding Boxen aus den Masken berechnen
        boxes = []
        for i in range(len(masks)):
            pos = torch.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Es gibt nur eine Klasse (Reh = 1)
        num_objs = len(boxes)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        
        image_id = torch.tensor([idx])
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
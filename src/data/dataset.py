from __future__ import annotations
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def train_transform(size: int = 224) -> transforms.Compose:
    """Training transforms with strong augmentation.
    
    Args:
        size: Image size.
        
    Returns:
        Transform pipeline.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.8, hue=0.25)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

def eval_transform(size: int = 224) -> transforms.Compose:
    """Evaluation transforms (deterministic).
    
    Args:
        size: Image size.
        
    Returns:
        Transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

class SareeDataset(Dataset):
    """Saree image dataset."""
    
    def __init__(
        self,
        df,
        root: str = ".",
        size: int = 224,
        train: bool = True,
    ):
        """Initialize dataset.
        
        Args:
            df: DataFrame with columns [image_path, design_id, ...].
            root: Root directory for image paths.
            size: Image size.
            train: Use training augmentations.
        """
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = train_transform(size) if train else eval_transform(size)
        self.labels = {
            v: i
            for i, v in enumerate(sorted(self.df["design_id"].astype(str).unique()))
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple:
        """Get item.
        
        Returns:
            (image, label, image_path)
        """
        row = self.df.iloc[index]
        image_path = f"{self.root}/{row['image_path']}"
        
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image = self.transform(img)
        
        label = self.labels[str(row["design_id"])]
        return image, label, row["image_path"]

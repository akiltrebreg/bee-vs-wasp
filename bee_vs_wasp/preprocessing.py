from pathlib import Path

import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms

# ---------- TRANSFORMS ----------
train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


test_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ---------- DATASET CSV ----------
def load_bee_dataset_csv(labels_path: Path) -> pd.DataFrame:
    """Load and preprocess dataset CSV with label encoding.

    Normalizes file paths and encodes string labels to integers.

    Args:
        labels_path: Path to labels.csv file

    Returns:
        DataFrame with normalized paths and encoded labels
    """
    df = pd.read_csv(labels_path)

    df["path"] = df["path"].apply(lambda path: path.replace("\\", "/"))

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])

    return df


# ---------- TRAIN/VAL/TEST SPLIT ----------
def split_dataframes(df: pd.DataFrame):
    """Split dataset into train, validation and test sets.

    Uses 'is_validation' and 'is_final_validation' flags to separate data.
    Returns tuple of (train_df, val_df, test_df) with reset indices.

    Args:
        df: DataFrame with 'is_validation' and 'is_final_validation' columns

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames
    """
    val_df = df[df["is_validation"] == 1].copy()
    test_df = df[df["is_final_validation"] == 1].copy()
    used_idx = val_df.index.union(test_df.index)

    train_df = df.drop(used_idx).copy()

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ---------- TORCH DATASET ----------
class BeeDataset(Dataset):
    """PyTorch Dataset for bee/wasp image classification.

    Loads images from disk and applies transformations.
    Supports both training mode (returns images with labels) and
    inference mode (returns only images).
    """

    def __init__(self, df, imgdir, train=True, transforms=None):
        """Initialize dataset.

        Args:
            df: DataFrame with 'path' and 'label' columns
            imgdir: Root directory containing images
            train: If True, returns (image, label) pairs; else only images
            transforms: Torchvision transforms to apply
        """
        self.df = df
        self.imgdir = imgdir
        self.train = train
        self.transforms = transforms

    def __len__(self):
        """Return total number of samples in dataset."""
        return len(self.df)

    def __getitem__(self, index):
        """Load and return a single sample.

        Reads image from disk, converts BGR to RGB, applies transforms.
        Returns (image, label) if train=True, else only image.

        Args:
            index: Index of sample to retrieve

        Returns:
            Tuple of (image, label) if train=True, else only image tensor
        """
        img_path = self.imgdir / self.df.iloc[index]["path"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image)

        if self.train:
            label = int(self.df.iloc[index]["label"])
            return image, label
        return image

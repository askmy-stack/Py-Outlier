"""Dataset loader implementations."""

from anomaly_detection.data_ingestion.loaders.csv_loader import CSVLoader
from anomaly_detection.data_ingestion.loaders.openml_loader import OpenMLLoader
from anomaly_detection.data_ingestion.loaders.registry_loader import RegistryLoader
from anomaly_detection.data_ingestion.loaders.ucf_loader import UCFLoader

__all__ = ["CSVLoader", "OpenMLLoader", "RegistryLoader", "UCFLoader"]

from pathlib import Path

import streamlit as st

from example.streamlit_utils import (
    labeled_image_dataframe_from_folder,
    columns_display,
    get_fsl_classifier_from_catalog_dataframe,
    write_to_disk,
)
from core.tensorflow_utils import image_dataset_from_paths

catalog = labeled_image_dataframe_from_folder(Path("data/default_catalog"))

st.title("Catalog")
columns_display(list(zip(catalog.image_path, catalog.label)), n_columns=3)


fsl_classifier = get_fsl_classifier_from_catalog_dataframe(catalog)

st.title("Predictions")

files = st.file_uploader(
    "Images to classify", accept_multiple_files=True, type=["jpg", "jpeg"]
)

log_folder = Path("data/log")
log_folder.mkdir(exist_ok=True)
file_paths = write_to_disk(files, log_folder)
if files:
    predictions = fsl_classifier.predictions_to_classes(
        fsl_classifier.predict(image_dataset_from_paths(file_paths))
    )
    columns_display(list(zip(file_paths, predictions.label)), n_columns=4)

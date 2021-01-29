from pathlib import Path

import pandas as pd
import streamlit as st

from example.streamlit_utils import (
    labeled_image_dataframe_from_folder,
    columns_display,
    get_fsl_classifier_from_catalog_dataframe,
    write_to_disk,
)
from tensorflow_utils import image_dataset_from_paths

log_folder = Path("data/log")
log_folder.mkdir(exist_ok=True)

n_classes = st.number_input("N classes", min_value=0, value=2)

sub_catalogues = []
for class_id in range(n_classes):
    class_name = st.text_input(
        f"{class_id} name", value=str(class_id), key=f"{class_id} name"
    )
    files = st.file_uploader(
        f"Catalog for {class_name}", accept_multiple_files=True, type=["jpg", "jpeg"]
    )
    file_paths = write_to_disk(files, log_folder)
    sub_catalog = pd.DataFrame({"label": class_name, "image_path": file_paths})
    sub_catalogues.append(sub_catalog)
    columns_display(list(zip(sub_catalog.image_path, sub_catalog.label)), n_columns=3)

catalog = pd.concat(sub_catalogues)
if catalog.empty:
    st.stop()


fsl_classifier = get_fsl_classifier_from_catalog_dataframe(catalog)

st.title("Predictions")

files = st.file_uploader(
    "Images to classify", accept_multiple_files=True, type=["jpg", "jpeg"]
)


file_paths = write_to_disk(files, log_folder)
if files:
    predictions = fsl_classifier.predictions_to_classes(
        fsl_classifier.predict(image_dataset_from_paths(file_paths))
    )
    columns_display(list(zip(file_paths, predictions.label)), n_columns=4)

from pathlib import Path

import streamlit as st
from example.streamlit_utils import (
    labeled_image_dataframe_from_folder,
    columns_display,
    get_fsl_classifier_from_catalog_dataframe,
)
from tensorflow_utils import image_dataset_from_paths

catalog = labeled_image_dataframe_from_folder(Path("../data/default_catalog"))

st.title("Catalog")
columns_display(list(zip(catalog.image_path, catalog.label)), n_columns=3)


fsl_classifier = get_fsl_classifier_from_catalog_dataframe(catalog)

to_predict = labeled_image_dataframe_from_folder(Path("../data/images"))
predictions = fsl_classifier.predictions_to_classes(
    fsl_classifier.predict(image_dataset_from_paths(list(to_predict.image_path)))
)

st.title("Predictions")
columns_display(list(zip(to_predict.image_path, predictions.label)), n_columns=4)

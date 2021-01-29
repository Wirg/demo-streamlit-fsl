from pathlib import Path
from typing import Tuple, List

import pandas as pd
import streamlit as st

from fsl_classifier import FSLClassifier
from tensorflow_utils import image_dataset_from_paths


@st.cache(allow_output_mutation=True)
def get_fsl_classifier():
    return FSLClassifier.example()


@st.cache(allow_output_mutation=True)
def get_fsl_classifier_from_catalog_dataframe(catalog_dataframe: pd.DataFrame):
    fsl_classifier = get_fsl_classifier()
    fsl_classifier.set_catalog(
        image_dataset_from_paths(list(catalog_dataframe.image_path)),
        catalog_dataframe.label,
    )
    return fsl_classifier


def labeled_image_dataframe_from_folder(folder: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [(str(p), p.parent.name) for p in folder.glob("*/*.jpg")],
        columns=["image_path", "label"],
    )


def columns_display(path_label_list: List[Tuple[str, str]], n_columns: int = 4):
    for i in range(0, len(path_label_list), n_columns):
        columns = st.beta_columns(n_columns)
        for k in range(n_columns):
            if i + k < len(path_label_list):
                image_path, prediction = path_label_list[i + k]
                with columns[k]:
                    st.image(image_path, caption=prediction, use_column_width=True)

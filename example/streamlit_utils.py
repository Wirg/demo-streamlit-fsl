from pathlib import Path
from typing import Tuple, List

import pandas as pd
import streamlit as st

from core.fsl_classifier import FSLClassifier
from core.tensorflow_utils import image_dataset_from_paths


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


@st.cache
def write_file_to_disk(file, log_folder: Path) -> str:
    log_file_path = str(log_folder / file.name)
    with open(log_file_path, "wb") as f:
        f.write(file.read())
    return log_file_path


@st.cache
def write_to_disk(files, log_folder: Path) -> List[str]:
    return [write_file_to_disk(file, log_folder) for file in files]

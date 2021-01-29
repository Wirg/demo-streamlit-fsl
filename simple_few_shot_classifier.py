from pathlib import Path

import pandas as pd

from core.fsl_classifier import FSLClassifier
from core.tensorflow_utils import image_dataset_from_paths

fsl_classifier = FSLClassifier.example()


def labeled_image_dataframe_from_folder(folder: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [(str(p), p.parent.name) for p in folder.glob("*/*.jpg")],
        columns=["image_path", "label"],
    )


catalog = labeled_image_dataframe_from_folder(Path("data/default_catalog"))
fsl_classifier.set_catalog(
    image_dataset_from_paths(list(catalog.image_path)), catalog.label
)


to_predict = labeled_image_dataframe_from_folder(Path("data/images"))
predictions = fsl_classifier.predict(
    image_dataset_from_paths(list(to_predict.image_path))
)

print(fsl_classifier.predictions_to_classes(predictions))

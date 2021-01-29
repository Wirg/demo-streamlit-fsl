# Demo Streamlit Few Shot Learning

Showcasing the power of streamlit to display a demo of a Few Shot Learning model using keras-fsl.
The goal of this project is to help you try streamlit.

## Install

You will need :
- python 3.8
- preferably a venv
- poetry 1+ : `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`
- install dependencies in the venv : `poetry install` in the root folder


## What to look at
### Try yourself
Go to `simple_few_shot_classifier.py` and try to transform it in a [streamlit app](https://docs.streamlit.io/en/stable/api.html#).
Run your app :
`PYTHONPATH=. streamlit run simple_few_shot_classifier.py`

### Some Demos

A few shot learning streamlit app :
- using the images in the `data/default_catalog` folder as catalog and `data/images` as images to predict `example/streamlit_simple_few_shot_classifier.py`
- using the images in the `data/default_catalog` folder as catalog and predicting images from user input `example/streamlit_simple_few_shot_classifier_with_input_from_user.py`
- getting images to predict and to add to the catalog from the user `example/streamlit_simple_few_shot_classifier_with_catalog_from_user.py`

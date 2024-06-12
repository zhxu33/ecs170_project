# ecs170_project
<img src="https://github.com/zhxu33/ecs170_project/assets/77419802/53060682-9e60-41ed-bb32-1a12513e20ac" width="200">

## app/
This contains the flask web app for human action prediction.

> ### Prerequisites
```
pip install Flask tensorflow
```

> ### To run the app
```
cd app
python app.py
```
Open http://localhost:5000 to view it in your browser.

## dataset/
Contains the original dataset/models pulled from [Kaggle](https://www.kaggle.com/datasets/emirhanai/human-action-detection-artificial-intelligence/data). Since the zipped file size is too large, the dataset has to be added in locally.

Download **archive.zip** from [Kaggle](https://www.kaggle.com/datasets/emirhanai/human-action-detection-artificial-intelligence/data). Unzip it and insert **emirhan_human_dataset** to **model/**.

## model/

> ### human-action-detection.ipynb
This notebook is where we preprocessed data and experimented with MobileNet CNN model.

> ### human-action-detection-using-efficientnet.ipynb
This is the final model we built with EfficientNet CNN.

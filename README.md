# ecs170_project
![image](https://github.com/zhxu33/ecs170_project/assets/77419802/e7154bd6-84f9-4568-be4c-82f8f5ea081f)

## app
This contains the flask app for human action prediction.

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

## dataset
Contains the original dataset/models pulled from [Kaggle]([https://pages.github.com/](https://www.kaggle.com/datasets/emirhanai/human-action-detection-artificial-intelligence/data). Since the zipped file size is too large, the dataset has to be added in locally.

Download **archive.zip** from [Kaggle]([https://pages.github.com/](https://www.kaggle.com/datasets/emirhanai/human-action-detection-artificial-intelligence/data). Unzip it and insert **emirhan_human_dataset** to **model/**.

## model/

> ### human-action-detection-using-cnn.ipynb
This notebook is where we preprocessed data and created our models using CNN.

> ### model.py
Contains the final model built using MobileNet CNN.

> ### To set up new model
```
cd model
python model.py
rm ../app/model/class_indices.json ../app/model/final_model.h5
mv class_indices.json ../app/model/ && mv final_model.h5 ../app/model/
```

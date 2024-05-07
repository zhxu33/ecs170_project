# ecs170_project
![image](https://github.com/zhxu33/ecs170_project/assets/77419802/53060682-9e60-41ed-bb32-1a12513e20ac)

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
Contains the original dataset/models pulled from [Kaggle](https://www.kaggle.com/datasets/emirhanai/human-action-detection-artificial-intelligence/data). Since the zipped file size is too large, the dataset has to be added in locally.

Download **archive.zip** from [Kaggle](https://www.kaggle.com/datasets/emirhanai/human-action-detection-artificial-intelligence/data). Unzip it and insert **emirhan_human_dataset** to **model/**.

## model/

> ### human-action-detection.ipynb
This notebook is where we preprocessed data and experimented with models using CNN.

> ### model.py
Contains the final model built using MobileNet CNN.

> ### human-action-detection-using-cnn.ipynb
This is a notebook we found on [Kaggle](https://www.kaggle.com/code/kirollosashraf/human-action-detection-using-cnn) to use as a scaffolding resource.

> ### To set up new model
```
cd model
python model.py
rm ../app/model/class_indices.json ../app/model/final_model.h5
mv class_indices.json ../app/model/ && mv final_model.h5 ../app/model/
```

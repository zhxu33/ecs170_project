# ecs170_project
![image](https://github.com/zhxu33/ecs170_project/assets/77419802/e7154bd6-84f9-4568-be4c-82f8f5ea081f)

## app
This contains the flask web app for human action prediction.

> ### To run the app
```
cd app
python app.py
```

## dataset
Contains the original dataset/models pulled from [Kaggle](https://pages.github.com/). Since the zipped file size is too large, the dataset has to be added in locally.

The dataset can also be downloaded [here] (https://file.io/j964kvgYlzKA). Insert **emirhan_human_dataset.zip** in **model/** and unzip it there.


## model/


> ### human-action-detection-using-cnn.ipynb
This notebook is where we preprocessed data and created our models using CNN.

> ### model.py
Contains the final model built using MobileNet CNN.

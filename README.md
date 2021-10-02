# emotion-recognition


## [Facial Expression Recognition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

train.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order. test.csv contains only the "pixels" column and your task is to predict the emotion column.

The training set consists of 28,709 examples. The public test set used for the leaderboard consists of 3,589 examples. The final test set, which was used to determine the winner of the competition, consists of another 3,589 examples.

This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part of an ongoing research project. They have graciously provided the workshop organizers with a preliminary version of their dataset to use for this contest.


##### Project structure

```
.
├── models
│   ├── best_weights_0.hdf5
│   └── haarcascade_frontalface_default.xml
├── README.md
├── requirments.txt
├── static
│   └── assets
├── templates
│   └── index.html
└── webstreaming.py
```

Clone the repository:
```
git clone git@github.com:walaeddine/RealTimeFacialExpressionRecognition.git
```


Install the required libraries stated in the file requirments.txt
```
pip install -r requirements.txt
```

Go ahead and run the web streamer

```
python webstreaming.py --ip 0.0.0.0 --port 8000
```
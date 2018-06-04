# Digit Recogniser Machine Learning Tensorflow Javascript And React

This project is a simple implemntation of digit recongition using tensorflow js.

If you want to to have sample data, you can use kaggle website.

Demo Link: [Here](https://ehsangazar.github.io/digit-recogniser-machine-learning-tensorflow-javascript/)


## Functions and Steps
In this readme, I try to explain what is happening in `src/App.js`.

### _readTrainingData
First we need to download all the data from train.csv to our project and use papaparse for parsing the CSV to array.
This data includes number of pixels with their information.
```
// Downloading 70 MB file
Papa.parse('/train.csv', {
    download: true,
    complete: async (results) => {
        // using result
    }
})
```

### _initializingTheModel
After downloading the information, we need to define our model which is a sequential model and we want to use SGD, but for using SGD, we need a loss function and layers of model plus an optimizer.
___
Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to discriminative learning of linear classifiers under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. Even though SGD has been around in the machine learning community for a long time, it has received a considerable amount of attention just recently in the context of large-scale learning. 
___

```
this.minstModel = tf.sequential();

this.minstModel.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
}));
this.minstModel.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
}));
this.minstModel.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
}));
this.minstModel.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
}));
this.minstModel.add(tf.layers.flatten());
this.minstModel.add(tf.layers.dense({
    units: 10,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
}));

const LEARNING_RATE = 0.0001;
const optimizer = tf.train.sgd(LEARNING_RATE);
this.minstModel.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
});
```
If your question is how can we develop this model with their layers to be efficient, I should say there is a competition here
[Kaggle Digit Recognition](https://www.kaggle.com/c/digit-recognizer)
The model above, it is what I found on internet plus a little change!

### _trainNewModel
After initializing the model, you can use this function for start the training, because the training data is so big, we have to use part of the file each time to train the model

```
const history = await this.minstModel.fit(
batch.inputs, 
batch.labels, 
{
    batchSize: batchSize
});
```

### _minscPredictionLabeled
And finally for evaluating our model, I didn't use the 100 inputs of training data and I developed to use it for knowing the accuracy or our model.
```
const output = this.minstModel.predict(evalData.inputs);
```
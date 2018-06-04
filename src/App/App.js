import React, { Component } from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';
import Papa from 'papaparse'
import CanvasDraw from "react-canvas-draw";

class App extends Component {
  constructor(props){
    super(props)
    this.minstModel = null
    this.state = {
      batchSize: 10,
      status: 0,
      numberOfTrainingData: 0,
      trainingData: [],
      trainingLabels: [],
      numberOfTestData: 0,
      testData: [],
      testLabels: [],
      step:0,
      predictionNumber: null
    }
  }

  componentDidMount() {
    console.log('Mounted')
  }
  
  _readTrainingData = async () => {
    // Downloading 70 MB file
    this.setState({
      status: this.state.status + 1,
    }, async () => {
      await Papa.parse('https://ehsangazar.github.io/digit-recogniser-machine-learning-tensorflow-javascript/train.csv', {
        download: true,
        complete: async (results) => {
          const newResults = await results.data.slice(1, results.datalength);
          const trainingLabels = newResults.map(item => item[0])
          const trainingData = newResults.map(item => item.slice(1, item.length))
          this.setState({
            status: this.state.status + 1,
            numberOfTrainingData: trainingData.length,
            trainingData: trainingData,
            trainingLabels: trainingLabels
          })
        }
      })
    })
  }

  _readTestKaggleData = async () => {
    // Downloading 48 MB file
    this.setState({
      status: this.state.status + 1,
    }, async () => {
      await Papa.parse('https://ehsangazar.github.io/digit-recogniser-machine-learning-tensorflow-javascript/test.csv', {
        download: true,
        complete: async (results) => {
          const newResults = await results.data.slice(1, results.datalength);
          const testData = newResults.map(item => item.slice(0, item.length))
          this.setState({
            status: this.state.status + 1,
            numberOfTestData: testData.length,
            testData: testData
          })
        }
      })
    })
  }

  _batchData = (batchSize, i, dataInputs, dataLabels = []) => {
    const inputsData = dataInputs.slice(i * batchSize, i * batchSize + batchSize)
    let labelsData = []
    let labels = []
    if (dataLabels.length > 0) {
      labelsData = dataLabels.slice(i * batchSize, i * batchSize + batchSize)
    }
    
    let newFlatInputData = []
    inputsData.forEach(image => {
      image.forEach(pixel => {
        newFlatInputData.push(pixel)
      })
    })
    const imagesShape = [batchSize, 28, 28, 1];

    const inputs = tf.tensor4d(newFlatInputData, imagesShape)

    if (dataLabels.length > 0) {
      labels = tf.oneHot(tf.tensor1d(labelsData, 'int32'), 10).toFloat()
    }
    return { inputs, labels }
  }

  _initializingTheModel = async () => {
    this.setState({
      status: this.state.status + 1
    }, () => {
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

      this.setState({
        status: this.state.status + 1
      })
    })   
  }


  _trainNewModel = async () => {
    let { batchSize, numberOfTrainingData, trainingData, trainingLabels } = this.state;    
    const numberOfSteps = numberOfTrainingData / batchSize - 1
    console.log('Training is about to Start')
    await this.setState({
      status: this.state.status + 1,
      step: 0,
    }, async () => {
      await setTimeout(async ()=>{
        for (let step = 0; step < numberOfSteps; step++) {
          const batch = this._batchData(batchSize, step, trainingData, trainingLabels);
          const history = await this.minstModel.fit(
            batch.inputs, 
            batch.labels, 
            {
              batchSize: batchSize
            });

          const loss = history.history.loss[0].toFixed(6);
          const acc = history.history.acc[0].toFixed(4);
          console.log(`  - step: ${step}: loss: ${loss}, accuracy: ${acc}`);
        }
        // After this
        this.setState({
          status: this.state.status + 1
        })
      },100)
    })
  }

  _minscPredictionLabeled = async (event) => {
    const {
      trainingData,
      trainingLabels,
      batchSize,
      step
    } = this.state
    this.setState({
      status: this.state.status + 1
    }, async () => {
      let correct = 0
      await setTimeout(async () => {
        const evalData = this._batchData(batchSize, step, trainingData, trainingLabels);
        const output = this.minstModel.predict(evalData.inputs);
        const predictions = output.argMax(1).dataSync();
        const labels = evalData.labels.argMax(1).dataSync();
        for (let i = 0; i < batchSize; i++) {
          if (predictions[i] === labels[i]) {
            correct++;
          }
          console.log(`Predictions[${i}]:`, predictions[i], ` - Label[${i}]:`, labels[i])
        }
        const accuracy = ((correct / batchSize) * 100).toFixed(2);
        console.log(`* Test set accuracy: ${accuracy}%\n`);
        this.setState({
          accuracy: accuracy,
          status: this.state.status + 1
        })
      }, 100)
    })
  }


  _minscPredictionKaggle = async (event) => {
    const { testData, numberOfTestData } = this.state
    this.setState({
      status: this.state.status + 1
    }, async () => {
      await setTimeout(async () => {
        const evalData = this._batchData(numberOfTestData, 0, testData, []);
        const output = this.minstModel.predict(evalData.inputs);
        const predictions = output.argMax(1).dataSync();
        for (let i = 0; i < numberOfTestData; i++) {
          console.log(`Kaggle predictions[${i}]:`, predictions[i])
        }
        this.setState({
          status: this.state.status + 1
        })
      }, 100)
    })
  }

  _clearCanvas = () => {
    this.canvas.clear()
  }

  _predictCanvas = async () => {
    const canvasData = this.canvas.getSaveData();
    let imageData = [];
    for (let indexX = 0; indexX < 28; indexX++) {
      imageData[indexX] = []
      for (let indexY = 0; indexY < 28; indexY++) {
        imageData[indexX][indexY] = 0
      }
    }

    await JSON.parse(canvasData).linesArray.forEach(item => {
      imageData[parseInt(item.startX / 10)][parseInt(item.startY / 10)] = 255
    })
    
    let newFlatInputData = []

    for (let indexX = 0; indexX < 28; indexX++) {
      for (let indexY = 0; indexY < 28; indexY++) {
        newFlatInputData.push(imageData[indexX][indexY])
      }
    }
    
    const imagesShape = [1, 28, 28, 1];

    const inputs = tf.tensor4d(newFlatInputData, imagesShape)

   const output = this.minstModel.predict(inputs);
   const prediction = output.argMax(1).dataSync();
   console.log('canvas prediction', prediction[0])
   this.setState({
     predictionNumber: prediction[0]
   })
  }


  render() {
    const { status, accuracy } = this.state
    return (
      <div className="App">
        <header className="App-header">
          <h1 className="App-title">
            Digit Recogniser Machine Learning 
            <br /> using Tensorflow Javascript and React
          </h1>
        </header>
        <div className="App-intro">
          <div className="App-training">
            <div className="section">
              <h1>
                Download Train Data (70MB)
              </h1>
              <p>
                First step is to download 70MB data of all the images with their labels
                <br />
                for training our model
                <br />
                you can find this data <a href="https://www.kaggle.com/c/digit-recognizer/data"> here </a>
              </p>
              { status === 0 &&
                <button onClick={this._readTrainingData}>
                  Download
                </button>
              }
              <br />
              { status === 1 &&
                <span className="red">
                  File is being Downloaded
                </span>
              }
              { status > 1 &&
                <span className="green">
                  File is downloaded and parsed
                </span>
              }
            </div>
            <br />
            { status > 1 &&
              <div className="section">
                <h1>
                  Initializing the SGD model
                </h1>
                <p>
                  We are using a sequential model with SGD training 
                </p>
                { status === 2 &&
                  <button onClick={this._initializingTheModel}>
                    Initialize
                  </button>
                }
                <br />
                { status === 3 &&
                  <span className="red">
                    Initializing
                  </span>
                }
                { status > 3 &&
                  <span className="green">
                    Initialized
                  </span>
                }
              </div>
             }
            <br />
            { status > 3 &&
              <div className="section">
                <h1>
                  Training the model using the data
                </h1>
                <p>
                  This might take a while to be completed 
                  <br />
                  our learning rate is 0.0001 and you can see the log in Inspect Element
                </p>
                { status === 4 &&
                  <button onClick={this._trainNewModel}>
                    Start Training
                  </button>
                }
                <br />
                { status === 5 &&
                  <span className="red">
                    Training
                  </span>
                }
                { status > 5 &&
                  <span className="green">
                    Trained
                  </span>
                }
              </div>
            }
            <br />
            { status > 5 &&
              <div className="section">
                <h1>
                  Predicting with test labeled data
                </h1>
                <p>
                  We are trying to predict based on our data
                </p>
                { status === 6 &&
                  <button onClick={this._minscPredictionLabeled}>
                    Start Predicting
                  </button>
                }
                <br />
               { status === 7 &&
                  <span className="red">
                    Predicting
                  </span>
                }
                { status > 7 &&
                  <span className="green">
                    Predicted
                  </span>
                }                
              </div>
            }
            { status > 7 &&
              <div className="section result">
                Accuracy : {accuracy}%
              </div>
            }
            { status > 5 &&
              <div className="section">
                <div className="canvas">
                  <div className="canvas-column">
                    <div className="canvas-canvas">
                      <CanvasDraw 
                        ref={e => this.canvas = e}
                        brushSize={50}
                        brushColor={"#444"}
                        canvasWidth={280}
                        canvasHeight={280}
                      />
                    </div>
                    <div className="canvas-buttons">
                      <button onClick={this._clearCanvas}>
                        Clear
                      </button>
                      <button onClick={this._predictCanvas}>
                        Predict
                      </button>
                    </div>
                  </div>
                  <div className="canvas-column">
                    <div className="prediction-column">
                      {this.state.predictionNumber}
                    </div>
                  </div>
                </div>
              </div>
            }
            { status > 7 &&
              <div className="section">
                <h1>
                  Download Train Data Kaggle (50MB) 
                </h1>
                <p>
                  Download kaggle data for testing
                  <br />
                  you can find this data <a href="https://www.kaggle.com/c/digit-recognizer/data"> here </a>
                </p>
                { status === 8 &&
                  <button onClick={this._readTestKaggleData}>
                    Download
                  </button>
                }
                <br />
                { status === 9 &&
                  <span className="red">
                    File Test Kaggle is being Downloaded
                  </span>
                }
                { status > 9 &&
                  <span className="green">
                    File Test Kaggle is downloaded and parsed
                  </span>
                }
              </div>
            }
            <br />
            { status > 9 &&
              <div className="section">
                <h1>
                  Predicting with test kaggle data
                </h1>
                <p>
                  In this step we try to solve the kaggle proble
                </p>
                { status === 10 &&
                  <button onClick={this._minscPredictionKaggle}>
                    Start Predicting
                  </button>
                }
                <br />
               { status === 11 &&
                  <span className="red">
                    Predicting
                  </span>
                }
                { status > 11 &&
                  <span className="green">
                    Predicted and prediction is in Console
                  </span>
                }
              </div>
            }
          </div>
        </div>
      </div>
    );
  }
}

export default App;

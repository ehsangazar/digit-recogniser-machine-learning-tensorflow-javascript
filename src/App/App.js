import React, { Component } from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';

class App extends Component {
  constructor(props){
    super(props)
  }

  componentDidMount() {
  }
  

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <h1 className="App-title">
            Digit Recogniser Machine Learning using Tensorflow Javascript and React
          </h1>
        </header>
        <p className="App-intro">
          App
        </p>
      </div>
    );
  }
}

export default App;

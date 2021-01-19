document.addEventListener('DOMContentLoaded', run);
//load();
//const dataPortal = 'https://auxplayer.com/?story=glorious-silver-crab&dataPortal='
//const dataPortal = 'https://wadelabs.casualos.com/?story=past-aquamarine-barnacle&dataPortal='
const dataPortal = 'http://192.168.0.46:3000/?story=stiff-silver-zebu&dataPortal='

async function run() {

  const data = await getData();
  // console.log(data);

  //Create the model
  const model = createModel();  
  tfvis.show.modelSummary({name: 'Model Summary'}, model);
  const tensorData = convertToTensor(data);
  const {inputs, labels} = tensorData;
  inputs.print();
  labels.print();

  // Train the model  
  await trainModel(model, inputs, labels);
  console.log('Done Training');
  // const weights = model.getWeights();
  // weights.print();
  
  const predictionData = await getPredictionData();
  //testModel(model, data, tensorData);
  let prediction = testModel(model, predictionData, tensorData);
  console.log(prediction);

  // sendData(prediction);
}

// Eventually add loading functionality
// async function load(){
//   let weightsUrl = 'https://casualos-playgroundfiles.s3.amazonaws.com/tfjs/';
//   //let weightsUrl = 'https://auxplayer.com/?story=opposite-cyan-trout&dataPortal=';

//   //const loadedModel = await tf.loadLayersModel('https://casualos-playgroundfiles.s3.amazonaws.com/model.json', {weightPathPrefix: weightsUrl});
//   const loadedModel = await tf.loadLayersModel('https://auxplayer.com/?story=opposite-cyan-trout&dataPortal=model.json', {weightPathPrefix: weightsUrl});
//   // const model = await tf.loadLayersModel('http://model-server.domain/download/model.json', {weightPathPrefix: weightsUrl});

//   if(loadedModel == null){
//     console.log("Model was not loaded");
//   }
//   else {
//     console.log("Model was loaded");
//     console.log(loadedModel);
//   }
//   const predictionData = await getPredictionData();
//   let prediction = testModel(loadedModel, predictionData);
//   console.log(prediction);
//   sendData(prediction);
// }

async function getData() {

  // const data = await fetch('https://auxplayer.com/?story=glorious-silver-crab&dataPortal=vibrationTraining.json');
  const data = await fetch(dataPortal + 'vibrationTraining.json');  
  const formattedData = await data.json(); 

  console.log(formattedData);

  return formattedData;

}

function createModel() {
  // Create a sequential model
  const model = tf.sequential(); 
  
  // Add a single input layer
  model.add(tf.layers.dense({inputShape: [1], units: 5, activation: "sigmoid", useBias: true}));
  
  // Add a another hidden layer - inputShape probably isn't needed
  model.add(tf.layers.dense({inputShape: [5], units: 2, activation: "sigmoid", useBias: true}));

  // Add an output layer
  model.add(tf.layers.dense({units: 2, useBias: true}));

  return model;
}

function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any 
  // intermediate tensors.
  
  return tf.tidy(() => {
    // Step 1. Shuffle the data    
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    // const inputTensor = tf.tensor2d(data.map(d => [d.red, d.green, d.blue]));
    const inputTensor = tf.tensor2d(data.map(d => [d.vibrationSD]));

    const labelTensor = tf.tensor2d(data.map(item => [
      item.classification === "good" ? 1 : 0,
      item.classification === "bad" ? 1 : 0,
    ]))
    
    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();  
    // const labelMax = labelTensor.max();
    // const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    //const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    //console.log('Here are the normalized inputs: ' + normalizedInputs);

    return {
      // inputs: inputTensor,
      // labels: labelTensor,

      inputs: normalizedInputs,
      // labels: normalizedLabels,
      labels: labelTensor,
      // // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin
      // labelMax,
      // labelMin,
    }
  });  
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.  
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });
  
  const batchSize = 32;
  const epochs = 50;
  
  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'], 
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

async function getPredictionData() {

  const data = await fetch(dataPortal + 'vibrationTesting.json');  
  const formattedData = await data.json(); 

  return formattedData;

}

function testModel(model, predictionData, normalizationData) {

  return tf.tidy( () => {
    
    const {inputMax, inputMin} = normalizationData;
    const testTensor = tf.tensor2d(predictionData.map(d => [d.vibrationSD]));
    console.log('actual prediction sds')
    testTensor.print();
    // console.log(testTensor.dataSync());
    // let test = model.predict(testTensor)
    // let testPredictions = test.dataSync();
    // console.log(testTensor.dataSync()[0])

    //This is wrong, should normalize based off of training data min and max
    // const predictionMax = testTensor.max();
    // const predictionMin = testTensor.min();
    const normalizedPredictionInputs = testTensor.sub(inputMin).div(inputMax.sub(inputMin));
    //const normalizedPredictionInputs = testTensor.sub(predictionMin).div(predictionMax.sub(predictionMin));
    console.log('normalized prediction sds');
    normalizedPredictionInputs.print();

    let predictionTensor = model.predict(normalizedPredictionInputs);
    console.log('predictions');
    predictionTensor.print();
    // let warmValue = predictionTensor.dataSync()[0];
    // let coldValue = predictionTensor.dataSync()[1];

    let predictionArray = [];

    // console.log(testTensor.dataSync().length);
    // console.log(predictionTensor.dataSync().length);

    for(i = 0; i < (testTensor.dataSync().length); i++){
      // let prediction = {
      //   red: testTensor.dataSync()[(i * 3)],
      //   green: testTensor.dataSync()[(i * 3) + 1],
      //   blue: testTensor.dataSync()[(i * 3) + 2],
      // }

      let prediction = {
        vibrationSD: testTensor.dataSync()[i]
      }

      if(predictionTensor.dataSync()[(i * 2)] > predictionTensor.dataSync()[(i * 2) + 1]){
        prediction.classification = 'good';
      }
      else {
        prediction.classification = 'bad';
      }
      // console.log(prediction);
      predictionArray.push(prediction);
    }

    return predictionArray

  });
  

}

// function sendData(prediction) {

//   fetch("https://auxplayer.com/webhook/?story=opposite-cyan-trout", {
//     method: 'POST',
//     body: JSON.stringify(prediction),
//     headers: {'Content-Type': 'application/json'}
//   })//.then(responseData => {
//     //console.log(responseData);
//     //})

// }

// async function saveModel(model){
//   await model.save("https://auxplayer.com/webhook/?story=opposite-cyan-trout")

//   // await model.save(tf.io.browserHTTPRequest(
//   //   'http://model-server.domain/upload',
//   //   {method: 'PUT', headers: {'header_key_1': 'header_value_1'} }));
// }

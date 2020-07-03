const nj = require('numjs');
const fs = require('fs');
const CSV = require('comma-separated-values');
const trainTestSplit = require('train-test-split');
const ConfusionMatrix = require('ml-confusion-matrix');
const _ = require('lodash');
var brain = require('brain.js');
var SLICE_INDEX = 1000

function brainToLabel(d) {
  return _.indexOf(_.values(d), _.max(_.values(d)))
}

let allData = CSV.parse(fs.readFileSync(('./data/train.csv'), 'utf-8')).slice(1, SLICE_INDEX)
let [trainData, validationData] = trainTestSplit(allData, 0.8, 1234)

let yTrainData = _.map(trainData, (d) => Number(d[0]))
let xTrainData = _.map(trainData, (d) => _.slice(d, 1))

let yValidationData = _.map(validationData, (d) => Number(d[0]))
let xValidationData = _.map(validationData, (d) => _.slice(d, 1))

xTrainData = _.map(xTrainData, (row) => _.map(row, (value) => value / 255))
xValidationData = _.map(xValidationData, (row) => _.map(row, (value) => value / 255))

// let model = new brain.NeuralNetworkGPU({ hiddenLayers: [480], activation: 'sigmoid' });
let model = new brain.NeuralNetwork({ hiddenLayers: [100], activation: 'sigmoid' })

model.train(
  _.map(xTrainData, (__, index) => ({ input: xTrainData[index], output: _.zipObject([yTrainData[index]], [1]) })), 
  {
    callback: (log) => console.log(log),
    callbackPeriod: 1,
    learningRate: 0.3
  }
)

let yResult = _.map(xValidationData, (d) => model.run(d)).map((v) => brainToLabel(v))
let CM2 = ConfusionMatrix.fromLabels(yValidationData, yResult)

console.log('Accuracy: ' + CM2.getAccuracy())

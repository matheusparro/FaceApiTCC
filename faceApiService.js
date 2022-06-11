const path = require("path");

const tf = require("@tensorflow/tfjs-node");

const faceapi = require("@vladmandic/face-api/dist/face-api.node.js");
const modelPathRoot = "./models";

let optionsSSDMobileNet;
let faceMatcher = null
async function image(file) {
  const decoded = tf.node.decodeImage(file);
  const casted = decoded.toFloat();
  const result = casted.expandDims(0);
  decoded.dispose();
  casted.dispose();
  return result;
}
function getFaceDetectorOptions() {
  return selectedFaceDetector === SSD_MOBILENETV1
    ? new faceapi.SsdMobilenetv1Options({ minConfidence })
    : new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold })
}

async function detect(tensor) {
  // ssd_mobilenetv1 options
  const SSD_MOBILENETV1 = 'ssd_mobilenetv1'
const TINY_FACE_DETECTOR = 'tiny_face_detector'


let selectedFaceDetector = SSD_MOBILENETV1

// ssd_mobilenetv1 options

// tiny_face_detector options
let inputSize = 512
let scoreThreshold = 0.5

let minConfidence = 0.5
try{
  const result = await faceapi.detectAllFaces(tensor, new faceapi.SsdMobilenetv1Options({ minConfidence })).withFaceLandmarks().withFaceDescriptors()
  if (!faceMatcher) faceMatcher = new faceapi.FaceMatcher(result)
  
  const displaySize = {width: '100%', hegiht: '100%'}

  const label = faceMatcher && result ? faceMatcher.findBestMatch(result[0].descriptor).toString():null
  //const resizedResults = faceapi.resizeResults(result, displaySize)
  if(label){
    const labelFormated = label.split(" ")
    if(labelFormated[0] == 'person'){
      return true
    }
  }
  return false
}catch(err){
  return false
}
}

async function main(file) {
  // console.log("FaceAPI single-process test");

  await faceapi.tf.setBackend("tensorflow");
  await faceapi.tf.enableProdMode();
  await faceapi.tf.ENV.set("DEBUG", false);
  await faceapi.tf.ready();

  // console.log(
  //   `Version: TensorFlow/JS ${faceapi.tf?.version_core} FaceAPI ${
  //     faceapi.version.faceapi
  //   } Backend: ${faceapi.tf?.getBackend()}`
  // );

  // console.log("Loading FaceAPI models");
  const modelPath = path.join(__dirname, modelPathRoot);
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath),
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath),

  optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options({
    minConfidence: 0.5,
  });

  const tensor = await image(file);
  const result = await detect(tensor);

  tensor.dispose();

  return result;
}

module.exports = {
  detect: main,
};
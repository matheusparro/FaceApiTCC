const express = require("express");
const fileUpload = require("express-fileupload");
const faceApiService = require('./faceApiService')
const app = express();
const port = process.env.PORT || 3000;

app.use(fileUpload());

app.post("/upload", async (req, res) => {
  if(!req.files){
    res.status(500).send({ error: 'Something failed!' });
  }
  const { file,file2 } = req.files;

  const result = await faceApiService.detect(file.data);
  const result2 = await faceApiService.detect(file2.data);
  res.json({
    isEqualFaces: result2,
  });
});
app.listen(port, () => {
  console.log("Server started on port" + port);
});
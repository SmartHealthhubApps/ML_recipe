import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:image/image.dart' as img;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text('Blur Faces Example')),
        body: Center(child: Text('Blur Faces Example')),
      ),
    );
  }
}

Future<img.Image> blurFaces(ui.Image image) async {
  // Convert ui.Image to bytes
  final ByteData? byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
  if (byteData == null) {
    throw Exception('Unable to convert ui.Image to byte data');
  }
  
  final Uint8List imageBytes = byteData.buffer.asUint8List();

  // Convert to input image for ML Kit
  final InputImage visionImage = InputImage.fromBytes(
    bytes: imageBytes,
    inputImageData: InputImageData(
      size: Size(image.width.toDouble(), image.height.toDouble()),
      imageRotation: InputImageRotation.rotation0,
      inputImageFormat: InputImageFormat.nv21,
      planeData: [
        InputImagePlaneMetadata(
          bytesPerRow: image.width,
          height: image.height,
          width: image.width,
        ),
      ],
    ),
  );

  // Initialize the face detector
  final FaceDetector faceDetector = GoogleMlKit.vision.faceDetector(
    FaceDetectorOptions(
      enableContours: false,
      enableLandmarks: false,
      enableClassification: false,
      enableTracking: false,
    ),
  );

  // Detect faces
  final List<Face> faces = await faceDetector.processImage(visionImage);

  // Decode image using the image package
  img.Image imgImage = img.decodeImage(imageBytes)!;

  // Blur faces
  for (Face face in faces) {
    final boundingBox = face.boundingBox;
    final faceRegion = img.copyCrop(
      imgImage,
      boundingBox.left.toInt(),
      boundingBox.top.toInt(),
      boundingBox.width.toInt(),
      boundingBox.height.toInt(),
    );

    final blurredFace = img.gaussianBlur(faceRegion, 25);
    imgImage = img.copyInto(
      imgImage,
      blurredFace,
      dstX: boundingBox.left.toInt(),
      dstY: boundingBox.top.toInt(),
    );
  }

  // Clean up
  faceDetector.close();

  return imgImage;
}


package com.example.myapplication;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.ProgressDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.util.SparseArray;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;


import com.github.mikephil.charting.charts.HorizontalBarChart;
import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;
import com.google.android.gms.vision.face.Landmark;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.github.mikephil.charting.data.BarData;
import com.github.mikephil.charting.data.BarDataSet;
import com.github.mikephil.charting.data.BarEntry;
import com.github.mikephil.charting.utils.ColorTemplate;


public class MainActivity extends AppCompatActivity {
    //-----------------------------LANDMARKS--------------------------------
//    private float Left_Mouth_x , Left_Mouth_y , Right_Mouth_x
//            , Right_Mouth_y , Bottom_Mouth_x , Bottom_Mouth_y   ;

    //----------------
//    private PointF Nose ,Right_Cheek ,Left_Cheek, Left_Ear, Right_Ear,Right_Ear_Tip , Left_Ear_Tip;
    //-----------------------------------------------------------------------
    //-------------probabilities---------------
//    private double P_MO , P_R , P_H , P_SA , P_N ,P_SU , P_F ,P_A ;
    //-----------------------------------------

    // layout attributes

    private ImageView showimg;
    private ImageView btnpicture;
    private ImageView charthelp ;
    private Bitmap imageBitmap= null;
    public Bitmap bitmap_1 ,bitmaaap=null;
    private static final int CAMERA_REQUEST = 1888;
    private static final int MY_CAMERA_PERMISSION_CODE = 100;
    public ProgressDialog progressDialog ;
    public ProgressBar progressBar;
    public ImageView cameraview;

    // operational attribute

    Face[] facesDetected;
    private int k = 1,r_c;
    public SparseArray<Face> sparseArray ;
    public String imagestring ="";
    public String [] strArr_EmV ;


    public String result;
    public int size;
    public boolean imageselected;

    /**
     * Number of results to show in the UI.
     *
     *
     */
    private static final float IMAGE_MEAN = 0f;

    private static final float IMAGE_STD = 255f;

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private static final float PROBABILITY_MEAN = 0.0f;

    private static final float PROBABILITY_STD = 1.0f;

    private static final int MAX_RESULTS = 3;


    /**
     * The loaded TensorFlow Lite model.
     */
    private MappedByteBuffer tfliteModel;

    /**
     * Image size along the x axis.
     */
    private int imageSizeX;

    /**
     * Image size along the y axis.
     */
    private int imageSizeY;

    /**
     * Optional GPU delegate for accleration.
     */
    private GpuDelegate gpuDelegate = null;

    /**
     * Optional NNAPI delegate for accleration.
     */
    private NnApiDelegate nnApiDelegate = null;

    /**
     * An instance of the driver class to run model inference with Tensorflow Lite.
     */
    protected Interpreter tflite;

    /**
     * Options for configuring the Interpreter.
     */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /**
     * Labels corresponding to the output of the vision model.
     */
    private List<String> labels;

    /**
     * Input image TensorBuffer.
     */
    private TensorImage inputImageBuffer;

    /**
     * Output probability TensorBuffer.
     */
    private TensorBuffer outputProbabilityBuffer;

    /**
     * Processer to apply post processing of the output probability.
     */
    private TensorProcessor probabilityProcessor;

    //--------charts---------
    HorizontalBarChart chart ;
    ArrayList<BarEntry> BARENTRY ;
    ArrayList<String> BarEntryLabels ;
    BarDataSet Bardataset ;
    BarData BARDATA ;

    @SuppressLint("WrongConstant")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //---------------------Toolbar----------------------------
        //hiding the default toolbar

        getSupportActionBar().hide();

        //-------------------------------------------------

        cameraview = (ImageView) findViewById(R.id.cameravideo);
        showimg = (ImageView) findViewById(R.id.showimg);
        btnpicture = (ImageView) findViewById(R.id.btnpicture);
//        textView.setMovementMethod(new ScrollingMovementMethod());
        //-----------------Charts--------------------
        chart = (HorizontalBarChart) findViewById(R.id.chart1);
        charthelp = (ImageView) findViewById(R.id.charthelp);
        chart.setVisibility(View.INVISIBLE);
        charthelp.setVisibility(View.VISIBLE);


        //----------------------------------------------
        Bitmap mBitmap = BitmapFactory.decodeResource(getApplicationContext().getResources(),R.raw.image);
        showimg.setImageBitmap(mBitmap);

        //processing
        try {
            tfliteModel = FileUtil.loadMappedFile(getApplicationContext(), "emotion_resnet256.tflite");
        } catch (IOException e) {
            e.printStackTrace();
        }

        //tfliteOptions.setNumThreads(numThreads);
        tflite = new Interpreter(tfliteModel, tfliteOptions);

        // Loads labels out from the label file.
        try {
            labels = FileUtil.loadLabels(getApplicationContext(), "emotion_label.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }


        // ---------------Create ProgressDialog----------------

        progressDialog = new ProgressDialog(MainActivity.this);
        progressDialog.setMessage("Loading..."); // Setting Message
        progressDialog.setTitle("ProgressDialog"); // Setting Title
        progressDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER); // Progress Dialog Style Spinner
        progressDialog.setIndeterminate(true);
        progressDialog.setCancelable(false);


        // -----------------------------------------------------
        cameraview.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(getApplicationContext(), cameraview.class));
            }
        });


        btnpicture.setOnClickListener(new View.OnClickListener() {

            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View v) {

                BARENTRY = new ArrayList<>();
                BarEntryLabels = new ArrayList<String>();

                if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
                {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, MY_CAMERA_PERMISSION_CODE);
                }


                //------------------------
                imageselected=false;
                selectImage();
                //------------------------




            }
        });

    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults)
    {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MY_CAMERA_PERMISSION_CODE)
        {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED)
            {
                Toast.makeText(this, "camera permission granted", Toast.LENGTH_LONG).show();
//                Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
//                startActivityForResult(cameraIntent, CAMERA_REQUEST);
            }
            else
            {
                Toast.makeText(this, "camera permission denied", Toast.LENGTH_LONG).show();
            }
        }
    }


    public void selectImage()
    {
        final CharSequence[] options = { "Take Photo", "Choose from Gallery","Cancel" };
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        builder.setTitle("Add Photo!");
        builder.setItems(options,new DialogInterface.OnClickListener() {

            @Override
            public void onClick(DialogInterface dialog, int which) {
                // TODO Auto-generated method stub

                if(options[which].equals("Take Photo"))
                {
                    //To take picture from camera
                    Intent takePicture = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(takePicture, 0);
                }
                else if(options[which].equals("Choose from Gallery"))
                {
                    //To pick photo from gallery
                    Intent pickPhoto = new Intent(Intent.ACTION_PICK,
                            android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                    startActivityForResult(pickPhoto, 1);
                }
                else if(options[which].equals("Cancel"))
                {
                    dialog.dismiss();
                }



            }
        });
        builder.show();

    }

    protected void onActivityResult(int requestCode, int resultCode, Intent imageReturnedIntent) {
        super.onActivityResult(requestCode, resultCode, imageReturnedIntent);
        r_c=requestCode;
        switch (requestCode) {
            case 0:
                if (resultCode == RESULT_OK) {
                    Bundle extras = imageReturnedIntent.getExtras();
                    imageBitmap = (Bitmap) extras.get("data");
                    showimg.setImageBitmap(imageBitmap);
                    imageselected=true;
                    chart.setVisibility(View.VISIBLE);
                    charthelp.setVisibility(View.INVISIBLE);

                }

                break;
            case 1:
                if (resultCode == RESULT_OK) {
                    Uri selectedImage = imageReturnedIntent.getData();
                    try {
                        ParcelFileDescriptor parcelFileDescriptor =
                                getContentResolver().openFileDescriptor(selectedImage, "r");
                        FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
                        imageBitmap = BitmapFactory.decodeFileDescriptor(fileDescriptor);
                        //rotate image bitmap
                        Matrix matrix = new Matrix();
                        //vertically(1,-1) || horizontally(-1,1)
                        //matrix.preScale(1.0f, -1.0f);
                        //rotate degree=-90
                        matrix.setRotate(-90);
                        imageBitmap = Bitmap.createBitmap(imageBitmap, 0, 0, imageBitmap.getWidth(), imageBitmap.getHeight(), matrix, true);
                        showimg.setImageBitmap(imageBitmap);
                        imageselected=true;
                        chart.setVisibility(View.VISIBLE);
                        charthelp.setVisibility(View.INVISIBLE);
                        parcelFileDescriptor.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                }
                break;
        }
        if(imageselected){

            DetectActivity();
        }

    }

    //------------------------------

    public void DetectActivity(){
        if (imageBitmap != null) {
            //-----progressDialog-----
            progressDialog.show(); // Display Progress Dialog

            //----SET_IMAGE-----
            Matrix matrix = new Matrix();
            DetectFace(imageBitmap, showimg);
            while (sparseArray.size() == 0 && k < 4) {
                matrix.setRotate(-90 * k);
                imageBitmap = Bitmap.createBitmap(imageBitmap, 0, 0, imageBitmap.getWidth(), imageBitmap.getHeight(), matrix, true);
                showimg.setImageBitmap(imageBitmap);
                //la fonction detect face
                //DetectFace();
                DetectFace(imageBitmap, showimg);
                k++;
            }
            k = 1;
        }


        if (size == 0) {
            Toast.makeText(getApplicationContext(), "No face detected in picture.", Toast.LENGTH_SHORT).show();
            charthelp.setVisibility(View.VISIBLE);
            chart.setVisibility(View.INVISIBLE);
            progressDialog.dismiss();

        }

        //-------------------------------------------------------



    }
    //  processing

    @RequiresApi(api = Build.VERSION_CODES.JELLY_BEAN_MR2)
    void DetectFace(Bitmap bitmap_x, ImageView btn) {

            //
            Bitmap newBitmap = Bitmap.createBitmap(bitmap_x.getWidth(), bitmap_x.getHeight(), Bitmap.Config.RGB_565);
            Canvas ctx = new Canvas(newBitmap);
            ctx.drawBitmap(bitmap_x, 0, 0, null);
            //----boxpaint------
            Paint boxPaint = new Paint();
            boxPaint.setColor(Color.GREEN);
            boxPaint.setStyle(Paint.Style.STROKE);

            //---------
            Paint paint_FN = new Paint();
            paint_FN.setColor(Color.RED);
            paint_FN.setStyle(Paint.Style.FILL);

            //----------
            Paint paint_L = new Paint();
            paint_L.setColor(Color.GREEN);
            paint_L.setStyle(Paint.Style.FILL);


            //----------


            //----------    processing
            FaceDetector facedetector = new FaceDetector.Builder(getApplicationContext())
                .setTrackingEnabled(false)
                .setLandmarkType(FaceDetector.ALL_LANDMARKS)
                .setClassificationType(FaceDetector.ALL_CLASSIFICATIONS)
                .setMode(FaceDetector.FAST_MODE)
                .build();

            if (!facedetector.isOperational()) {
                Toast.makeText(getApplicationContext(), "the current Google Play Services library in your phone dosn't supports the vision API.", Toast.LENGTH_SHORT).show();
                return;
            }

            Frame frame = new Frame.Builder().setBitmap(bitmap_x).build();


            sparseArray = facedetector.detect(frame);
//            textView.append(sparseArray.size() + " faces detected \n");
            size = sparseArray.size();

            if(size>1){
                chart.setDescription("Face 1");
            }
            else{
                chart.setDescription("");
            }

            int faceresult = 1 ;

            for (int i = 0; i < sparseArray.size(); i++) {
                Face face = sparseArray.valueAt(i);
                float x = face.getPosition().x;
                float y = face.getPosition().y;
                float h = y + face.getHeight();
                float w = x + face.getWidth();
                //--------------------
//                  bitmaaap = newBitmap;
//                bitmaaap = Bitmap.createBitmap(newBitmap, (int) face.getPosition().x, (int) face.getPosition().y, (int) face.getWidth(), (int) face.getHeight());
//                final long startTime = SystemClock.uptimeMillis();
//                Bitmap croppedImage = BitmapUtils.cropBitmap( bitmap,   face.getBoundingBox());


                int j = i + 1;
                RectF rectF = new RectF(x, y, w, h);
                //--------------------
                boxPaint.setStrokeWidth(5f);
                paint_FN.setTextSize(100);
                if(r_c==1){
                    boxPaint.setStrokeWidth(5f);
                    paint_FN.setTextSize(100);
                }
                else {
                    boxPaint.setStrokeWidth(1.5f);
                    paint_FN.setTextSize(20);
                }

                //--------------------

                ctx.drawRoundRect(rectF, 2, 2, boxPaint);
                //-------------------
                double viewWidth = ctx.getWidth();
                double viewHeight = ctx.getHeight();
                double imageWidth = bitmap_x.getWidth();
                double imageHeight = bitmap_x.getHeight();
                double scale = Math.min(viewWidth / imageWidth, viewHeight / imageHeight);
                //-------------------
                float cx = (float) (face.getPosition().x * scale);
                float cy = (float) (face.getPosition().y * scale);
                if(size>1){
                    ctx.drawText("Face" + j, cx, cy + 10.0f, paint_FN);

                }

                //----------------------------
//                textView.setText("happiness: " + String.format("%.2f", face.getIsSmilingProbability()) + "\n" );
//                textView.append("right eye: " + String.format("%.2f", face.getIsRightEyeOpenProbability())+"\n");
//                textView.append("left eye: " + String.format("%.2f", face.getIsLeftEyeOpenProbability())+"\n");
                float smilingProbability = face.getIsSmilingProbability();
                float leftEyeOpenProbability = face.getIsLeftEyeOpenProbability();
                float rightEyeOpenProbability = face.getIsRightEyeOpenProbability();
                float eulerY = face.getEulerY();
                float eulerZ = face.getEulerZ();

//                textView.append("Face " + j + " : ");
//                textView.append("Smiling: " + smilingProbability + "   ");
//                textView.append("Left eye open: " + leftEyeOpenProbability + "   ");
//                textView.append("Right eye open: " + rightEyeOpenProbability + "   ");
//                textView.append("Euler Y: " + eulerY + "   ");
//                textView.append("Euler Z: " + eulerZ + "\n");

                //All landmarks (draw points)
                for (Landmark landmark : face.getLandmarks()) {
                    int cx2 = (int) (landmark.getPosition().x * scale);
                    int cy2 = (int) (landmark.getPosition().y * scale);

                    if (r_c == 1) {
                        ctx.drawCircle(cx2, cy2, 5, paint_L);
                    } else {
                        ctx.drawCircle(cx2, cy2, 1.5f, paint_L);

                    }
                }
                //-------------------------------
                if(faceresult==1) {
                    faceresult = 0;
                    bitmaaap = Bitmap.createBitmap(newBitmap, (int) face.getPosition().x, (int) face.getPosition().y, (int) face.getWidth(), (int) face.getHeight());

                    int imageTensorIndex = 0;
                    int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
                    imageSizeY = imageShape[1];
                    imageSizeX = imageShape[2];
                    DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

                    int probabilityTensorIndex = 0;
                    int[] probabilityShape =
                            tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
                    DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

                    inputImageBuffer = new TensorImage(imageDataType);
                    outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
                    probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();
                    if (bitmaaap != null) {
                        inputImageBuffer = loadImage(bitmaaap);

                        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());

                        showresult();

                    }
                    //-------This method dismiss the progressdialog.--------
                    progressDialog.dismiss();
                    //-------------------------------------------------------
                }

                /*

                for (Landmark landmark : face.getLandmarks()) {
                    switch (landmark.getType()) {
                        case Landmark.LEFT_MOUTH:
                            Left_Mouth_x = landmark.getPosition().x;
                            Left_Mouth_y = landmark.getPosition().y;

                            break;
                        case Landmark.RIGHT_MOUTH:
                            Right_Mouth_x = landmark.getPosition().x;
                            Right_Mouth_y = landmark.getPosition().y;

                            break;
                        case Landmark.BOTTOM_MOUTH:
                            Bottom_Mouth_x = landmark.getPosition().x;
                            Bottom_Mouth_y = landmark.getPosition().y;

                            break;
                        case Landmark.NOSE_BASE:
                            Nose = landmark.getPosition();
                            break;
                        case Landmark.RIGHT_CHEEK:
                            Right_Cheek = landmark.getPosition();
                            break;
                        case Landmark.LEFT_CHEEK:
                            Left_Cheek = landmark.getPosition();
                            break;
                        case Landmark.LEFT_EAR:
                            Left_Ear = landmark.getPosition();
                            break;
                        case Landmark.RIGHT_EAR:
                            Right_Ear = landmark.getPosition();
                            break;
                        case Landmark.RIGHT_EAR_TIP:
                            Right_Ear_Tip = landmark.getPosition();
                            break;
                        case Landmark.LEFT_EAR_TIP:
                            Left_Ear_Tip = landmark.getPosition();
                            break;

                    }
                }

                //----------mouth open or closed-----------
//    Left mouth  ______a______ Right mouth
//                \ Cc   Bb  /
//                 \        /
//                  \      /
//                 b \ Aa /c
//                    \  /
//                     \/
//                Bottom_Mouth

                //calculating of a,b and c values

                double a = Math.sqrt(Math.pow((Left_Mouth_x - Right_Mouth_x), 2) + Math.pow((Left_Mouth_y - Right_Mouth_y), 2));
                double b = Math.sqrt(Math.pow((Left_Mouth_x - Bottom_Mouth_x), 2) + Math.pow((Left_Mouth_y - Bottom_Mouth_y), 2));
                double c = Math.sqrt(Math.pow((Right_Mouth_x - Bottom_Mouth_x), 2) + Math.pow((Right_Mouth_y - Bottom_Mouth_y), 2));

                //----Now let's find the (Aa)° Angle ...
                // degree(AA) = arccos[(b² + c² − a²) ÷ 2bc]
                double degree_Aa = Math.acos((b * b + c * c - a * a) / (2 * b * c)) * (180 / Math.PI);
                double degree_Bb = Math.acos((a * a + c * c - b * b) / (2 * a * c)) * (180 / Math.PI);
                double degree_Cc = Math.acos((a * a + b * b - c * c) / (2 * a * b)) * (180 / Math.PI);

                if (degree_Aa <= 124  && degree_Bb >= 27 && degree_Cc >= 27) {
                    textView.append("---the degree of mouth is :" + degree_Aa + "/" + degree_Bb + "/" + degree_Cc + "\n ---Mouth is open \n");
                } else {
                    textView.append("---the degree of mouth is :" + degree_Aa + "/" + degree_Bb + "/" + degree_Cc + "\n ---Mouth is closed \n");
                }
                //----------------

                //----------mouth open or closed-----------
//    Left_Check  _____aa_____ Right_Check
//                \          /
//                 \        /
//                  \      /
//                bb \ Aa1/ cc
//                    \  /
//                     \/
//                Nose

                //calculating of aa,bb and cc values

                double aa = Math.sqrt(Math.pow((Left_Cheek.x - Right_Cheek.x), 2) + Math.pow((Left_Cheek.y - Right_Cheek.y), 2));
                double bb = Math.sqrt(Math.pow((Left_Cheek.x - Nose.x), 2) + Math.pow((Left_Cheek.y - Nose.y), 2));
                double cc = Math.sqrt(Math.pow((Right_Cheek.x - Nose.x), 2) + Math.pow((Right_Cheek.y - Nose.y), 2));

                //----Now let's find the (Aa1)° Angle ...
                // degree(Aa1) = arccos[(b² + c² − a²) ÷ 2bc]
                double degree_Aa1 = Math.acos((bb * bb + cc * cc - aa * aa) / (2 * bb * cc)) * (180 / Math.PI);

                //----------------
//                textView.append("\n Degree nose/check : " + degree_Aa1 + "\n");
//                //----------------
//                textView.append("Nose : "+ Nose
//                        +"\n Right_Cheek : " + Right_Cheek
//                        +"\n left_Cheek : "+ Left_Cheek
//                        +"\n Left_Ear : "+ Left_Ear
//                        +"\n Right_Ear : "+ Right_Ear
//                        +"\n Right_Ear_Tip :"+ Right_Ear_Tip
//                        +"\n Left_Ear_Tip : " + Left_Ear_Tip +"\n");

                //------------probability of mouth open---------

                if (degree_Aa <= 91)
                    P_MO = 1;
                if (degree_Aa > 91 && degree_Aa <= 92)
                    P_MO = 0.98;
                if (degree_Aa > 92 && degree_Aa <= 93)
                    P_MO = 0.97;
                if (degree_Aa > 93 && degree_Aa <= 94)
                    P_MO = 0.96;
                if (degree_Aa > 94 && degree_Aa <= 95)
                    P_MO = 0.95;
                if (degree_Aa > 95 && degree_Aa <= 96)
                    P_MO = 0.94;
                if (degree_Aa > 96 && degree_Aa <= 97)
                    P_MO = 0.93;
                if (degree_Aa > 97 && degree_Aa <= 98)
                    P_MO = 0.92;
                if (degree_Aa > 98 && degree_Aa <= 99)
                    P_MO = 0.91;
                if (degree_Aa > 99 && degree_Aa <= 100)
                    P_MO = 0.90;
                if (degree_Aa > 100 && degree_Aa <= 101)
                    P_MO = 0.89;
                if (degree_Aa > 101 && degree_Aa <= 102)
                    P_MO = 0.88;
                if (degree_Aa > 102 && degree_Aa <= 103)
                    P_MO = 0.87;
                if (degree_Aa > 103 && degree_Aa <= 104)
                    P_MO = 0.86;
                if (degree_Aa > 104 && degree_Aa <= 105)
                    P_MO = 0.85;
                if (degree_Aa > 105 && degree_Aa <= 106)
                    P_MO = 0.84;
                if (degree_Aa > 106 && degree_Aa <= 107)
                    P_MO = 0.83;
                if (degree_Aa > 107 && degree_Aa <= 108)
                    P_MO = 0.82;
                if (degree_Aa > 108 && degree_Aa <= 109)
                    P_MO = 0.81;
                if (degree_Aa > 109 && degree_Aa <= 110)
                    P_MO = 0.77;
                if (degree_Aa > 110 && degree_Aa <= 111)
                    P_MO = 0.73;
                if (degree_Aa > 111 && degree_Aa <= 112)
                    P_MO = 0.71;//---
                if (degree_Aa > 112 && degree_Aa <= 113)
                    P_MO = 0.65;
                if (degree_Aa > 113 && degree_Aa <= 114)
                    P_MO = 0.60;
                if (degree_Aa > 114 && degree_Aa <= 115)
                    P_MO = 0.55;
                if (degree_Aa > 115 && degree_Aa <= 116)
                    P_MO = 0.50;
                if (degree_Aa > 116 && degree_Aa <= 117)
                    P_MO = 0.46;
                if (degree_Aa > 117 && degree_Aa <= 118)
                    P_MO = 0.43;
                if (degree_Aa > 118 && degree_Aa <= 119)
                    P_MO = 0.42;
                if (degree_Aa > 119 && degree_Aa <= 120)
                    P_MO = 0.39;
                if (degree_Aa > 120 && degree_Aa <= 121)
                    P_MO = 0.36;
                if (degree_Aa > 121 && degree_Aa <= 122)
                    P_MO = 0.33;
                if (degree_Aa > 122 && degree_Aa <= 123)
                    P_MO = 0.30;
                if (degree_Aa > 123 && degree_Aa <= 124)
                    P_MO = 0.27;
                if (degree_Aa > 124 && degree_Aa <= 125)
                    P_MO = 0.18;
                if (degree_Aa > 125 && degree_Aa <= 126)
                    P_MO = 0.15;
                if (degree_Aa > 126 && degree_Aa <= 127)
                    P_MO = 0.12;
                if (degree_Aa > 127 && degree_Aa <= 128)
                    P_MO = 0.9;
                if (degree_Aa > 128 && degree_Aa <= 129)
                    P_MO = 0.6;
                if (degree_Aa > 129 && degree_Aa <= 130)
                    P_MO = 0.3;
                if (degree_Aa > 130)
                    P_MO = 0;

                //--------Probabilities----------
                if(rightEyeOpenProbability<0.3 && leftEyeOpenProbability<0.3) {
                    P_H = (1 - rightEyeOpenProbability) * (1 - leftEyeOpenProbability) * smilingProbability;
                }
                else {
                    P_H = smilingProbability;
                }

                P_SA = 1-P_H;

                if(P_H>=0.5){
                    P_N  = P_H * (1-P_MO) ;
                    P_H  = P_H - P_N;
                    if(P_MO < 0.80 ) {
                        P_SU = P_N * (P_MO + 0.2 ) ;
                    }else{
                        P_SU = P_N * P_MO ;
                    }

                    P_N = P_N - P_SU ;
                    P_F  = P_SU * rightEyeOpenProbability *rightEyeOpenProbability ;
                    P_SU = P_SU - P_F ;
                }
                else{
                    P_N  = P_SA * (1-P_MO) ;
                    P_SA  = P_SA - P_N;
                    if(P_MO < 0.80 ) {
                        P_SU = P_N * (P_MO + 0.2 ) ;
                    }else{
                        P_SU = P_N * P_MO ;
                    }
                    P_N = P_N - P_SU ;
                    P_F  = P_SU * rightEyeOpenProbability *rightEyeOpenProbability ;
                    P_SU = P_SU - P_F ;
                }

                P_A  = P_SA * P_MO  ;
                P_SA = P_SA - P_A;
                //-------------Show results-------------
*/             //labels = {"Angry","Happy","Neutral","Sad","Surprise"}

//                if(sparseArray.size()!=0) {
//                    //--------find the max probability--------
//                    int indexMaxProba = 0;
//                    for(int k = 1 ; k < 5 ; k++ ){
//                        if( Float.parseFloat(strArr_EmV[indexMaxProba]) < Float.parseFloat(strArr_EmV[k]) ){
//                            indexMaxProba = k ;
//                        }
//                    }
//                    textView.append("\n Happiness probability : " + Float.parseFloat(strArr_EmV[1])*100 + "%"
//                            + "\n Sadness probability : " + Float.parseFloat(strArr_EmV[3])*100 + "%"
//                            + "\n Neutral probability : " + Float.parseFloat(strArr_EmV[2])*100 + "%"
//                            + "\n Angry probability : " + Float.parseFloat(strArr_EmV[0])*100 + "%"
//                            + "\n Surprised probability : " + Float.parseFloat(strArr_EmV[4])*100 + "%"
////                            + "\n Fear probability :" + Float.parseFloat(strArr_EmV[0])*100 + "%"
//                            + "\n The dominated emotion is : " + labels[indexMaxProba]
//                            + "\n");
//
//
//                }

                        //----------------------------------------------

            }

            btn.setImageDrawable(new BitmapDrawable(getResources(), newBitmap));
//            btn.setImageDrawable(new BitmapDrawable(getResources(), bitmaaap));


    }
    //---------------------------------------------
    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        Bitmap bitmap01 = bitmap.copy(Bitmap.Config.ARGB_8888,true);
        inputImageBuffer.load(bitmap01);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    private MappedByteBuffer loadmodelfile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor=activity.getAssets().openFd("emotion_resnet256.tflite");
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startoffset = fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startoffset,declaredLength);
    }

    private TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }
    private TensorOperator getPostprocessNormalizeOp(){
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    private void showresult(){

        BARENTRY = new ArrayList<>();
        BarEntryLabels = new ArrayList<String>();

        Map<String, Float> labeledProbability =
                new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                        .getMapWithFloatValue();
        float maxValueInMap =(Collections.max(labeledProbability.values()));
        int xIndex = 0 ;
        String str ;
//        Map.Entry<String, Float> maxEntry = null;
//
//        for (Map.Entry<String, Float> entry : labeledProbability.entrySet())
//        {
//            if (maxEntry == null || entry.getValue().compareTo(maxEntry.getValue()) > 0)
//            {
//                maxEntry = entry;
//            }
//        }
        for (Map.Entry<String, Float> entry : labeledProbability.entrySet()) {
            switch (entry.getKey()){
                case "Surprised" :
                    str =" \uD83D\uDE2E";
                    break;
                case "Sad" :
                    str= "\uD83D\uDE14";
                    break;
                case "Neutral" :
                    str= "\uD83D\uDE10";
                    break;
                case "Happy" :
                    str= "\uD83D\uDE03";
                    break;
                case "Fearful" :
                    str= "\uD83D\uDE31";
                    break;
                case "Disgusted" :
                    str= "\uD83E\uDD2E";
                    break;
                case "Angry" :
                    str= "\uD83E\uDD2C";
                    break;
                case "Uncertain" :
                    str= "\uD83E\uDD14";
                    break;
                default:
                    str="";
                    break;

            }

            BARENTRY.add(new BarEntry(entry.getValue()*100, xIndex));
            BarEntryLabels.add(str + " " + entry.getKey());

            xIndex++;
        }
        Bardataset = new BarDataSet(BARENTRY, "Emotions");
        BARDATA = new BarData(BarEntryLabels, Bardataset);
        Bardataset.setColors(ColorTemplate.COLORFUL_COLORS);
        chart.setData(BARDATA);
        chart.getLegend().setEnabled(false);
        chart.setDragEnabled(true); // on by default
        chart.setVisibleXRange( 8,8 ); // sets the viewport to show 3 bars
        chart.animateY(3000);


    }


 }


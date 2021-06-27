package com.example.myapplication;

import com.github.mikephil.charting.charts.HorizontalBarChart;
import com.github.mikephil.charting.data.BarData;
import com.github.mikephil.charting.data.BarDataSet;
import com.github.mikephil.charting.data.BarEntry;
import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;
import com.google.android.gms.vision.face.Landmark;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import android.annotation.SuppressLint;
import android.annotation.TargetApi;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.graphics.drawable.BitmapDrawable;
import android.hardware.Camera;
import android.hardware.Camera.Parameters;
import android.hardware.Camera.PreviewCallback;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.os.Build;
import android.os.Bundle;

import androidx.annotation.RequiresApi;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.util.SparseArray;
import android.view.KeyEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

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

import com.github.mikephil.charting.utils.ColorTemplate;


public class cameraview extends Activity  implements SurfaceHolder.Callback,PreviewCallback{

    private SurfaceView surfaceview;

    private SurfaceHolder surfaceHolder;

    private Camera camera;

    private Parameters parameters;

    int width = 640;

    int height = 480;

    int framerate = 24;

    int biterate = 8500*1000;
    int FRONT = 1 ;
    int  BACK = 0 ;
    int F_or_B = FRONT;
    public SparseArray<Face> sparseArray ;
    private static int yuvqueuesize = 10;

    public static ArrayBlockingQueue<byte[]> YUVQueue = new ArrayBlockingQueue<byte[]>(yuvqueuesize);

    private AvcEncoder avcCodec;
    private final static int CAMERA_OK = 10001;
    public Button SwitchCam ;

    private static String[] PERMISSIONS_STORAGE = {
            "android.permission.CAMERA",
            "android.permission.WRITE_EXTERNAL_STORAGE" };

    private static final float IMAGE_MEAN = 0f;
    private static final float IMAGE_STD = 255f;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 1.0f;
    private static final int MAX_RESULTS = 3;
    private MappedByteBuffer tfliteModel;
    private int imageSizeX;
    private int imageSizeY;
    private GpuDelegate gpuDelegate = null;
    private NnApiDelegate nnApiDelegate = null;
    protected Interpreter tflite;
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    private List<String> labels;
    private TensorImage inputImageBuffer;
    private TensorBuffer outputProbabilityBuffer;
    private TensorProcessor probabilityProcessor;
    private TextView texteview1;
    //--------charts---------
    public HorizontalBarChart chart ;
    public ArrayList<BarEntry> BARENTRY ;
    public ArrayList<String> BarEntryLabels ;
    public BarDataSet Bardataset ;
    public BarData BARDATA ;
    private ImageView charthelp_cam;
    private ImageView logo1;
    private  int faceresult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.cameraview);
        //--------------------------------------------
        charthelp_cam = (ImageView) findViewById(R.id.charthelp_cam);
        //--------------------------------------------
        surfaceview = findViewById(R.id.surfaceview);
        SupportAvcCodec();
        if (Build.VERSION.SDK_INT>22) {
            if (!checkPermissionAllGranted(PERMISSIONS_STORAGE)){
                ActivityCompat.requestPermissions(cameraview.this,
                        PERMISSIONS_STORAGE, CAMERA_OK);
            }else{
                init();
            }
        }else{
            init();
        }
        //----------------------------------------------------------------------

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

        //-----------------Charts--------------------
        chart = (HorizontalBarChart) findViewById(R.id.chart1);
        charthelp_cam  = (ImageView) findViewById(R.id.charthelp_cam);
        chart.setDescription("");
        chart.setVisibility(View.INVISIBLE);
        charthelp_cam.setVisibility(View.VISIBLE);
        //----------------------------------------------

    }

    private void init(){
        surfaceHolder = surfaceview.getHolder();
        surfaceHolder.addCallback(this);
    }


    private boolean checkPermissionAllGranted(String[] permissions) {
        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }


    @Override
    public void onRequestPermissionsResult(int requestCode,String[] permissions,int[] grantResults) {
        switch (requestCode) {
            case CAMERA_OK:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    init();
                } else {
                    showWaringDialog();
                }
                break;
            default:
                break;
        }
    }

    private void showWaringDialog() {
        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle("Error")
                .setMessage("the app can not access to phone's camera")
                .setPositiveButton("try latter", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        finish();
                    }
                }).show();
    }



    @Override
    public void surfaceCreated(SurfaceHolder holder) {


    }


    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        camera = getFontCamera();
        startcamera(camera);
        avcCodec = new AvcEncoder(this.width,this.height,framerate,biterate);
        avcCodec.StartEncoderThread();
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        if (null != camera) {
            camera.setPreviewCallback(null);
            camera.stopPreview();
            camera.release();
            camera = null;
            avcCodec.StopThread();
        }
    }


    @Override
    public void onPreviewFrame(byte[] data, android.hardware.Camera camera) {
        // TODO Auto-generated method stub
        putYUVData(data,data.length);
        Camera.Parameters parameters = camera.getParameters();

        int width = parameters.getPreviewSize().width;
        int height = parameters.getPreviewSize().height;

        YuvImage yuv = new YuvImage(data, parameters.getPreviewFormat(), width, height, null);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuv.compressToJpeg(new Rect(0, 0, width, height), 50, out);

        byte[] bytes = out.toByteArray();
        final Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);

        //-------------------------------
        Matrix matrix1 = new Matrix();
        Matrix matrix2 = new Matrix();
        //-------------------------------

        matrix1.setRotate(-90*1);

        final Bitmap bitmap1 = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix1, true);;

        matrix2.postScale(-1,1, bitmap1.getWidth() / 2f, bitmap1.getHeight() / 2f);

        final Bitmap bitmap2 = Bitmap.createBitmap(bitmap1, 0, 0, bitmap1.getWidth(), bitmap1.getHeight(), matrix2, true);

        //-------------------------------
        cameraview.this.runOnUiThread(new Runnable() {

            @Override
            public void run() {

                //-------------------------------
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

                ((ImageView) findViewById(R.id.imageview)).setImageBitmap(bitmap2);
                DetectFace(bitmap2 , (ImageView) findViewById(R.id.imageview));



            }
        });
    }

    public void onBackPressed() {

        this.finish();
        // Go back to previous activity or closes app if last activity

        // Finish all activities in stack and app closes
        //finishAffinity();
        return;


    }

    public void putYUVData(byte[] buffer, int length) {
        if (YUVQueue.size() >= 10) {
            YUVQueue.poll();
        }
        YUVQueue.add(buffer);
    }

    @SuppressLint("NewApi")
    private boolean SupportAvcCodec(){
        if(Build.VERSION.SDK_INT>=18){
            for(int j = MediaCodecList.getCodecCount() - 1; j >= 0; j--){
                MediaCodecInfo codecInfo = MediaCodecList.getCodecInfoAt(j);

                String[] types = codecInfo.getSupportedTypes();
                for (int i = 0; i < types.length; i++) {
                    if (types[i].equalsIgnoreCase("video/avc")) {
                        return true;
                    }
                }
            }
        }
        return false;
    }


    private void startcamera(Camera mCamera){
        if(mCamera != null){
            try {
                mCamera.setPreviewCallback(this);
                mCamera.setDisplayOrientation(90);
                if(parameters == null){
                    parameters = mCamera.getParameters();
                }
                parameters = mCamera.getParameters();
                parameters.setPreviewFormat(ImageFormat.NV21);
                parameters.setPreviewSize(width, height);
                mCamera.setParameters(parameters);
                mCamera.setPreviewDisplay(surfaceHolder);
                mCamera.startPreview();

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @TargetApi(9)
    private Camera getBackCamera() {
        Camera c = null;
        try {
            c = Camera.open(0); // attempt to get a Camera instance
        } catch (Exception e) {
            e.printStackTrace();
        }
        return c; // returns null if camera is unavailable
    }
    private Camera getFontCamera() {
        Camera c = null;
        try {
            c = Camera.open(1); // attempt to get a Camera instance
        } catch (Exception e) {
            e.printStackTrace();
        }
        return c; // returns null if camera is unavailable
    }

    @RequiresApi(api = Build.VERSION_CODES.DONUT)
    void DetectFace(Bitmap bitmap_x, ImageView btn) {


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
//        if(r_c==0){
////                paint.setTextSize(20.0f);
////                boxPaint.setStrokeWidth(1);
//        }
//        else {
////                paint.setTextSize(100.0f);
////                boxPaint.setStrokeWidth(5);
//        }

        //----------
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
//        textView.append(sparseArray.size() + " faces detected \n");
        int size = sparseArray.size();
        if(size==0){
            chart.setVisibility(View.INVISIBLE);
            charthelp_cam.setVisibility(View.VISIBLE);
        }
        else{
            chart.setVisibility(View.VISIBLE);
            charthelp_cam.setVisibility(View.INVISIBLE);

        }
        if(size>1){
            chart.setDescription("Face 1");
        }
        else{
            chart.setDescription("");
        }
        faceresult = 1 ;
        for (int i = 0; i < sparseArray.size(); i++) {
            Face face = sparseArray.valueAt(i);
            float x = face.getPosition().x;
            float y = face.getPosition().y;
            float h = y + face.getHeight();
            float w = x + face.getWidth();
            int j = i + 1;
            RectF rectF = new RectF(x, y, w, h);
            //--------------------
            boxPaint.setStrokeWidth(1);

            paint_FN.setTextSize(30);

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
            if(size>1) {
                ctx.drawText("Face" + j, cx, cy + 23.0f, paint_FN);
            }
            //----------------------------
//                textView.setText("happiness: " + String.format("%.2f", face.getIsSmilingProbability()) + "\n" );
//                textView.append("right eye: " + String.format("%.2f", face.getIsRightEyeOpenProbability())+"\n");
//                textView.append("left eye: " + String.format("%.2f", face.getIsLeftEyeOpenProbability())+"\n");
//            textView.append("Right eye open: " + rightEyeOpenProbability + "   ");
//            textView.append("Euler Y: " + eulerY + "   ");
            float smilingProbability = face.getIsSmilingProbability();
            float leftEyeOpenProbability = face.getIsLeftEyeOpenProbability();
            float rightEyeOpenProbability = face.getIsRightEyeOpenProbability();
            float eulerY = face.getEulerY();
            float eulerZ = face.getEulerZ();

//            textView.append("Face " + j + " : ");
//            textView.append("Smiling: " + smilingProbability + "   ");
//            textView.append("Left eye open: " + leftEyeOpenProbability + "   ");
//            textView.append("Euler Z: " + eulerZ + "\n");

            //All landmarks (draw points)
            for (Landmark landmark : face.getLandmarks()) {
                int cx2 = (int) (landmark.getPosition().x * scale);
                int cy2 = (int) (landmark.getPosition().y * scale);
                ctx.drawCircle(cx2, cy2, 1, paint_L);
            }
            //-------------
            if(faceresult==1) {
                final Bitmap bitmap3 = Bitmap.createBitmap(newBitmap, (int) face.getPosition().x, (int) face.getPosition().y, (int) face.getWidth(), (int) face.getHeight());
                if(bitmap3!=null) {
                    inputImageBuffer = loadImage(bitmap3);
                    chart.setVisibility(View.VISIBLE);
                    charthelp_cam.setVisibility(View.INVISIBLE);
                    tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
                    showresult();
                }
                else{
                    chart.setVisibility(View.INVISIBLE);
                    charthelp_cam.setVisibility(View.VISIBLE);
                }

                faceresult = 0;

            }

            //----------------------------------------------
        }

        btn.setImageDrawable(new BitmapDrawable(getResources(), newBitmap));

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


        Map<String, Float> labeledProbability =
                new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                        .getMapWithFloatValue();
        float maxValueInMap =(Collections.max(labeledProbability.values()));
        int xIndex = 0 ;
        String str ;
        //--------initialise charts---------------
        BARENTRY = new ArrayList<>();
        BarEntryLabels = new ArrayList<String>();
        //----------------------------------------
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
        chart.animateY(0);

    }

}
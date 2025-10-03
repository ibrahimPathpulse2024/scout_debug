package com.surendramaran.yolov8tflite

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.Surface
import android.view.View
import android.view.WindowInsets
import android.view.WindowInsetsController
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import com.surendramaran.yolov8tflite.databinding.ActivityMainBinding
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener {

    private lateinit var binding: ActivityMainBinding

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    private lateinit var detector: Detector
    private lateinit var cameraExecutor: ExecutorService

    // SINGLE reusable bitmap (no per-frame allocations)
    private var reusableBitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Enable immersive fullscreen mode (hide status bar and navigation bar)
        enableImmersiveMode()
        
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
        detector.setup()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        // End Tracking button click handler
        binding.btnEndTracking.setOnClickListener {
            stopTracking()
        }
    }

    private fun enableImmersiveMode() {
        // Hide status bar and navigation bar for true fullscreen experience
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val controller = WindowInsetsControllerCompat(window, window.decorView)
        controller.hide(WindowInsetsCompat.Type.systemBars())
        controller.systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            // Re-enable immersive mode when regaining focus
            enableImmersiveMode()
        }
    }

    private fun stopTracking() {
        // Clean up resources
        detector.clear()
        cameraExecutor.shutdown()
        
        // Close the app
        finishAndRemoveTask()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        // Because we lock to landscape in the Manifest, we can pin rotation to 0
        val rotation = Surface.ROTATION_0

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            // Optional: cap resolution to reduce copy cost (uncomment to enforce)
            // .setTargetResolution(Size(960, 720))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setImageQueueDepth(1)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            try {
                // RGBA_8888 gives us a single plane [0]
                val plane = imageProxy.planes[0]
                val width = imageProxy.width
                val height = imageProxy.height
                val rowStride = plane.rowStride
                val pixelStride = plane.pixelStride // should be 4 for RGBA

                // Create (or recreate if size changes) one reusable Bitmap
                val bmp = reusableBitmap
                if (bmp == null || bmp.width != width || bmp.height != height) {
                    reusableBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                }
                val bitmap = reusableBitmap!!

                val buffer: ByteBuffer = plane.buffer
                buffer.rewind()

                // Fast path: contiguous rows
                if (pixelStride == 4 && rowStride == width * 4) {
                    bitmap.copyPixelsFromBuffer(buffer)
                } else {
                    // Fallback: copy row-by-row if rowStride includes padding
                    val rowBytes = ByteArray(width * 4)
                    var y = 0
                    while (y < height) {
                        val pos = buffer.position()
                        buffer.get(rowBytes, 0, rowBytes.size)
                        bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(rowBytes))
                        buffer.position(pos + rowStride)
                        y++
                    }
                }

                // No per-frame rotate or scale here
                detector.detect(bitmap)
            } catch (t: Throwable) {
                t.printStackTrace()
            } finally {
                imageProxy.close()
            }
        }

        cameraProvider.unbindAll()
        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )
            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) {
        if (it[Manifest.permission.CAMERA] == true) { startCamera() }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.clear()
        cameraExecutor.shutdown()
    }

    override fun onEmptyDetect() {
        binding.overlay.invalidate()
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.apply {
                setResults(boundingBoxes)
                invalidate()
            }
        }
    }

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf(
            Manifest.permission.CAMERA
        ).toTypedArray()
    }
}

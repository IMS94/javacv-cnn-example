package org.imesha.examples.javacv;

import org.bytedeco.javacpp.opencv_core.CvRect;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

import static org.bytedeco.javacpp.opencv_core.Point;
import static org.bytedeco.javacpp.opencv_core.Scalar;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * An exmaple to demonstrate JavaCV's frame grabbing and other features
 *
 * @author Imesha Sudasingha
 */
public class JavaCVExample {

    private static final Logger logger = LoggerFactory.getLogger(JavaCVExample.class);

    private FFmpegFrameGrabber frameGrabber;
    private volatile boolean running = false;

    private HaarFaceDetector faceDetector = new HaarFaceDetector();
    private CNNAgeDetector ageDetector = new CNNAgeDetector();
    private CNNGenderDetector genderDetector = new CNNGenderDetector();

    private OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();

    public void start() {
        frameGrabber = new FFmpegFrameGrabber("/dev/video0");
        frameGrabber.setFormat("video4linux2");
        frameGrabber.setImageWidth(1280);
        frameGrabber.setImageHeight(720);

        logger.debug("Starting frame grabber");
        try {
            frameGrabber.start();
            logger.debug("Started frame grabber with image width-height : {}-{}", frameGrabber.getImageWidth(), frameGrabber.getImageHeight());
        } catch (FrameGrabber.Exception e) {
            logger.error("Error when initializing the frame grabber");
            throw new RuntimeException("Unable to start the FrameGrabber");
        }

        running = true;
        while (running) {
            try {
                // Here we grab frames from our camera
                final Frame frame = frameGrabber.grab();

                Map<CvRect, Mat> detectedFaces = faceDetector.detect(frame);
                Mat mat = toMatConverter.convert(frame);

                detectedFaces.entrySet().forEach(rectMatEntry -> {
                    String age = ageDetector.predictAge(rectMatEntry.getValue(), frame);
                    CNNGenderDetector.Gender gender = genderDetector.predictGender(rectMatEntry.getValue(), frame);

                    String caption = String.format("%s:[%s]", gender, age);
                    logger.debug("Face's caption : {}", caption);

                    rectangle(mat, new Point(rectMatEntry.getKey().x(), rectMatEntry.getKey().y()),
                            new Point(rectMatEntry.getKey().width() + rectMatEntry.getKey().x(), rectMatEntry.getKey().height() + rectMatEntry.getKey().y()),
                            Scalar.RED, 2, CV_AA, 0);

                    int posX = Math.max(rectMatEntry.getKey().x() - 10, 0);
                    int posY = Math.max(rectMatEntry.getKey().y() - 10, 0);
                    putText(mat, caption, new Point(posX, posY), CV_FONT_HERSHEY_PLAIN, 1.0,
                            new Scalar(255, 255, 255, 2.0));
                });

                // Show the processed mat in UI
                toMatConverter.convert(mat);
            } catch (FrameGrabber.Exception e) {
                logger.error("Error when grabbing the frame", e);
            }
        }

        logger.debug("Stopped frame grabbing since the state is 'not running'");
    }

    public void stop() {
        running = false;
        try {
            logger.debug("Releasing and stopping FrameGrabber");
            frameGrabber.release();
            frameGrabber.stop();
        } catch (FrameGrabber.Exception e) {
            logger.error("Error occurred when stopping the FrameGrabber", e);
        }
    }

    public static void main(String[] args) {
        JavaCVExample javaCVExample = new JavaCVExample();

        logger.info("Starting javacv example");
        new Thread(javaCVExample::start).start();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info("Stopping javacv example");
            javaCVExample.stop();
        }));

        try {
            Thread.currentThread().join();
        } catch (InterruptedException ignored) { }
    }
}

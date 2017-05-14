package org.imesha.examples.javacv;

import org.bytedeco.javacpp.opencv_core.CvRect;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.imesha.examples.javacv.util.ImageUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.util.Map;

import static org.bytedeco.javacpp.opencv_core.Point;
import static org.bytedeco.javacpp.opencv_core.Scalar;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * An example to demonstrate JavaCV's frame grabbing and other features
 *
 * @author Imesha Sudasingha
 */
public class JavaCVExample {

    private static final Logger logger = LoggerFactory.getLogger(JavaCVExample.class);

    private FFmpegFrameGrabber frameGrabber;
    private OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();
    private volatile boolean running = false;

    private HaarFaceDetector faceDetector = new HaarFaceDetector();
    private CNNAgeDetector ageDetector = new CNNAgeDetector();
    private CNNGenderDetector genderDetector = new CNNGenderDetector();

    private JFrame window;
    private JPanel videoPanel;

    public JavaCVExample() {
        window = new JFrame();
        videoPanel = new JPanel();

        window.setLayout(new BorderLayout());
        window.setSize(new Dimension(1280, 720));
        window.add(videoPanel, BorderLayout.CENTER);
        window.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                stop();
            }
        });
    }

    /**
     * Starts the frame grabbers and then the frame processing. Grabbed and processed frames will be displayed in the
     * {@link #videoPanel}
     */
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
            logger.error("Error when initializing the frame grabber", e);
            throw new RuntimeException("Unable to start the FrameGrabber", e);
        }

        SwingUtilities.invokeLater(() -> {
            window.setVisible(true);
        });

        process();

        logger.debug("Stopped frame grabbing.");
    }

    /**
     * Private method which will be called to star frame grabbing and carry on processing the grabbed frames
     */
    private void process() {
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
                Frame processedFrame = toMatConverter.convert(mat);

                Graphics graphics = videoPanel.getGraphics();
                BufferedImage resizedImage = ImageUtils.getResizedBufferedImage(processedFrame, videoPanel);
                SwingUtilities.invokeLater(() -> {
                    graphics.drawImage(resizedImage, 0, 0, videoPanel);
                });
            } catch (FrameGrabber.Exception e) {
                logger.error("Error when grabbing the frame", e);
            } catch (Exception e) {
                logger.error("Unexpected error occurred while grabbing and processing a frame", e);
            }
        }
    }

    /**
     * Stops and released resources attached to frame grabbing. Stops frame processing and,
     */
    public void stop() {
        running = false;
        try {
            logger.debug("Releasing and stopping FrameGrabber");
            frameGrabber.release();
            frameGrabber.stop();
        } catch (FrameGrabber.Exception e) {
            logger.error("Error occurred when stopping the FrameGrabber", e);
        }

        window.dispose();
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

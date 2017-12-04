package org.imesha.examples.javacv;

import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING;

/**
 * Face detector using haar classifier cascades
 *
 * @author Imesha Sudasingha
 */
public class HaarFaceDetector {

    private static final Logger logger = LoggerFactory.getLogger(HaarFaceDetector.class);

    private opencv_objdetect.CvHaarClassifierCascade haarClassifierCascade;
    private CvMemStorage storage;
    private OpenCVFrameConverter.ToIplImage iplImageConverter;
    private OpenCVFrameConverter.ToMat toMatConverter;

    public HaarFaceDetector() {
        iplImageConverter = new OpenCVFrameConverter.ToIplImage();
        toMatConverter = new OpenCVFrameConverter.ToMat();

        try {
            File haarCascade = new File(this.getClass().getResource("/detection/haarcascade_frontalface_alt.xml").toURI());
            logger.debug("Using Haar Cascade file located at : {}", haarCascade.getAbsolutePath());
            haarClassifierCascade = new opencv_objdetect.CvHaarClassifierCascade(cvLoad(haarCascade.getAbsolutePath()));
        } catch (Exception e) {
            logger.error("Error when trying to get the haar cascade", e);
            throw new IllegalStateException("Error when trying to get the haar cascade", e);
        }
        storage = CvMemStorage.create();
    }

    /**
     * Detects and returns a map of cropped faces from a given captured frame
     *
     * @param frame the frame captured by the {@link org.bytedeco.javacv.FrameGrabber}
     * @return A map of faces along with their coordinates in the frame
     */
    public Map<CvRect, Mat> detect(Frame frame) {
        Map<CvRect, Mat> detectedFaces = new HashMap<>();

        IplImage iplImage = iplImageConverter.convert(frame);

        /*
         * return a CV Sequence (kind of a list) with coordinates of rectangle face area.
         * (returns coordinates of left top corner & right bottom corner)
         */
        CvSeq detectObjects = cvHaarDetectObjects(iplImage, haarClassifierCascade, storage, 1.5, 3, CV_HAAR_DO_CANNY_PRUNING);
        Mat matImage = toMatConverter.convert(frame);

        int numberOfPeople = detectObjects.total();
        for (int i = 0; i < numberOfPeople; i++) {
            CvRect rect = new CvRect(cvGetSeqElem(detectObjects, i));
            Mat croppedMat = matImage.apply(new Rect(rect.x(), rect.y(), rect.width(), rect.height()));
            detectedFaces.put(rect, croppedMat);
        }

        return detectedFaces;
    }

    @Override
    public void finalize() {
        cvReleaseMemStorage(storage);
    }
}

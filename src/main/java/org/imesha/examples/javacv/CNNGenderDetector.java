package org.imesha.examples.javacv;

import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacv.Frame;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_dnn.*;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 * The class responsible for recognizing gender. This class use the concept of CNN (Convolution Neural Networks) to
 * identify the gender of a detected face.
 *
 * @author Imesha Sudasingha
 */
public class CNNGenderDetector {

    private static final Logger logger = LoggerFactory.getLogger(CNNGenderDetector.class);

    private Net genderNet;

    public CNNGenderDetector() {
        try {
            genderNet = new Net();
            File protobuf = new File(getClass().getResource("/caffe/deploy_gendernet.prototxt").toURI());
            File caffeModel = new File(getClass().getResource("/caffe/gender_net.caffemodel").toURI());
            Importer importer = createCaffeImporter(protobuf.getAbsolutePath(), caffeModel.getAbsolutePath());
            importer.populateNet(genderNet);
            importer.close();
        } catch (Exception e) {
            logger.error("Error reading prototxt", e);
            throw new IllegalStateException("Unable to start CNNGenderDetector", e);
        }
    }

    /**
     * Predicts gender of a given cropped face
     *
     * @param face  the cropped face as a {@link Mat}
     * @param frame the original frame where the face was cropped from
     * @return Gender
     */
    public Gender predictGender(Mat face, Frame frame) {
        try {
            Mat croppedMat = new Mat();
            resize(face, croppedMat, new Size(256, 256));
            normalize(croppedMat, croppedMat, 0, Math.pow(2, frame.imageDepth), NORM_MINMAX, -1, null);

            Blob inputBlob = new Blob(croppedMat);
            genderNet.setBlob(".data", inputBlob);
            genderNet.forward();
            Blob prob = genderNet.getBlob("prob");

            Indexer indexer = prob.matRefConst().createIndexer();
            logger.debug("CNN results {},{}", indexer.getDouble(0, 0), indexer.getDouble(0, 1));
            if (indexer.getDouble(0, 0) > indexer.getDouble(0, 1)) {
                logger.debug("Male detected");
                return Gender.MALE;
            } else {
                logger.debug("Female detected");
                return Gender.FEMALE;
            }
        } catch (Exception e) {
            logger.error("Error when processing gender", e);
        }
        return Gender.NOT_RECOGNIZED;
    }

    public enum Gender {
        MALE,
        FEMALE,
        NOT_RECOGNIZED
    }
}

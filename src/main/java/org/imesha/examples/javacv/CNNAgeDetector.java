package org.imesha.examples.javacv;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacv.Frame;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URISyntaxException;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_dnn.*;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 * Age predictor using Convolution Neural Networks
 *
 * @author Imesha Sudasingha
 */
public class CNNAgeDetector {

    private static final Logger logger = LoggerFactory.getLogger(CNNAgeDetector.class);

    private static final String[] AGES = new String[]{"0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60-"};

    private Net ageNet;

    public CNNAgeDetector() {
        try {
            ageNet = new Net();
            File protobuf = new File(getClass().getResource("/caffe/deploy_agenet.prototxt").toURI());
            File caffeModel = new File(getClass().getResource("/caffe/age_net.caffemodel").toURI());

            Importer importer = createCaffeImporter(protobuf.getAbsolutePath(), caffeModel.getAbsolutePath());
            importer.populateNet(ageNet);
            importer.close();
        } catch (URISyntaxException e) {
            logger.error("Unable to load the caffe model", e);
            throw new IllegalStateException("Unable to load the caffe model", e);
        }
    }

    /**
     * Predicts the age of a {@link Mat} supplied to this method. The {@link Mat} is supposed to be the cropped face
     * of a human whose age is to be predicted.
     *
     * @param face  cropped face
     * @param frame whole frame where the target human is also present
     * @return Predicted age range
     */
    public String predictAge(Mat face, Frame frame) {
        try {
            Mat resizedMat = new Mat();
            resize(face, resizedMat, new Size(256, 256));
            normalize(resizedMat, resizedMat, 0, Math.pow(2, frame.imageDepth), NORM_MINMAX, -1, null);

            Blob inputBlob = new Blob(resizedMat);
            ageNet.setBlob(".data", inputBlob);
            ageNet.forward();
            Blob prob = ageNet.getBlob("prob");

            DoublePointer pointer = new DoublePointer(new double[1]);
            Point max = new Point();
            minMaxLoc(prob.matRefConst(), null, pointer, null, max, null);
            return AGES[max.x()];
        } catch (Exception e) {
            logger.error("Error when processing gender", e);
        }
        return null;
    }
}

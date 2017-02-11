/*
 * The MIT License (MIT)
 * Copyright (c) 2016 Imesha Sudasingha
 * <p>
 * Permission is hereby granted, free of charge,
 * to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * <p>
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 * <p>
 * THE SOFTWARE IS PROVIDED "AS IS",
 * WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package org.imesha.examples.javacv.util;

import net.coobird.thumbnailator.Thumbnails;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

/**
 * Utils to be used for images related tasks
 *
 * @author erandi
 */
public class ImageUtils {

    private static final Logger logger = LoggerFactory.getLogger(ImageUtils.class);
    private static Java2DFrameConverter frameConverter = new Java2DFrameConverter();
    private static OpenCVFrameConverter.ToMat matConverter = new OpenCVFrameConverter.ToMat();


    /**
     * Method to get resized buffered image when user passes the relevant frame and video panel.
     *
     * @param frame      frame to be converted to {@link BufferedImage}
     * @param videoPanel the {@link JPanel} which is to be used to obtain panel size
     * @return resized {@link BufferedImage}
     */
    public static BufferedImage getResizedBufferedImage(Frame frame, JPanel videoPanel) {
        BufferedImage resizedImage = null;

        try {
            /*
             * We get notified about the frames that are being added. Then we pass each frame to BufferedImage. I have used
             * a library called Thumbnailator to achieve the resizing effect along with performance
             */
            resizedImage = Thumbnails.of(frameConverter.getBufferedImage(frame))
                    .size(videoPanel.getWidth(), videoPanel.getHeight())
                    .asBufferedImage();
        } catch (IOException e) {
            logger.error("Unable to convert the image to a buffered image", e);
        }

        return resizedImage;
    }
}

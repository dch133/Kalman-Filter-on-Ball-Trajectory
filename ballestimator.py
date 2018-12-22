import numpy as np
import cv2
import sys
import time
bgr_color = 29,2,128 #129,119,195
color_threshold = 50 #color range

hsv_color = cv2.cvtColor( np.uint8([[bgr_color]] ), cv2.COLOR_BGR2HSV)[0][0]
HSV_lower = np.array([hsv_color[0] - color_threshold, hsv_color[1] - color_threshold, hsv_color[2] - color_threshold])
HSV_upper = np.array([hsv_color[0] + color_threshold, hsv_color[1] + color_threshold, hsv_color[2] + color_threshold])


def detect_ball(frame):
    x, y, radius = -1, -1, -1

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_frame, HSV_lower, HSV_upper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = (-1, -1)

    # only proceed if at least one contour was found
    if len(contours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(mask)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) #position of the ball

        # check that the radius is larger than some threshold
        if radius > 10: #CHANGED
            #outline ball
            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            #show ball center
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

    return center[0], center[1], radius


if __name__ == "__main__":
    filepath = sys.argv[1]

    cap = cv2.VideoCapture(filepath)

    dist = []
    times = []
    count = 0
    kalmanFilter = []
    variance = []
    initialEstimatedVariance = 1
    deltaX = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    deltat = 524  # num frames(varies per video test case)
    noise = 1  # noise Q
    sensorCovariance = 83791.65209996712



    variance.append(initialEstimatedVariance)
    #start = time.clock()
    while(cap.isOpened()):
        count+=1
        # Capture frame-by-frame
        ret, frame = cap.read()

        #detect_ball(frame)
        [x,y,r] = detect_ball(frame)
        #t = time.clock() - start
        if (len(dist) is not 0):
            x_prev = dist[len(dist) - 1]

            velocity = float(deltaX)/deltat

            dist.append(x)

            estimatedX = x_prev + velocity
            estimatedVariance = variance[len(variance)-1] + noise

            kalmanGain = float(estimatedVariance)/(estimatedVariance + sensorCovariance)

            kalmanPosition = estimatedX + (kalmanGain*(x - estimatedX))

            kalmanFilter.append(kalmanPosition)

            updatedVariance = estimatedVariance-(kalmanGain*estimatedVariance)
            variance.append(updatedVariance)

            print (kalmanFilter)
            #print(variance)

        else: #don't do any calculations on the 1st point
            x_initial = detect_ball(frame)[0]
            dist.append(x_initial)
            kalmanFilter.append(x_initial)

        # Display the resulting frame
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
        imS = cv2.resizeWindow('frame', (960, 540))  # Resize image

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



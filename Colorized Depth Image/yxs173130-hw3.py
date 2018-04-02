import numpy as np
import cv2
from numpy.linalg import inv

def loadMatrix():

    file = open('InvIntrinsicIR', 'r')
    InvIntrinsicIR = []
    for line in file:
        InvIntrinsicIR.append(line.strip().split(','))
    InvIntrinsicIR = np.array(InvIntrinsicIR).astype(np.float)

    file = open('IntrinsicRGB', 'r')
    IntrinsicRGB = []
    for line in file:
        IntrinsicRGB.append(line.strip().split(','))
    IntrinsicRGB = np.array(IntrinsicRGB).astype(np.float)

    file = open('TransformationD-C', 'r')
    Transformation = []
    for line in file:
        Transformation.append(line.strip().split(','))
    Transformation = np.array(Transformation).astype(np.float)
    file.close()

    return InvIntrinsicIR, IntrinsicRGB, Transformation

def colorizedImage(rgbImg, depthImg, InvIntrinsicIR, IntrinsicRGB, Transformation):
    depthRow, depthColumn = depthImg.shape
    rgbRow, rgbColumn, rgbChannel = rgbImg.shape
    newImg = np.zeros((depthRow,depthColumn,3));

    for i in range(0, depthRow):
        for j in range(0, depthColumn):
            depth = depthImg[i,j]
            #IR matrix * Depth => Real word coordinate
            depth3D = np.matmul([j,i,1], InvIntrinsicIR)
            if depth > 0:
                realImg = np.multiply(depth3D, depth)
                #Convert 3D to 4D in order to transform to RGB from depth
                realImg = np.append(realImg, 1)
                rgbCoordi = np.matmul(realImg,Transformation)

                #Get the w', convert 4D to 3D
                rgbDepth = rgbCoordi[2]
                rgbCoordi = rgbCoordi/rgbDepth
                rgbCoordi = rgbCoordi[0:3]

                #Convert 3D to 2D by multiplying intrinsicRGB
                inter1 = np.matmul(rgbCoordi, IntrinsicRGB)
                if(np.rint(inter1[1]) > 0) and (np.rint(inter1[1]) < rgbRow) and (np.rint(inter1[0]) > 0) and (np.rint(inter1[0]) < rgbColumn):
                    newImg[i,j,0] = rgbImg[int(np.rint(inter1[1])),int(np.rint(inter1[0])),0]
                    newImg[i,j,1] = rgbImg[int(np.rint(inter1[1])),int(np.rint(inter1[0])),1]
                    newImg[i,j,2] = rgbImg[int(np.rint(inter1[1])),int(np.rint(inter1[0])),2]
    return newImg

def color_detect(image, a, b):
    lower = np.array([0,60,40])
    upper = np.array([16,255,255])


    img = cv2.imread(image)
    image = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.erode(mask, None, iterations = 2)
    mask = cv2.dilate(mask, None, iterations = 1)

    cv2.imwrite(b, mask)
    cv2.imshow(b, mask)

    cnt = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    c1 = max(cnt, key = cv2.contourArea)
    cnt.remove(c1)
    c2 = max(cnt, key = cv2.contourArea)

    ((x1, y1), radius1) = cv2.minEnclosingCircle(c1)
    ((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
    #
    # moment = cv2.moments(c1)
    # center = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))
    # (xm1, ym1) = center
    #
    # moment2 = cv2.moments(c2)
    # center2 = (int(moment2["m10"] / moment2["m00"]), int(moment2["m01"] / moment2["m00"]))
    # (xm2, ym2) = center2
    cv2.rectangle(img, (int(x1 - radius1), int(y1 - radius1)),(int(x1 + radius1), int(y1 + radius1)),(0, 255, 0), 2)
    cv2.rectangle(img, (int(x2 - radius2), int(y2 - radius2)),(int(x2 + radius2), int(y2 + radius2)),(0, 255, 0), 2)

    cv2.imwrite(a, img)
    cv2.imshow(a, img)

    if (cv2.waitKey(0) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
    return (x1, y1), (x2, y2)

def compute_velocity(invIntrinsicIR, a1, b1, a2, b2):
    # file = open('InvIntrinsicIR', 'r')
    # invIntrinsicIR = []
    # for line in file:
    #     invIntrinsicIR.append(line.strip().split(','))
    # invIntrinsicIR = np.array(invIntrinsicIR).astype(np.float)
    # print(invIntrinsicIR)
    depth1 = cv2.imread('depth-63647317626781.png', cv2.IMREAD_ANYDEPTH)
    depth2 = cv2.imread('depth-63647317628081.png', cv2.IMREAD_ANYDEPTH)
    (x1, y1) = (int(b1), int(a1))
    (x2, y2) = (int(b2), int(a2))
    matrix1 = np.matmul(np.array([x1, y1, 1]), invIntrinsicIR)
    matrix1 = np.multiply(matrix1, depth1[x1][y1])
    matrix2 = np.matmul(np.array([x2, y2, 1]), invIntrinsicIR)
    matrix2 = np.multiply(matrix2, depth1[x1][y1])
    distance = np.sqrt(np.square(matrix1[1] - matrix2[1]) + np.square(matrix1[2] - matrix2[2]))

    velocity = 1.0 * distance / 1.3
    return velocity

if __name__ == "__main__":
    rgbImg_initial = cv2.imread('color-63647317626781.png')
    depthImg_initial = cv2.imread('depth-63647317626781.png',cv2.IMREAD_ANYDEPTH)
    rgbImg_final = cv2.imread('color-63647317628081.png')
    depthImg_final = cv2.imread('depth-63647317628081.png',cv2.IMREAD_ANYDEPTH)

    InvIntrinsicIR, IntrinsicRGB, Transformation = loadMatrix()
    newImg_initial = colorizedImage(rgbImg_initial, depthImg_initial, InvIntrinsicIR, IntrinsicRGB, Transformation)
    newImg_final = colorizedImage(rgbImg_final, depthImg_final, InvIntrinsicIR, IntrinsicRGB, Transformation)

    cv2.imwrite("colorize_initial.png", newImg_initial)
    cv2.imwrite("colorize_final.png", newImg_final)

    (initial_x1, initial_y1), (initial_x2, initial_y2) = color_detect("colorize_initial.png", "initial_detect_1.jpg", "tmp_1.jpg")
    (final_x1, final_y1), (final_x2, final_y2) = color_detect("colorize_final.png", "final_detect_1.jpg", "tmp_2.jpg")

    velocity1 = compute_velocity(InvIntrinsicIR, initial_x1, initial_y1, final_x1, final_y1)
    velocity2 = compute_velocity(InvIntrinsicIR, initial_x2, initial_y2, final_x2, final_y2)
    print('relative velocity =', velocity1 + velocity2, "mm/s")

import cv2 as cv
import numpy as np
import argparse

def main() -> None:
    
    parser = argparse.ArgumentParser(description='image')
    parser.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
    args = parser.parse_args()
    
    image = cv.imread(args.image) 
    
    if image is None:
        raise ValueError("Could not read the image.")
    
    # Resize the image
    ratio = 1000/image.shape[0]
    orig = image.copy()
    resize_image = cv.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation = cv.INTER_CUBIC)
    
    gray = cv.cvtColor(resize_image, cv.COLOR_RGB2GRAY)
    gray = cv.GaussianBlur(gray,(5,5),0)
    edged = cv.Canny(gray, 75, 200)
    
    #Find contours
    contours,_ = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key = cv.contourArea, reverse = True)[:5]
    
    screen_cnt = None
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            screen_cnt = approx
            break

    cv.drawContours(resize_image, screen_cnt, -1, (0, 255, 0), 5)
    cv.imshow("Original Image", resize_image)
    #cv.imshow("Gray Image", gray)
    #cv.imshow("edged Image", edged)
    cv.waitKey(0)

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

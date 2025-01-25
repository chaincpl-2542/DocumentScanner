import cv2 as cv
import numpy as np
import pytesseract
import argparse

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  
    rect[2] = pts[np.argmax(s)]  
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] 
    rect[3] = pts[np.argmax(diff)] 
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts.reshape(4, 2))
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def update_canny(val=None):
    threshold1 = cv.getTrackbarPos('Threshold1', 'Canny Edges')
    threshold2 = cv.getTrackbarPos('Threshold2', 'Canny Edges')
    edged = cv.Canny(gray, threshold1, threshold2)
    cv.imshow("Canny Edges", edged)

def main() -> None:
    parser = argparse.ArgumentParser(description="OCR")
    parser.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
    args = parser.parse_args()

    image = cv.imread(args.image)
    
    ratio = 1000 / image.shape[0]
    orig = image.copy()
    resize_image = cv.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv.INTER_CUBIC)

    global gray
    gray = cv.cvtColor(resize_image, cv.COLOR_RGB2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    cv.namedWindow("Canny Edges")
    cv.createTrackbar("Threshold1", "Canny Edges", 50, 255, update_canny)
    cv.createTrackbar("Threshold2", "Canny Edges", 150, 255, update_canny)

    # Initial call to show Canny edges
    update_canny()
    
    cv.waitKey(0)

    edged = cv.Canny(gray, cv.getTrackbarPos('Threshold1', 'Canny Edges'), cv.getTrackbarPos('Threshold2', 'Canny Edges'))
    contours, _ = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]

    screen_cnt = None
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is not None:
        contour_image = resize_image.copy()
        cv.drawContours(contour_image, [screen_cnt], -1, (0, 255, 0), 2)
        cv.imshow("Contours", contour_image)
        key = cv.waitKey(0)

        if key == 27:
            cv.destroyAllWindows()
            return

    warped = four_point_transform(orig, screen_cnt * (1 / ratio))

    warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    _, binary_document = cv.threshold(warped_gray, 127, 255, cv.THRESH_BINARY)

    print("OCR...")
    text = pytesseract.image_to_string(binary_document, lang="eng")

    print(text)

    cv.imshow("Original", resize_image)
    cv.imshow("Scan", binary_document)
    cv.imwrite("document.png", binary_document)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
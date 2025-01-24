import cv2 as cv
import numpy as np
import pytesseract
import argparse


def order_points(pts):
    """Order points in clockwise order (top-left, top-right, bottom-right, bottom-left)."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  
    rect[2] = pts[np.argmax(s)]  
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] 
    rect[3] = pts[np.argmax(diff)] 
    return rect


def four_point_transform(image, pts):
    """Apply perspective transform to obtain a top-down view of an image."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Document Scanner with OCR")
    parser.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
    args = parser.parse_args()

    image = cv.imread(args.image)
    if image is None:
        raise ValueError("Could not read the image.")

    ratio = 1000 / image.shape[0]
    orig = image.copy()
    resize_image = cv.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv.INTER_CUBIC)

    gray = cv.cvtColor(resize_image, cv.COLOR_RGB2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(gray, 75, 200)

    contours, _ = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]

    screen_cnt = None
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is None:
        print("No document detected!")
        return

    warped = four_point_transform(orig, screen_cnt * (1 / ratio))

    warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    _, binary_document = cv.threshold(warped_gray, 127, 255, cv.THRESH_BINARY)

    print("Performing OCR...")
    text = pytesseract.image_to_string(binary_document, lang="eng")
    print("Detected Text:")
    print(text)

    cv.imshow("Original Image", resize_image)
    cv.imshow("Scanned Document", binary_document)
    cv.imwrite("scanned_document.jpg", binary_document)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

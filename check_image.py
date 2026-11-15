
import cv2

image = cv2.imread("captured_images/inspection_20261115_171003.jpg")

if image is None:
    print("Image not found!")
else:
    print("Image dimensions:", image.shape)

    cv2.imshow("Captured Label", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    enlarged = cv2.resize(
        gray, None,
        fx=2, fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    cv2.imshow("Enlarged Grayscale Image", enlarged)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

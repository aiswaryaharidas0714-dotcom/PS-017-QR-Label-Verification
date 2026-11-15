
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image_path = "captured_images/inspection_20261115_184609.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

print("Image size:", image.shape)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

gray = cv2.GaussianBlur(gray, (3, 3), 0)

processed = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31, 11
)

cv2.imwrite("ocr_processed.jpg", processed)

for mode in [6, 11, 12]:
    print("\nOCR MODE:", mode)

    config = f"--oem 3 --psm {mode}"

    text = pytesseract.image_to_string(
        processed,
        config=config
    )

    print(text)

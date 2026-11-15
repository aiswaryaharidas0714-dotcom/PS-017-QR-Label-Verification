
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image_path = "C:\\Users\\ELCOT\\PS-017-QR-Label-Verification\\captured_images\\sample_product_label.png"
image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

print("Original image:", image.shape)

# Select the text area manually
print("Select the area containing the product text.")
roi = cv2.selectROI("Select Text Area", image, False)
cv2.destroyWindow("Select Text Area")

x, y, w, h = roi
cropped = image[y:y+h, x:x+w]
print("Selected region:")
print("X:", x)
print("Y:", y)
print("Width:", w)
print("Height:", h)

cv2.imwrite("selected_text_region.jpg", cropped)

if w == 0 or h == 0:
    print("No area selected!")
    exit()

cropped = image[y:y+h, x:x+w]

# Enlarge the selected area
enlarged = cv2.resize(cropped, None, fx=3, fy=3)

# Convert to grayscale
gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

# Improve contrast
processed = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)[1]

print("\n========== SELECTED AREA OCR ==========")

for mode in [6, 11, 12]:
    print("\nPSM MODE:", mode)
    text = pytesseract.image_to_string(
        processed, config=f"--oem 3 --psm {mode}"
    )
    print(text if text.strip() else "NO TEXT DETECTED")

cv2.imshow("Selected Text Area", cropped)
cv2.imshow("Processed Text Area", processed)

cv2.waitKey(0)
cv2.destroyAllWindows()

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print("Tesseract OCR Connected Successfully!")
print("OCR Version:", pytesseract.get_tesseract_version())
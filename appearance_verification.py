import cv2

def verify_appearance(reference_path, captured_path):

    reference = cv2.imread(reference_path)
    captured = cv2.imread(captured_path)

    if reference is None or captured is None:
        print("Image not found!")
        return

    if reference.shape != captured.shape:
        print("Image dimensions do not match!")
        return

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    captured_gray = cv2.cvtColor(captured, cv2.COLOR_BGR2GRAY)

    difference = cv2.absdiff(reference_gray, captured_gray)

    _, threshold = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    changed_pixels = cv2.countNonZero(threshold)
    total_pixels = threshold.shape[0] * threshold.shape[1]

    percentage = (changed_pixels / total_pixels) * 100

    print("Difference Percentage:", round(percentage, 2), "%")

    if percentage > 0.5:
        print("Appearance Result: FAIL")
    else:
        print("Appearance Result: PASS")


print("ORIGINAL LABEL TEST")
verify_appearance(
    "sample_product_label.png",
    "sample_product_label.png"
)

print("\nALTERED LABEL TEST")
verify_appearance(
    "sample_product_label.png",
    "altered_product_label.png"
)
import cv2
import imutils
import pytesseract
import os

# -------------------------------------------------
# Set the Tesseract OCR path
# -------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------------------------------
# Get current project directory
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Image path
image_path = os.path.join(BASE_DIR, "Car Images", "5.jpg")

# -------------------------------------------------
# Read Image
# -------------------------------------------------
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    print("Checked path:", image_path)
    exit()

# Resize image
image = imutils.resize(image, width=600)

# Display Original
cv2.imshow("Original Image", image)

# -------------------------------------------------
# Convert to Gray
# -------------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise removal
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Edge detection
edged = cv2.Canny(gray, 30, 200)

cv2.imshow("Edges", edged)

# -------------------------------------------------
# Find contours
# -------------------------------------------------
contours, _ = cv2.findContours(
    edged.copy(),
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

numberPlateContour = None

# Search for rectangle
for contour in contours:

    perimeter = cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)

    if len(approx) == 4:
        numberPlateContour = approx
        break

# -------------------------------------------------
# Check if number plate found
# -------------------------------------------------
if numberPlateContour is None:
    print("Number Plate Not Detected.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# Draw contour
cv2.drawContours(image, [numberPlateContour], -1, (0, 255, 0), 3)

cv2.imshow("Detected Plate", image)

# -------------------------------------------------
# Crop the Number Plate
# -------------------------------------------------
mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mask[:] = 0

cv2.drawContours(mask, [numberPlateContour], -1, 255, -1)

result = cv2.bitwise_and(image, image, mask=mask)

x, y, w, h = cv2.boundingRect(numberPlateContour)

cropped = result[y:y+h, x:x+w]

if cropped.size == 0:
    print("Failed to crop number plate.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

cv2.imshow("Cropped Plate", cropped)

# -------------------------------------------------
# Save cropped image
# -------------------------------------------------
output_folder = os.path.join(BASE_DIR, "Cropped Images-Text")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

crop_path = os.path.join(output_folder, "plate.png")

cv2.imwrite(crop_path, cropped)

# -------------------------------------------------
# OCR
# -------------------------------------------------
text = pytesseract.image_to_string(cropped, config='--psm 7')

print("\nDetected Number Plate:")
print("------------------------")
print(text.strip())

# -------------------------------------------------
# Finish
# -------------------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()
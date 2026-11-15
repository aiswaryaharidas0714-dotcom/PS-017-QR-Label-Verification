from PIL import Image, ImageDraw, ImageFont

# Create label background
label = Image.new("RGB", (1000, 650), "white")
draw = ImageDraw.Draw(label)

# Load fonts
font = ImageFont.truetype("arial.ttf", 32)
title_font = ImageFont.truetype("arial.ttf", 45)

# Product heading
draw.text((50, 30), "HERBAL SHAMPOO", fill="darkgreen", font=title_font)
draw.text((50, 120), "Product ID: PROD001", fill="black", font=font)
draw.text((50, 190), "Batch Number: BATCH101", fill="black", font=font)
draw.text((50, 260), "Manufacturing Date: 01-11-2026", fill="black", font=font)
draw.text((50, 330), "Expiry Date: 01-11-2028", fill="black", font=font)

# Load real QR code
qr = Image.open("product_qr.png").convert("RGB")
qr = qr.resize((300, 300))

# Add QR code to label
label.paste(qr, (600, 150))

draw.text((650, 470), "SCAN PRODUCT QR", fill="black", font=font)

# Save label
label.save("sample_product_label.png")

print("Product label created successfully!")
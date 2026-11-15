import qrcode

product_data = "PROD001"

qr = qrcode.make(product_data)

qr.save("product_qr.png")

print("QR Code generated successfully!")
print("Saved as product_qr.png")
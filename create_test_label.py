from PIL import Image, ImageDraw

image = Image.open("captured_images/sample_product_label.png")
draw = ImageDraw.Draw(image)

draw.rectangle((0, 0, 550, 100), fill="white")
draw.text((50, 30), "FAKE PRODUCT LABEL", fill="red")

image.save("captured_images/fake_product_label.png")
print("Altered test label created successfully!")
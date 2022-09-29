from PIL import Image
import math


f = open("two1.txt", 'w+')  
im = Image.open('E:\code\scenarioagentcnn\PicClass.jpg')

width = im.size[0]
height = im.size[1]
print(width,height)
rgb_im = im.convert('RGB')
for i in range(width):
	for j in range(height):
		r, g, b = rgb_im.getpixel((i, j))
		print(r,",", g,",",b,file=f)

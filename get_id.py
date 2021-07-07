import glob

path = './myLIP_6000/train/image/*' #path_to_image
image_paths = glob.glob(path)
print(len(image_paths))
for image_path in image_paths:
    f = open("id.txt", "a")
    s = image_path[24:-4]+'\n'
    f.write(s)
f.close()
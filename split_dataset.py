import os
import glob
from PIL import Image

def save_split(image_class, split, images):
    for i, image in enumerate(images):
        try:
            im  = Image.open(image)
            im_name = image_class + '-' + str(i) + '.jpg'
            path = "./Split_Dataset/" + split + "/" + image_class + '/'
            if (not os.path.exists(path)):
                os.makedirs(path)
            im.save((path + im_name), 'JPEG')
        except OSError:
            print('Skipped image ', i, ' of ', image_class)
            continue

def split_dataset(dataset_path, train_fraction):
    print("Applying split::")
    print("Train: ", (train_fraction * 100), " Test: ", round((1 - train_fraction) * 100))
    classes = os.listdir(dataset_path)
    for c in classes:
        folders = os.listdir(dataset_path + c + '/')
        images = []
        for f in folders:
            images.extend(glob.glob(dataset_path + c + '/' + f + '/*.jpg'))
        size = len(images)
        train = images[:int(train_fraction * size)]
        test = images[int(train_fraction * size):]
        print("Splitting images of class: ", c, " Train: ", len(train), " Test: ", len(test))
        save_split(c, 'Train', train)
        save_split(c, 'Test', test)
        
def main():
    dataset_path = "Images/"
    split_dataset(dataset_path, 0.8)

if __name__ == '__main__':
    main()

import os
import glob
from PIL import Image

def save_partition(image_class, partition, images):
    for i, image in enumerate(images):
        try:
            im  = Image.open(image)
            im_name = image_class + '-' + str(i) + '.jpg'
            path = "./Partitioned_Dataset/partition_" + partition + "/" + image_class + '/'
            if (not os.path.exists(path)):
                os.makedirs(path)
            im.save((path + im_name), 'JPEG')
        except OSError:
            print('Skipped image ', i, ' of ', image_class)
            continue

def save_all_partitions(c, partitions):
    for (i, partition) in enumerate(partitions):
        save_partition(c, str(i), partition)

def distribute_images(partitions, images):
    n = len(partitions)
    for (i, im) in enumerate(images):
        partitions[(i % n)].append(im)
    return partitions

def partition_dataset(dataset_path, num_partitions=2):
    print("Applying partition::")
    print("Number of partitions: ", num_partitions)
    classes = os.listdir(dataset_path)
    for c in classes:
        folders = os.listdir(dataset_path + c + '/')
        images = []
        for f in folders:
            images.extend(glob.glob(dataset_path + c + '/' + f + '/*.jpg'))
        print("Partitioning images of class: ", c, " | Number of images:", len(images),
                " | Average number of images per partition: ", (len(images) // num_partitions))
        partitions = [[] for i in range(num_partitions)]
        partition = distribute_images(partitions, images)
        save_all_partitions(c, partitions)

def main():
    dataset_path = "Images/"
    partition_dataset(dataset_path, num_partitions=5)

if __name__ == '__main__':
    main()


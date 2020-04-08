import os
import shutil

original_dataset_dir = 'C:/Users/User/Desktop/celeb/'

base_dir = 'C:/Users/User/PycharmProjects/Local/WGAN/celeb'
#os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'training')
#os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
#os.mkdir(test_dir)

fnames = ['{}.jpg'.format(i) for i in range(1,10000)]
for fname in fnames:
    source = os.path.join(original_dataset_dir, fname)
    destionation = os.path.join(train_dir, fname)
    shutil.copyfile(source,destionation)

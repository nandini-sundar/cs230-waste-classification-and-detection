import argparse
import random
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='images_unsplit/', help="Directory with the waste classification dataset")
parser.add_argument('--output_dir', default='images/', help="Where to write the new data")

def save_image_and_xml(data_dir, filename, output_dir):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    imagesFolder = os.path.join(data_dir,"images")
    print(imagesFolder)
    annotationsFolder = os.path.join(data_dir,"annotations")
    print(annotationsFolder)
    sourceImg = os.path.join(imagesFolder,filename+".jpg")
    sourceXml = os.path.join(annotationsFolder+'/voc_xmls/'+,filename+".xml")
    sourceTxt = os.path.join(annotationsFolder+'/yolo_labels/'+,filename+".txt")
    destinationImg = os.path.join(output_dir,filename+".jpg")
    destinationXml = os.path.join(output_dir, filename + ".xml")
    destinationTxt = os.path.join(output_dir, filename + ".txt")
    dest = shutil.copyfile(sourceImg, destinationImg)
    dest = shutil.copyfile(sourceXml, destinationXml)
    dest = shutil.copyfile(sourceTxt, destinationTxt)

if __name__ == '__main__':
    args = parser.parse_args()
    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
    filenames = os.listdir(args.data_dir+'/images')
    print(filenames)
    filenames = [f.replace('.jpg','') for f in filenames if f.endswith('.jpg')]

    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.8 * len(filenames))
    train_filenames = filenames[:split]
    print(len(train_filenames))
    valtest_filenames = filenames[split:]
    split = int(0.5 * len(valtest_filenames))
    val_filenames = valtest_filenames[:split]
    print(len(val_filenames))
    test_filenames = valtest_filenames[split:]
    print(len(test_filenames))

    filenames = {'train': train_filenames,
                 'val': val_filenames,
                 'test':test_filenames}
    print(filenames)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    for split in ['train', 'val', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_set'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        for filename in filenames[split]:
            print(filename)
            save_image_and_xml(args.data_dir, filename, output_dir_split)

import os
import glob


def get_cityscapes_list_augmented(root, image_path, label_path, lst_path, is_fine=True, sample_rate=1):
    index = 0
    train_lst = []
    #label_prefix = 'gtFine_labelIds' if is_fine else 'gtCoarse_labelIds'
    label_prefix = 'gtFine_instanceIds' if is_fine else 'gtCoarse_labelIds'

    # images
    all_images = glob.glob(os.path.join(root, image_path, '*/*.png'))
    all_images.sort()
    for p in all_images:
        if os.path.isfile(p):
            p = p.replace(root, '')
            l = p.replace(image_path, label_path).replace('leftImg8bit', label_prefix)

            index += 1
            if index % 100 == 0:
                print "%d out of %d done." % (index, len(all_images))
            if index % sample_rate != 0:
                continue
            #for i in range(1, 8):
            for i in range(4, 5):
                train_lst.append([str(index), p, l, "512", str(256 * i)])
                #train_lst.append([str(index), p, l])

        else:
            print "dismiss %s" % (p)

    train_out = open(lst_path, "w")
    for line in train_lst:
        print >> train_out, '\t'.join(line)


if __name__ == '__main__':
    root = '../data/cityscapes/'
    image_path = 'leftImg8bit/train/'
    label_path = 'gtFine/train/'
    lst_path = '../data/cityscapes/train_DUC.lst'

    get_cityscapes_list_augmented(root, image_path, label_path, lst_path, is_fine=True, sample_rate=1)

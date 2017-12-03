from __future__ import print_function, division
import os, sys, cv2
os.chdir("/home/asaeed9/work/deeplearning")
from theano.sandbox import cuda
#cuda.use('gpu1')
import utils; reload(utils)
from utils import *
from IPython.display import FileLink
from keras.preprocessing import image, sequence
from shutil import copyfile, move
from random import shuffle

####
from keras.layers.convolutional import *
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
import datetime

data_dir="/home/asaeed9/work/data/2cat_18112017"
data_src="sample"
data_dir_F = data_dir + "/F/"
#data_dir_F_sample = data_dir + "/F_sample/"
path="/home/asaeed9/work/data/2cat_18112017/" + data_src + "/"
results_path = "/home/asaeed9/work/data/2cat_18112017/"+ data_src + "/results"
test_path = path + '/test/' #We use all the test data


def get_file_count(dir_path):
    return len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])


def get_train_valid_size():
    tr_count = get_file_count(path + "/train/cats") + get_file_count(path + "/train/dogs")
    vl_count = get_file_count(path + "/valid/cats") + get_file_count(path + "/valid/dogs")

    return tr_count, vl_count


def move_files(src_path, dest_path, pattern, n):
    #     print(src_path)
    #     print(dest_path)
    #     print(pattern)
    #     print(n)

    os.chdir(src_path)

    try:
        if type(pattern) is np.ndarray:
            for img in pattern:
                move(os.path.join(src_path + '/', os.path.basename(img)),
                     os.path.join(dest_path + '/', os.path.basename(img)))
        else:
            if n:
                g = glob(src_path + '/' + pattern)
                shuf = np.random.permutation(g)
                for i in range(n):
                    move(os.path.join(src_path + '/', os.path.basename(shuf[i])),
                         os.path.join(dest_path + '/', os.path.basename(shuf[i])))
            else:
                for file in glob(src_path + '/' + pattern):
                    move(os.path.join(src_path + '/', os.path.basename(file)),
                         os.path.join(dest_path + '/', os.path.basename(file)))

    except IOError, e:
        print("Unable to move file - {}".format(e))


def prep_experiments():
    move_files(path + "/train/" + "/cats", data_dir_F, "cat*.jpg", None)
    move_files(path + "/train/" + "/dogs", data_dir_F, "dog*.jpg", None)

    move_files(path + "/valid/" + "/cats", data_dir_F, "cat*.jpg", None)
    move_files(path + "/valid/" + "/dogs", data_dir_F, "dog*.jpg", None)

    move_files(path + "/unlabel/" + "/cats", data_dir_F, "cat*.jpg", None)
    move_files(path + "/unlabel/" + "/dogs", data_dir_F, "dog*.jpg", None)

    move_files(path + "/test/" + "/cats", data_dir_F, "cat*.jpg", None)
    move_files(path + "/test/" + "/dogs", data_dir_F, "dog*.jpg", None)


def prep_experiment2(block):
    move_files(path + "/train/" + "/cats", data_dir_F, "cat*.jpg", None)
    move_files(path + "/train/" + "/dogs", data_dir_F, "dog*.jpg", None)

    move_files(path + "/valid/" + "/cats", data_dir_F, "cat*.jpg", None)
    move_files(path + "/valid/" + "/dogs", data_dir_F, "dog*.jpg", None)

    print("Training/Validation images moved to set F.")

    move_block_images(block['train'], data_dir_F, path + '/train/', False)
    move_block_images(block['valid'], data_dir_F, path + '/valid/', False)

    print("First block of images moved to Training Set")

    move_files(data_dir_F, path + "/unlabel/" + "/cats", "cat*.jpg", None)
    move_files(data_dir_F, path + "/unlabel/" + "/dogs", "dog*.jpg", None)

    print("All rest moved to Unlabel.")


def move_random_data(n, src_dir, dest_path, clean):
    if clean:
        move_files(dest_path + "/cats", src_dir, "*.jpg", None)
        move_files(dest_path + "/dogs", src_dir, "*.jpg", None)

    move_files(src_dir, dest_path + "/cats", "cat*.jpg", n)
    move_files(src_dir, dest_path + "/dogs", "dog*.jpg", n)


def move_block_images(images, src_dir, dest_path, clean):
    if clean:
        move_files(dest_path + "/cats", src_dir, "*.jpg", None)
        move_files(dest_path + "/dogs", src_dir, "*.jpg", None)

    move_files(src_dir, dest_path + "/cats", images['cats'], None)
    move_files(src_dir, dest_path + "/dogs", images['dogs'], None)


def get_data_blocks(src_dir):
    os.chdir(src_dir)
    r_set = []
    batch_size = 250

    g_cat = glob(src_dir + "/cat*.jpg")
    shuf_cats = np.random.permutation(g_cat)

    g_dog = glob(src_dir + "/dog*.jpg")
    shuf_dogs = np.random.permutation(g_dog)

    rk_cats = int(math.floor(len(shuf_cats) / batch_size))
    rk_dogs = int(math.floor(len(shuf_dogs) / batch_size))

    if rk_cats != rk_dogs:
        print("cats/dogs set isn't equal.")

    for index in range(rk_cats):
        valid_size = int(math.floor(.2 * batch_size))
        train_size = batch_size - valid_size

        block_start = batch_size * index
        block_end = batch_size * (index + 1)
        # print(block_start, block_end)
        r_set.append({'train': {'cats': shuf_cats[block_start:block_start + train_size],
                                'dogs': shuf_dogs[block_start:block_start + train_size]},
                      'valid': {'cats': shuf_cats[block_start + train_size:block_end],
                                'dogs': shuf_dogs[block_start + train_size:block_end]}})

    return r_set


def fit(old_model, path, results_path, nepoch, batch_size, train_size, valid_size):
    gen_t = image.ImageDataGenerator(rotation_range=15, height_shift_range=0.05,
                                     shear_range=0.1, channel_shift_range=20, width_shift_range=0.1)

    tr_batches = gen_t.flow_from_directory(path + 'train', batch_size=batch_size)
    val_batches = gen_t.flow_from_directory(path + 'valid', class_mode='categorical',
                                            shuffle=True, batch_size=batch_size * 2)
    model = None
    if old_model:
        model = train_model(old_model, tr_batches, val_batches, nepoch)
    else:
        model = train_model(None, tr_batches, val_batches, nepoch)

    # model.summary()

    # model.save_weights(results_path+ '/' + 'ft_' + str(train_size) + '.e' + str(nepoch))
    last_file_timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
    # print('File Time Stamp:{}'.format(last_file_timestamp))
    model.save_weights(results_path + '/ft_{}'.format(last_file_timestamp))

    # print("saved file...")

    # model.load_weights(results_path+'/ft_{}'.format(last_file_timestamp))

    return model, last_file_timestamp


def get_train_model():
    model = Sequential([
        BatchNormalization(axis=1, input_shape=(3, 256, 256)),
        Convolution2D(32, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3, 3)),
        Dropout(0.2),
        #             Convolution2D(64,3,3, activation='relu'),
        #             BatchNormalization(axis=1),
        # MaxPooling2D((3,3)),
        Convolution2D(64, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3, 3)),
        Dropout(0.2),
        Convolution2D(32, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3, 3)),
        Dropout(0.2),
        Flatten(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])
    model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, tr_batches, val_batches, epoch):
    if not model:
        model = Sequential([
            BatchNormalization(axis=1, input_shape=(3, 256, 256)),
            Convolution2D(32, 3, 3, activation='relu'),
            BatchNormalization(axis=1),
            MaxPooling2D((3, 3)),
            Dropout(0.2),
            #             Convolution2D(64,3,3, activation='relu'),
            #             BatchNormalization(axis=1),
            # MaxPooling2D((3,3)),
            Convolution2D(64, 3, 3, activation='relu'),
            BatchNormalization(axis=1),
            MaxPooling2D((3, 3)),
            Dropout(0.2),
            Convolution2D(32, 3, 3, activation='relu'),
            BatchNormalization(axis=1),
            MaxPooling2D((3, 3)),
            Dropout(0.2),
            Flatten(),
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(2, activation='softmax')
        ])
        model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(tr_batches, tr_batches.nb_sample, nb_epoch=2, validation_data=val_batches,
                        nb_val_samples=val_batches.nb_sample)

    model.optimizer.lr = 0.1
    model.fit_generator(tr_batches, tr_batches.nb_sample, nb_epoch=1, validation_data=val_batches,
                        nb_val_samples=val_batches.nb_sample)

    model.optimizer.lr = 0.001
    model.fit_generator(tr_batches, tr_batches.nb_sample, nb_epoch=epoch - 3, validation_data=val_batches,
                        nb_val_samples=val_batches.nb_sample)

    return model


def get_test_model():
    model = Sequential([
        BatchNormalization(axis=1, input_shape=(3, 256, 256)),
        Convolution2D(32, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3, 3)),
        #             Convolution2D(64,3,3, activation='relu'),
        #             BatchNormalization(axis=1),
        # MaxPooling2D((3,3)),
        Convolution2D(64, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3, 3)),
        Convolution2D(32, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3, 3)),
        Flatten(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dense(2, activation='softmax')
    ])

    model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def run_experiment1(block, nepoch, batch_size, tr_model):
    move_block_images(block['train'], data_dir_F, path + '/train/', False)
    move_block_images(block['valid'], data_dir_F, path + '/valid/', False)

    train_size, valid_size = get_train_valid_size()
    tr_model, file_timestamp = fit(tr_model, path, results_path, nepoch, batch_size, train_size, valid_size)

    model = None
    model = get_test_model()
    # model.load_weights(results_path+'/ft_' + str(train_size) + '.e' + str(nepoch))
    # print('{0}/ft_{1}'.format(last_file_timestamp))
    # print('Last File Timestamp- before loading:{}'.format(file_timestamp))
    model.load_weights(results_path + '/ft_{}'.format(file_timestamp))

    print("\nVerification on Experiment#1 Test set.")
    probs, test_batches, loss, accuracy = predict(path, model, "test")
    # print('\nExperiment#1 Accuracy:{}'.format(accuracy))
    # print('Experiment#2 Loss:{}'.format(loss))

    return probs, test_batches, loss, accuracy, train_size, valid_size, tr_model


def onehot(x): return np.array(OneHotEncoder().fit_transform(x.reshape(-1, 1)).todense())


def pred_batch(imgs, classes):
    preds = model.predict(imgs)
    idxs = np.argmax(preds, axis=1)

    print('Shape: {}'.format(preds.shape))
    print('First 5 classes: {}'.format(classes[:5]))
    print('First 5 probabilities: {}\n'.format(preds[:5]))
    print('Predictions prob/class: ')

    for i in range(len(idxs)):
        idx = idxs[i]
        print('  {:.4f}/{}'.format(preds[i, idx], classes[idx]))


def predict(path, model, predict_type):
    gen_test = image.ImageDataGenerator()
    test_batches = gen_test.flow_from_directory(path + predict_type, class_mode=None, target_size=(256, 256),
                                                shuffle=False, batch_size=1)
    test_data = np.concatenate([test_batches.next() for i in range(test_batches.nb_sample)])
    test_labels = onehot(test_batches.classes)
    score = model.evaluate(test_data, test_labels)

    probs = model.predict(test_data)

    # loss_score.append(score[0])
    # accuracy_score.append(score[1])

    # print("\nLoss:{}, Accuracy:{}".format(score[0], score[1]))
    # print("\nProbabilities:{}".format(probs))
    return probs, test_batches, score[0], score[1]


def move_samples_imbalance(retrain_set, dest_path):
    for fil in retrain_set:
        fil_cpy = fil[fil.find('/') + 1:]

        if "cat" in fil_cpy:
            os.rename(os.path.join(path + "unlabel/cats/" + fil_cpy),
                      os.path.join(path + dest_path + "/cats/" + fil_cpy))
        else:
            os.rename(os.path.join(path + "unlabel/dogs/" + fil_cpy),
                      os.path.join(path + dest_path + "/dogs/" + fil_cpy))

    return len(retrain_set)  # images copied


def move_unlabel_block(retrain_set):
    ndog = sum('dog' in name for name in retrain_set)
    ncat = sum('cat' in name for name in retrain_set)
    valid_dogs = int(math.floor(.2 * ndog))
    valid_cats = int(math.floor(.2 * ncat))
    train_dogs = ndog - valid_dogs
    train_cats = ncat - valid_cats

    # print("Total Dogs: {}, Total Cats: {}".format(ndog, ncat))
    # print("Valid Dog: {}, Valid Cat: {}".format(valid_dog, valid_cat))

    dog_images = [os.path.basename(img) for img in retrain_set if 'dog' in img]
    cat_images = [os.path.basename(img) for img in retrain_set if 'cat' in img]

    train_dog_images = np.array(dog_images[0:train_dogs])
    valid_dog_images = np.array(dog_images[train_dogs:train_dogs + valid_dogs])
    train_cat_images = np.array(cat_images[0:train_cats])
    valid_cat_images = np.array(cat_images[train_cats:train_cats + valid_cats])

    # train_dog_images
    move_files(path + "/unlabel/dogs/", path + "/train/" + "/dogs", train_dog_images, None)
    move_files(path + "/unlabel/cats/", path + "/train/" + "/cats", train_cat_images, None)
    move_files(path + "/unlabel/dogs/", path + "/valid/" + "/dogs", valid_dog_images, None)
    move_files(path + "/unlabel/cats/", path + "/valid/" + "/cats", valid_cat_images, None)


def run_experiment2(nepoch, batch_size, tr_model):
    retrain_size = 500
    another = False

    train_size, valid_size = get_train_valid_size()
    tr_model, file_timestamp = fit(tr_model, path, results_path, nepoch, batch_size, train_size, valid_size)

    model = None
    model = get_test_model()
    # model.load_weights(results_path+'/ft_' + str(train_size) + '.e' + str(nepoch))
    # print('{0}/ft_{1}'.format(last_file_timestamp))
    # print('Last File Timestamp- before loading:{}'.format(file_timestamp))
    model.load_weights(results_path + '/ft_{}'.format(file_timestamp))

    print("\nVerification on Unlabel set.")
    probs, test_batches, loss, accuracy = predict(path, model, "unlabel")
    print('\nUnlabel Accuracy:{}'.format(accuracy))
    print('Unlabel Loss:{}'.format(loss))

    # get the top retrain_size most confused images
    retrain_idx = np.argsort(abs(0.5 - probs[:, 1]))[:retrain_size]
    retrain_set = [test_batches.filenames[i] for i in retrain_idx]

    if get_file_count(path + "/unlabel/cats") + get_file_count(path + "/unlabel/dogs") >= 500:
        another = True

    move_unlabel_block(retrain_set)

    print("\nVerification on Test set.")
    model_test = None
    model_test = get_test_model()
    # model.load_weights(results_path+'/ft_' + str(train_size) + '.e' + str(nepoch))
    # print('{0}/ft_{1}'.format(last_file_timestamp))
    # print('Last File Timestamp- before loading:{}'.format(file_timestamp))
    model_test.load_weights(results_path + '/ft_{}'.format(file_timestamp))

    probs, test_batches, loss, accuracy = predict(path, model_test, "test")

    return probs, test_batches, loss, accuracy, train_size, valid_size, another, tr_model

if __name__ == "__main__":
    nepoch = 50
    batch_size = 100
    tr_model = None

    exp1_loss_array = []
    exp1_accuracy_array = []
    exp1_training_size = []
    exp2_loss_array = []
    exp2_accuracy_array = []
    exp2_training_size = []

    prep_experiments()
    # build test set...
    test_set = 200  # each
    move_random_data(test_set, data_dir_F, test_path, True)
    r_set = get_data_blocks(data_dir_F)

    for block in r_set:
        probs, test_batches, loss, accuracy, train_size, valid_size,tr_model = run_experiment1(block, nepoch, batch_size,
                                                                                      tr_model)

        exp1_accuracy_array.append(accuracy)
        exp1_loss_array.append(loss)
        exp1_training_size.append(train_size + valid_size)

        print('\nExperiment I Training Set Size:{}'.format(exp1_training_size))
        print('Experiment I Accuracy:{}'.format(exp1_accuracy_array))
        print('Expdriment I Loss:{}'.format(exp1_loss_array))

    # preparations for experiment#2 
    tr_model = None
    prep_experiment2(r_set[0])
    another = True
    while another:
        probs, test_batches, loss, accuracy, train_size, valid_size, another, tr_model  = run_experiment2(nepoch, batch_size,
                                                                                               tr_model)

        exp2_accuracy_array.append(accuracy)
        exp2_loss_array.append(loss)
        exp2_training_size.append(train_size + valid_size)

        print('\nExperiment II Training Set Size:{}'.format(exp2_training_size))
        print('Experiment II Accuracy:{}'.format(exp2_accuracy_array))
        print('Expdriment II Loss:{}'.format(exp2_loss_array)) 

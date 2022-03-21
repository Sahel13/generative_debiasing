from keras.preprocessing.image import ImageDataGenerator


def load_vae_data(source_dir, input_dim, batch_size):
    """
    Data loader for training VAE.
    """
    data_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    train_data = data_gen.flow_from_directory(
        directory=source_dir,
        target_size=input_dim[:2],
        class_mode='input',
        batch_size=batch_size,
        shuffle=True,
        subset='training'
    )
    val_data = data_gen.flow_from_directory(
        directory=source_dir,
        target_size=input_dim[:2],
        class_mode='input',
        batch_size=batch_size,
        shuffle=True,
        subset='validation'
    )
    return train_data, val_data


def load_classifier_data(source_dir, input_dim, batch_size):
    """
    `source_dir` should contain two folders -  one containing
    positive training examples and the other containing negative
    training examples.
    """
    data_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    train_data = data_gen.flow_from_directory(
        directory=source_dir,
        target_size=input_dim[:2],
        classes=['imagenet', 'celeba'],
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True,
        follow_links=True,
        subset='training'
    )
    val_data = data_gen.flow_from_directory(
        directory=source_dir,
        target_size=input_dim[:2],
        classes=['imagenet', 'celeba'],
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True,
        follow_links=True,
        subset='validation'
    )
    return train_data, val_data

import os
from utils.dataloader import load_classifier_data


# Check if the labels are mapped correctly:
def test_label_mapping():
    data_folder = os.path.join('..', 'datasets', 'combined_dataset')
    input_dim = (128, 128, 3)
    batch_size = 256
    _, val_data = load_classifier_data(data_folder, input_dim, batch_size)
    mapping = val_data.class_indices
    assert mapping['imagenet'] == 0

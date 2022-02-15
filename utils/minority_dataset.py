import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_celeba(df, image_folder, target_size, batch_size):
    """
    Load CelebA images with file names.
    Returns: (images, file_names)
    """
    data_gen = ImageDataGenerator(rescale=1./255)

    data_flow = data_gen.flow_from_dataframe(
        dataframe=df,
        directory=image_folder,
        x_col='image_id',
        y_col='image_id',
        target_size=target_size,
        class_mode='raw',
        batch_size=batch_size,
        shuffle=False
    )

    return data_flow


class MinorityDataset():
    """
    Class to contain the functions needed to create the minority dataset.
    """
    def __init__(self, data_flow, vae, latent_dim):
        self.vae = vae
        self.latent_dim = latent_dim
        self.data_flow = data_flow
        self.batch_size = data_flow[0][0].shape[0]

    def get_latent_mean(self):
        """
        Get the mean values of all latent variables over the entire dataset.
        Returns: An array of dimensions [num_images, latent_dim].
        """
        mean, _ = self.vae.encoder.predict(self.data_flow, verbose=1)
        return mean

    def get_sub_dataset(self, mean, bins=20, extremes=3):
        """
        Find the images with under-represented features by plotting histograms.
        Returns: A list with boolean values indicating whether a
        given image is present in the minority dataset or not.
        Each position in the list corresponds to the position in
        the alphabetically sorted list of images.
        """
        # Initialize an empty minority dataset.
        minority_list = np.full(shape=mean.shape[0], fill_value=False)

        # For each latent variable:
        for i in range(self.latent_dim):
            latent_distribution = mean[:, i]

            # Generate a histogram of the latent distribution:
            _, bin_edges = np.histogram(
                latent_distribution, density=False, bins=bins)
            bin_edges[0] = -float('inf')
            bin_edges[-1] = float('inf')

            # Find which bin in the latent distribution
            # every data sample falls into.
            bin_idx = np.digitize(latent_distribution, bin_edges)

            # Find the extremum bins.
            front = range(extremes)
            back = range(bins - extremes, bins, 1)
            indices = [*front, *back]

            # Find which images fall in the extremum bins.
            ind_minority_list = np.isin(bin_idx, indices)
            minority_list = np.logical_or(minority_list, ind_minority_list)

        # Print the number of images in the minority dataset.
        print(f"The minority dataset has {minority_list.sum()} images"
              + f" ({minority_list.sum()/len(minority_list) * 100:.2f}% of total).")

        return minority_list

    def create_new_df(self, minority_list, df, file_name):
        """
        Saves the minority dataset in a new file.
        """
        min_df = df.copy()

        # Drop rows that are not present in the minority dataset.
        rows_to_drop = []
        for i in range(len(minority_list)):
            if not minority_list[i]:
                rows_to_drop.append(i)

        min_df.drop(rows_to_drop, inplace=True)

        # Save the DataFrame to a new file.
        min_df.to_csv(file_name, index=False)

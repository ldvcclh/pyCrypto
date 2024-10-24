import pandas as pd

class DataCleaner:
    """
    A class to clean time-series data.

    Parameters:
    -----------
    filename (str): The name of the file containing the data.
    path (str): The path to the folder containing the data.
    print_info (bool, optional): Whether to print information during the cleaning process. Defaults to False.

    Methods:
    --------
    clean_data():
        Executes the main cleaning pipeline.
    check_duplicates():
        Removes duplicate indexes.
    check_nan():
        Checks for NaN values and logs column-level details.
    fill_missing_seconds():
        Fills missing time steps with forward fill.
    """

    def __init__(self, filename, datapath, print_info=False):
        self.filename = filename
        self.print_info = print_info
        self.datapath = datapath
        self.df = pd.read_parquet(self.filename, engine='pyarrow')

    def log(self, message):
        """Helper function to handle optional printing."""
        if self.print_info:
            print(message)

    def check_duplicates(self):
        """Checks and removes duplicate indexes."""
        duplicate_indexes = self.df.index[self.df.index.duplicated()]
        self.log(f"Duplicated indexes: {duplicate_indexes}")

        if len(duplicate_indexes) > 0:
            self.df = self.df[~self.df.index.duplicated(keep='first')]

    def check_nan(self,final=False):
        """Checks for NaN values and logs details."""
        if self.df.isnull().values.any():
            self.log("NaN values found in the DataFrame.")
            nan_count = self.df.isnull().sum()
            if final:
                raise ValueError(
                    f"Error: NaN values remain in the DataFrame after cleaning. \nNumber of NaN by column: \n{nan_count}")
            else:
                self.log(f"Number of NaN by column: \n{nan_count}")
        else:
            self.log("No NaN values in the DataFrame.")

    def fill_missing_seconds(self):
        """Fills missing seconds in the time series by forward filling."""
        all_seconds = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq='s')
        missing_indexes = all_seconds.difference(self.df.index)

        if len(missing_indexes) == 0:
            self.log("Every second is present.")
        else:
            self.log(f"It misses {len(missing_indexes)} second(s)")
            self.log(missing_indexes)

        # Reindex and fill missing values
        self.df = self.df.reindex(all_seconds)
        #self.df.fillna(method='ffill', inplace=True)
        self.df.ffill(inplace=True)

    def clean_data(self):
        """Executes the entire cleaning process."""
        # Ensure the index is in datetime format
        self.df.index = pd.to_datetime(self.df.index)

        # Run individual checks
        self.check_duplicates()
        self.check_nan()
        self.fill_missing_seconds()
        self.check_nan(final=True)  # Final NaN check after filling

        # Save cleaned data
        self.df.to_parquet(self.datapath+'data_cleaned.parquet', engine='pyarrow')

        return self.df


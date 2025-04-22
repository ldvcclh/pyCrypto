import pandas as pd

class TargetCalculator:
    """
    A class to calculate various targets for time-series data.

    Parameters:
    -----------
    filename (str): The name of the file containing the cleaned data.
    path (str): The path to the folder containing the data.
    print_info (bool, optional): Whether to print information during the process. Defaults to False.

    Methods:
    --------
    calculate_pct_change_target(time=120):
        Calculates the percentage change target over the specified time period.
    save_data(output_filename):
        Saves the DataFrame to a parquet file.
    """

    def __init__(self, filename, datapath, print_info=False):
        self.filename = filename
        self.print_info = print_info
        self.datapath = data_path
        self.df = pd.read_parquet(self.filename, engine='pyarrow')

    def log(self, message):
        """Helper function to handle optional printing."""
        if self.print_info:
            print(message)

    def calculate_pct_change_target(self, time=120):
        """
        Calculates the percentage change target over a specified time period.

        Parameters:
        -----------
        time (int): The number of seconds over which to calculate the percentage change. Default is 120 seconds.
        """
        self.log(f"Calculating percentage change target over {time} seconds...")

        self.df['pct_change_close'] = self.df['Close'].pct_change(periods=time).shift(periods=-time)
        self.df['target'] = (self.df['pct_change_close'] > 0).astype(int)

        self.log("Percentage change target calculation complete.")

    def save_data(self, output_filename):
        """
        Saves the DataFrame with the calculated target to a parquet file.

        Parameters:
        -----------
        output_filename (str): The name of the output file to save the data.
        """
        self.df.to_parquet(self.datapath+output_filename, engine='pyarrow')
        self.log(f"Data saved to {self.datapath+output_filename}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Analysis:
    def __init__(self, json_url):
        self.json_url = json_url
        self.df = None
        self.deviation_columns = ['mean', 'max', 'min', 'floor_mean', 'floor_max', 'floor_min', 'ceiling_mean', 'ceiling_max', 'ceiling_min']
        self.plot_folder = "plots"

    def _create_plot_folder(self):
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

    def _calculate_accuracy(self):
        print(' - - Accuracy - - ')
        self.df['corner_diff'] = abs(self.df['gt_corners'] - self.df['rb_corners'])
        accuracy = len(self.df[self.df['corner_diff'] == 0]) / len(self.df) * 100
        print("Accuracy of corner prediction: {:.2f}%".format(accuracy))
        print()

    def _analyze_deviations(self):
        '''
        The model's average deviation is approx. 12.90 degrees, with a high standard deviation of 21.84. 
        The maximum deviation reaches about 121.30 degrees. 
        The mean floor and ceiling deviations are 11.02 and 14.77 degrees, respectively. 
        This indicates significant errors in certain predictions made by the model.
        '''
        print(' - - Deviation values analysis - - ')
        print(self.df[self.deviation_columns].describe())
        print()

    def _analyze_rooms(self):
        '''
        The output provides the average corner difference (which is 0 for all rooms due to the high corner prediction accuracy) and the average deviation for each room type.
        This information is useful to identify if there are specific rooms where your model is consistently performing poorly.
        For instance, the '#229 - Dining Room' has an average deviation of 55.15, which is much higher than the mean deviation of 12.90.
        This might indicate that your model has difficulty with this room type and could benefit from further training or fine-tuning.
        '''
        print(' - - Room-wise analysis - - ')
        room_analysis = self.df.groupby('name').agg({'corner_diff': 'mean', 'mean': 'mean'}).rename(columns={'corner_diff': 'avg_corner_diff', 'mean': 'avg_deviation'})
        print(room_analysis)
        print()

    def _create_plots(self):
        plot_paths = []
        for column in self.deviation_columns:
            fig = plt.figure(figsize=(10, 5))
            sns.histplot(data=self.df, x=column, bins=30, kde=True)
            plt.title(f'Distribution of {column}')
            plot_path = f'{self.plot_folder}/{column}_histogram.png'
            fig.savefig(plot_path)
            plt.close(fig)
            plot_paths.append(plot_path)
        return plot_paths

    def analyze_and_draw(self):
        self.df = pd.read_json(self.json_url)
        self._create_plot_folder()
        self._calculate_accuracy()
        self._analyze_deviations()
        self._analyze_rooms()
        return self._create_plots()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


url = "https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json"
df = pd.read_json(url)


df['corner_diff'] = abs(df['gt_corners'] - df['rb_corners'])
accuracy = len(df[df['corner_diff'] == 0]) / len(df) * 100
print("Accuracy of corner prediction: {:.2f}%".format(accuracy))


# Deviation values analysis
deviation_columns = ['mean', 'max', 'min', 'floor_mean', 'floor_max', 'floor_min', 'ceiling_mean', 'ceiling_max', 'ceiling_min']
print(df[deviation_columns].describe())
'''
The model's average deviation is approx. 12.90 degrees, with a high standard deviation of 21.84. 
The maximum deviation reaches about 121.30 degrees. 
The mean floor and ceiling deviations are 11.02 and 14.77 degrees, respectively. 
This indicates significant errors in certain predictions made by the model.
'''


# Room-wise analysis
room_analysis = df.groupby('name').agg({'corner_diff': 'mean', 'mean': 'mean'}).rename(columns={'corner_diff': 'avg_corner_diff', 'mean': 'avg_deviation'})
print(room_analysis)
'''
The output provides the average corner difference (which is 0 for all rooms due to the high corner prediction accuracy) and the average deviation for each room type.
This information is useful to identify if there are specific rooms where your model is consistently performing poorly.
For instance, the '#229 - Dining Room' has an average deviation of 55.15, which is much higher than the mean deviation of 12.90.
This might indicate that your model has difficulty with this room type and could benefit from further training or fine-tuning.
'''


# Visualizing the deviations
for column in deviation_columns:
    fig = plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x=column, bins=30, kde=True)
    plt.title(f'Distribution of {column}')
    fig.savefig(f'plots/{column}_histogram.png')
    plt.close(fig)

    '''
    Based on the histograms we can see positive-skewed chart, which in most cases tells that the model is performing well.
    '''

import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
matplotlib.use('Agg')

img_dir = "./Static/img"

"""A plotter to plot the prediction result for the audio:"""

class TimescalePlotter:
    def __init__(self, x, sample_rate, label_values, prob_values, speech_values, filename, color_map={'coughing': 'red', 'sneezing': 'blue', 'speech': 'green', 'silence': 'grey'}):
        audio_duration = float(len(x)/sample_rate)
        self.timestamps = list(np.linspace(start=0, stop=audio_duration, num=len(label_values)))
        data = {
            'Time': self.timestamps,
            'Label': list(label_values),
            'Probability': list(prob_values),
        }
        self.df = pd.DataFrame(data)
        self.color_map = color_map
        self.intervals = self._get_intervals()
        self.merged_intervals = self._merge_intervals()     # neglecting probs
        self.filename = filename[:-4]
        self.speech_probs = speech_values

    def _merge_intervals(self):
        merged_intervals = []
        current_label = self.df['Label'].iloc[0]
        start_time = self.df['Time'].iloc[0]

        for i in range(1, len(self.df)):
            if self.df['Label'].iloc[i] != current_label:
                merged_intervals.append((start_time, self.df['Time'].iloc[i], current_label))
                current_label = self.df['Label'].iloc[i]
                start_time = self.df['Time'].iloc[i]

        # Add the last interval
        merged_intervals.append((start_time, self.df['Time'].iloc[-1] + 1, current_label))
        return merged_intervals

    def _get_intervals(self):
        intervals = []
        current_label = self.df['Label'].iloc[0]
        current_prob = self.df['Probability'].iloc[0]
        start_time = self.df['Time'].iloc[0]

        for i in range(1, len(self.df)):
            intervals.append((start_time, self.df['Time'].iloc[i], current_label, current_prob))
            current_label = self.df['Label'].iloc[i]
            current_prob = self.df['Probability'].iloc[i]
            start_time = self.df['Time'].iloc[i]

        # Add the last interval
        intervals.append((start_time, self.df['Time'].iloc[-1] + 1, current_label, current_prob))
        return intervals

    def plot_labels(self, figsize=(20, 2)):
        fig, ax = plt.subplots(figsize=figsize)  # Adjust the height by setting figsize to (width, height)

        # Add colored intervals with transparencies corresponding to probabilities
        for start_time, end_time, label, prob in self.intervals:
            ax.axvspan(start_time, end_time, color=self.color_map[label], alpha=prob, label=label)

        # Formatting the plot
        ax.set_xlim(self.df['Time'].min() - 1, self.df['Time'].max() + 2)  # Adjust x-axis limits
        ax.set_ylim(0, 1.0)  # Adjust y-axis limits to focus on the labels
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Label')
        ax.set_title("Framewise Classification of "+self.filename+".wav")  # Set the plot title

        # Customizing ticks and grid
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Show up to 10 major ticks on the x-axis
        ax.yaxis.set_visible(False)  # Hide y-axis
        ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)

        # Create custom legend handles with alpha=1.0
        unique_handles = [plt.Line2D([0, 1], [0, 0], color=color, lw=8, alpha=1.0) for label, color in self.color_map.items()]
        unique_labels = [*self.color_map]
        # Add legend with custom handles
        ax.legend(handles=unique_handles, labels=unique_labels, loc='upper right')

        # Display the plot
        plt.savefig(os.path.join(img_dir,self.filename)+"_labels.png")


    def plot_speech_probs(self, figsize=(20, 2), k=2):
        time_intervals = self.df['Time']
        probability_data = self.speech_probs

        # Spline interpolation for smooth curve
        x_smooth = np.linspace(time_intervals.min(), time_intervals.max(), 300)
        spl = make_interp_spline(time_intervals, probability_data, k=k)  # k is the degree of the splines
        y_smooth = spl(x_smooth)
        # Clip the smoothed data to ensure it remains within [0, 1]
        y_smooth = np.clip(y_smooth, 0, 1)

        plt.figure(figsize=figsize)
        plt.plot(x_smooth, y_smooth, color='b', label='Smoothed Curve')
        plt.scatter(time_intervals, probability_data, color='r', label='Original Datapoints')  # Optional: show original data points
        plt.title("Speech Probability of "+self.filename+".wav")
        plt.xlabel('Time')
        plt.ylabel('Speech Probability')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(img_dir,self.filename)+"_speech_probs.png")

    def plot(self, figsize=(20, 2), k=2):
        fig, ax = plt.subplots(figsize=figsize)  # Adjust the height by setting figsize to (width, height)

        ## Add colored intervals regardless of probabilities
        # for start_time, end_time, label in self.merged_intervals:
        #     ax.axvspan(start_time, end_time, color=self.color_map[label], alpha=0.3, label=label)

        # Add colored intervals with transparencies corresponding to probabilities
        for start_time, end_time, label, prob in self.intervals:
            ax.axvspan(start_time, end_time, color=self.color_map[label], alpha=prob, label=label)

        # # Add dummy entries for labels not present in data
        # for label, color in self.color_map.items():
        #     if label not in self.df['Label'].unique():
        #         ax.axvspan(np.NaN, np.NaN, color=color, alpha=0.3, label=label)

        # Formatting the plot
        ax.set_xlim(self.df['Time'].min() - 1, self.df['Time'].max() + 2)  # Adjust x-axis limits
        ax.set_ylim(0, 1.05)  # Adjust y-axis limits to focus on the labels
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Speech Probability')
        ax.set_title("Framewise Classification and Speech Probability of "+self.filename+".wav")  # Set the plot title

        # Prepare data for plotting the speech probability curve line
        time_intervals = self.df['Time']
        probability_data = self.speech_probs

        # Spline interpolation for smooth curve
        x_smooth = np.linspace(time_intervals.min(), time_intervals.max(), 300)
        spl = make_interp_spline(time_intervals, probability_data, k=k)  # k is the degree of the splines
        y_smooth = spl(x_smooth)
        # Clip the smoothed data to ensure it remains within [0, 1]
        y_smooth = np.clip(y_smooth, 0, 1)

        # Plot the speech probability curve line
        ax.plot(x_smooth, y_smooth, color='white', alpha=0.85, label='Speech Probability')
        # ax.scatter(time_intervals, probability_data, color='white', label='Original Data')  # Optional: show original data points
        # ax.grid(True)

        # Customizing ticks and grid
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Show up to 10 major ticks on the x-axis
        # ax.yaxis.set_visible(False)  # Hide y-axis
        ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)

        # Get existing legend handles and labels
        handles, labels = ax.get_legend_handles_labels()
        # Create custom legend handles with alpha=1.0
        unique_handles = [plt.Line2D([0, 1], [0, 0], color=color, lw=8, alpha=1.0) for label, color in self.color_map.items()]
        unique_handles.append(handles[-1])
        unique_labels = [*self.color_map]
        unique_labels.append(labels[-1])
        # Add legend with custom handles
        ax.legend(handles=unique_handles, labels=unique_labels, bbox_to_anchor=(1, 1), facecolor='gold', framealpha=0.5)

        # Display the plot
        # plt.show()

        save_path = os.path.join(img_dir,self.filename)+".png"
        plt.savefig(save_path)


        return save_path

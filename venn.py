import pandas as pd
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Verdana']
from matplotlib import pyplot as plt
from matplotlib_venn import venn3

def main():
	infile = '../Data/reviews_classified.csv'
	outfile = '../Output/venn_predicted.png'
	outfile_training = '../Output/venn_labeled.png'
	outfile_measures = '../Output/venn_measures.png'
	
	cols_pred = ['food_pred_nbayes', 'service_pred_nbayes', 'money_pred_nbayes']
	cols_labeled = ['food', 'service', 'money']
	cols_measures = ['food_present', 'service_present', 'money_present']

	df = pd.read_csv(infile, usecols = cols_pred + cols_labeled + cols_measures)

	plot_venn(df, outfile, cols_pred)
	plot_venn(df, outfile_training, cols_labeled)
	plot_venn(df, outfile_measures, cols_measures)

def plot_venn(df, outfile, cols):
	[food_col, service_col, money_col] = cols

	food = len(df.loc[(df[food_col] == 1) & (df[service_col] == 0) & (df[money_col] == 0)])
	service = len(df.loc[(df[food_col] == 0) & (df[service_col] == 1) & (df[money_col] == 0)])
	money = len(df.loc[(df[food_col] == 0) & (df[service_col] == 0) & (df[money_col] == 1)])
	food_and_service = len(df.loc[(df[food_col] == 1) & (df[service_col] == 1) & (df[money_col] == 0)])
	food_and_money = len(df.loc[(df[food_col] == 1) & (df[service_col] == 0) & (df[money_col] == 1)])
	service_and_money = len(df.loc[(df[food_col] == 0) & (df[service_col] == 1) & (df[money_col] == 1)])
	all_three = len(df.loc[(df[food_col] == 1) & (df[service_col] == 1) & (df[money_col] == 1)])

	subset_sizes = [food, service, food_and_service, money, food_and_money, service_and_money, all_three]
	subset_sizes = [n + 1 if n == 0 else n for n in subset_sizes]
	v = venn3(subsets = subset_sizes, set_labels = ('Food', 'Service', 'Money'))

	for text in v.set_labels:
	    text.set_fontsize(18)

	subsets = ['001', '010', '100', '110', '101', '111', '011']
	for subset in subsets:
		try:
			label = v.get_label_by_id(subset)
			label.set_fontsize(14)
		except: pass

	plt.title("Topics Mentioned", fontsize=20)
	plt.savefig(outfile)
	plt.close()

if __name__ == "__main__":
	main()
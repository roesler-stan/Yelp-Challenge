def write_labels(cols, outfile):
	""" Write variable labels at top of file """
	with open(outfile, 'w') as outf:
		outf.write('nodedef>name INTEGER, label VARCHAR\n')
		for col_no, col in enumerate(cols):
			out_string = str(col_no) + ',' + col
			outf.write(out_string + '\n')

def write_edges(comatrix, outfile):
	with open(outfile, 'a') as outf:
		outf.write('edgedef>Source INTEGER, Target INTEGER, directed BOOLEAN, weight DOUBLE\n')
		headers = ['Source', 'Target', 'directed', 'weight']
		writer = csv.DictWriter(outf, fieldnames=headers)
		for source, target, weight in zip(comatrix.row, comatrix.col, comatrix.data):
			# Only get the upper triangle, and ignore self-edges
			if source >= target:
				continue
			writer.writerow({'Source': source, 'Target': target, 'directed': 'false', 'weight': weight})
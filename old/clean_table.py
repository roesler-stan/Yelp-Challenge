def clean_ols_table(infile, outfile):
    with open(outfile, 'w') as outf:
        with open(infile, 'r') as f:
            for line in f:
                # Skip lines that are just dashes
                if '----' in line or '====' in line:
                    continue
                line = re.sub('_', ' ', line)
                line = re.sub(' {2,100}', '+++', line)
                # If it's a number followed by a space, the space is a separator
                line = re.sub(r'(\d )', r'\1+++', line)
                # Undo the R2 line
                line = re.sub('^R2 \+\+\+', 'R2 ', line)
                if ',' in line:
                    line = line.strip()
                    words = line.split('+++')
                    words = ['"' + word + '"' for word in words]
                    line = '+++'.join(words)
                    line += '\n'
                line = re.sub('\+\+\+', ',', line)
                line += ','
                outf.write(line)
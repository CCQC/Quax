import re
import pandas as pd
pd.options.display.max_rows = 999

#def maybe(string):
#    """
#    A regex wrapper for an arbitrary string.
#    Allows a string to be present, but still matches if it is not present.
#    """
#    return r'(?:{:s})?'.format(string)

with open("00_RESULTS_8_18_2020") as f:
    text = f.read()

#trial_data_re = '(\w+)\s(\w+)\/(\w+-\w+)\s(\w+)'
trial_data_re = '(\w+)\s(\w+(?:\(\w+\))?)\/(\w+-\w+)\s(\w+\s+\w+)'
timing_re = '(\d+):(\d+)\.\d+elapsed'
memory_re = '(\d+)maxresident'

# Parse job labels
labels = re.findall(trial_data_re, text)
# Parse timings and convert to seconds
tmp_timings = re.findall(timing_re, text)
timings = []
for t in tmp_timings:
    tmp = 60 * int(t[0]) + int(t[1])
    timings.append(tmp)
# Parse memory and convert to GB
memory = re.findall(memory_re, text)
memory = [int(i) / 1e6 for i in memory]

df = pd.DataFrame(labels, columns = ['Mol', 'Method', 'Basis', 'Dertype'])
df['Timing (s)'] = timings
df['Memory (GB)'] = memory


df = df.sort_values(['Method','Basis','Dertype','Mol'])

print(df)

df.to_csv('data.csv')


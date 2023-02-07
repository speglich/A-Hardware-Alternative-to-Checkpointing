import os
import glob
import pandas as pd

extension = 'csv'

results = {'reverse' : 'results/reverse',
            'forward' : 'results/forward'}

# MPI
#results = {'reverse' : 'results-mpi/reverse',
#            'forward' : 'results-mpi/forward'}

for r in results:
    all_filenames = [i for i in glob.glob('{}/*.{}'.format(results[r], extension))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    combined_csv.to_csv('{}.csv'.format(r))
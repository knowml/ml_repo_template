# join results files from different experiments 
# sort by value (score) to easly select best performance settings

import pandas as pd
import glob

files = glob.glob('scripts/02_model_selection/*.csv')

df = pd.DataFrame()

for file in files:
    if file == 'scripts/02_model_selection\\joined_results.csv':
        continue
    df_add = pd.read_csv(file, header=0, sep='|', encoding='UTF-8').assign(file=file)
    df = pd.concat([df,df_add], axis=0, ignore_index=True)
    
df.sort_values('value').to_csv('scripts/02_model_selection/joined_results.csv', encoding='UTF-8', sep='|', index=False)
import pandas as pd
import os

# set the base directory
base_dir = r'C:\Users\giova\Documents\GitHub\daily_sentiment_feed'
data_dir = os.path.join(base_dir, 'data')
output_dir = os.path.join(base_dir, 'output')

# load the source_and_type.csv to get paywalled and credibility info
source_file = os.path.join(data_dir, 'source_and_type.csv')
source_df = pd.read_csv(source_file)

# create mappings: lowercase for matching since sources might vary in case
paywalled_set = set(source_df[source_df['IS_PAYWALLED'] == 1]['SOURCE_NAME'].str.lower().str.strip())
credibility_map = dict(zip(source_df['SOURCE_NAME'].str.lower().str.strip(), source_df['CREDIBILITY_TYPE']))

print(f"loaded {len(paywalled_set)} paywalled sources and {len(credibility_map)} credibility mappings")

# function to clean a single csv
def clean_csv(file_name):
    csv_path = os.path.join(output_dir, file_name)
    if not os.path.exists(csv_path):
        print(f"file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # ensure SOURCE column exists and is string
    if 'SOURCE' not in df.columns:
        print(f"no SOURCE column in {file_name}")
        return
    
    df['SOURCE'] = df['SOURCE'].astype(str).str.lower().str.strip()
    
    # update PAYWALLED: convert to bool, True if in paywalled_set
    if 'PAYWALLED' in df.columns:
        df['PAYWALLED'] = df['SOURCE'].isin(paywalled_set)
    else:
        df['PAYWALLED'] = df['SOURCE'].isin(paywalled_set)
        print(f"added missing PAYWALLED column to {file_name}")
    
    # update CREDIBILITY_TYPE: map from dict, default to 'Relevant Article' if no match
    if 'CREDIBILITY_TYPE' in df.columns:
        df['CREDIBILITY_TYPE'] = df['SOURCE'].map(credibility_map).fillna('Relevant Article')
    else:
        df['CREDIBILITY_TYPE'] = df['SOURCE'].map(credibility_map).fillna('Relevant Article')
        print(f"added missing CREDIBILITY_TYPE column to {file_name}")
    
    # remove rows with null published_date
    if 'PUBLISHED_DATE' in df.columns:
        initial_rows = len(df)
        df = df.dropna(subset=['PUBLISHED_DATE'])
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(f"removed {removed_rows} rows with null PUBLISHED_DATE from {file_name}")
    else:
        print(f"no PUBLISHED_DATE column in {file_name}, skipping date filter")
    
    # save back to the same file (overwrite)
    df.to_csv(csv_path, index=False)
    print(f"updated {file_name} with {len(df)} rows")

# process both files
csv_files = ['enterprise_risks_online_sentiment.csv', 'emerging_risks_online_sentiment.csv']
for file in csv_files:
    clean_csv(file)

print("cleanup complete")
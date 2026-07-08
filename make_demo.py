import pandas as pd

print("Loading massive dataset...")
df = pd.read_csv('data/creditcard.csv')

# Grab all the frauds (492 rows) and 1,000 normal transactions
print("Extracting demo sample...")
frauds = df[df['Class'] == 1]
normals = df[df['Class'] == 0].sample(n=1000, random_state=42)

# Combine and save as a lightweight file
demo_df = pd.concat([frauds, normals]).sample(frac=1, random_state=42)
demo_df.to_csv('data/demo_creditcard.csv', index=False)

print("✅ Success! demo_creditcard.csv created. It is ready for GitHub!")
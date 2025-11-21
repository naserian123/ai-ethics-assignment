# Load COMPAS dataset (adjust the filename ONLY if needed)
compas = pd.read_csv('compas-scores-two-years.csv')

# Select useful columns for fairness analysis
compas = compas[['age', 'sex', 'race', 'priors_count', 'two_year_recid', 'decile_score']]

# Remove any missing rows
compas = compas.dropna()

# Create protected attribute:
# race_black = 1  → African-American
# race_black = 0  → everyone else
compas['race_black'] = np.where(compas['race'] == 'African-American', 1, 0)

# Convert to AIF360 dataset format
label_dataset = BinaryLabelDataset(
    df=compas,
    label_names=['two_year_recid'],
    protected_attribute_names=['race_black'],
    favourable_label=0,
    unfavourable_label=1
)

# Split into train and test
train, test = label_dataset.split([0.7], shuffle=True)

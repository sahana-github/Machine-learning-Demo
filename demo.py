import pandas as pd

# Load the dataset from Excel
data = pd.read_excel('input.xlsx')

# Specify the column to categorize
column_name = 'short description'

# Define the rules and corresponding categories
rules = {
    'Onedrive': ['Onedrive', 'onedrive not working'],
    'Whatapp': ['whatsapp', 'whatpp slow'],
    'PingId': ['pingid', 'authentications'],
    # Add more rules as needed
}

# Create a new column for the predicted categories
data['Category'] = ''

# Apply the rule-based classification on each entry
for index, row in data.iterrows():
    text = str(row[column_name]).lower()
    matched_category = None
    
    for category, keywords in rules.items():
        for keyword in keywords:
            if keyword in text:
                matched_category = category
                break
                
        if matched_category:
            break
            
    data.at[index, 'Category'] = matched_category

# Save the updated data to a new Excel file
data.to_excel('output.xlsx', index=False)

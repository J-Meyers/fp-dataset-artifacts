import json
import argparse
import pandas as pd

# Same as eval_evaluation.py
# Instead using pandas dataframe to store the data

def read_evaluation_file(evaluation_file):
    lines = []
    with open(evaluation_file, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

# Load lines into a dataframe
def load_lines(lines):
    df = pd.DataFrame(lines)
    return df

# Add a column to the dataframe to indicate whether the prediction was correct
def add_correct_column(df):
    df['correct'] = df['label'] == df['predicted_label']
    return df

# Add a column to the dataframe to indicate whether the premise contains 'not' and another column to indicate whether the hypothesis contains 'not'
def add_not_columns(df):
    df['premise_contains_not'] = df['premise'].str.contains(' not ')
    df['hypothesis_contains_not'] = df['hypothesis'].str.contains(' not ')
    return df

# Add a column to the dataframe to indicate whether the premise and hypothesis both contain 'not'
def add_double_negation_column(df):
    df['double_negation'] = df['premise_contains_not'] & df['hypothesis_contains_not']
    return df

# Single negation column
def add_single_negation_column(df):
    df['single_negation'] = (df['premise_contains_not'] | df['hypothesis_contains_not']) & ~df['double_negation']
    return df

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_file', type=str, required=True)
    args = parser.parse_args()
    lines = read_evaluation_file(args.evaluation_file)
    df = load_lines(lines)
    df = add_correct_column(df)
    df = add_not_columns(df)
    df = add_double_negation_column(df)
    df = add_single_negation_column(df)

    # Print the number of correct and incorrect predictions
    print('Number of correct predictions: {}'.format(df['correct'].sum()))
    print('Number of incorrect predictions: {}'.format(len(df) - df['correct'].sum()))

    # Print the number of incorrect predictions containing not
    print('Number of incorrect predictions containing not: {}'.format(df[df['correct'] == False]['single_negation'].sum()))

    # Print the number of incorrect predictions with double negation
    print('Number of incorrect predictions with double negation: {}'.format(df[df['correct'] == False]['double_negation'].sum()))

    # Print the number of correct predictions containing not
    print('Number of correct predictions containing not: {}'.format(df[df['correct'] == True]['single_negation'].sum()))

    # Print the number of correct predictions with double negation
    print('Number of correct predictions with double negation: {}'.format(df[df['correct'] == True]['double_negation'].sum()))

    # Print the percentage of incorrect predictions containing not
    print('Percentage of incorrect predictions containing not: {}'.format(df[df['correct'] == False]['single_negation'].sum() / len(df[df['correct'] == False])))

    # Print the percentage of incorrect predictions with double negation
    print('Percentage of incorrect predictions with double negation: {}'.format(df[df['correct'] == False]['double_negation'].sum() / len(df[df['correct'] == False])))

    # Display all of the above stats in a table which is broken down by label
    print(df.groupby('label').agg({'correct': ['sum', 'count'], 'single_negation': ['sum'], 'double_negation': ['sum']}))

    # Display the above stats in a table which is broken down by label and predicted label
    print(df.groupby(['predicted_label']).agg({'correct': ['sum', 'count'], 'single_negation': ['sum'], 'double_negation': ['sum']}))

    # Same as above two prints but use percentages instead of sums and counts
    print(df.groupby('label').agg({'correct': ['mean'], 'single_negation': ['mean'], 'double_negation': ['mean']}))
    print(df.groupby(['predicted_label']).agg({'correct': ['mean'], 'single_negation': ['mean'], 'double_negation': ['mean']}))


    # Create table of percentage correct for each label on each of the three sets (with double negation, with single negation, and without negation)
    df_double_negation = df[df['double_negation'] == True]
    if not df_double_negation.empty:
        print("Double Negation")
        print(df_double_negation.groupby('label').agg({'correct': ['mean']}))
        print(df_double_negation.groupby(['label', 'predicted_label']).agg({'correct': ['mean']}))
        # Same two tables as above, now with counts instead of percentages
        print(df_double_negation.groupby('label').agg({'correct': ['sum', 'count']}))
        print(df_double_negation.groupby(['label', 'predicted_label']).agg({'correct': ['sum', 'count']}))

    df_single_negation = df[df['single_negation'] == True]
    if not df_single_negation.empty:
        print("Single Negation")
        print(df_single_negation.groupby('label').agg({'correct': ['mean']}))
        print(df_single_negation.groupby(['label', 'predicted_label']).agg({'correct': ['mean']}))
        # Same two tables as above, now with counts instead of percentages
        print(df_single_negation.groupby('label').agg({'correct': ['sum', 'count']}))
        print(df_single_negation.groupby(['label', 'predicted_label']).agg({'correct': ['sum', 'count']}))

    df_no_negation = df[df['single_negation'] == False]
    if not df_no_negation.empty:
        print("No Negation")
        print(df_no_negation.groupby('label').agg({'correct': ['mean']}))
        print(df_no_negation.groupby(['label', 'predicted_label']).agg({'correct': ['mean']}))
        # Same two tables as above, now with counts instead of percentages
        print(df_no_negation.groupby('label').agg({'correct': ['sum', 'count']}))
        print(df_no_negation.groupby(['label', 'predicted_label']).agg({'correct': ['sum', 'count']}))

    # Table of overall accuracy for each category (with double negation, with single negation, and without negation)
    print("Category Accuracy")
    # Double Negation
    if not df_double_negation.empty:
        print("Double Negation")
        print(df_double_negation['correct'].mean())
    # Single Negation
    if not df_single_negation.empty:
        print("Single Negation")
        print(df_single_negation['correct'].mean())

    # No Negation
    if not df_no_negation.empty:
        print("No Negation")
        print(df_no_negation['correct'].mean())

    # Tabel of overall accuracy for each label, and counts of each label
    print("Overall Accuracy")
    print(df.groupby('label').agg({'correct': ['mean', 'count']}))
    print(df.groupby('predicted_label').agg({'correct': ['mean', 'count']}))
    print(df.groupby(['label', 'predicted_label']).agg({'correct': ['mean', 'count']}))


if __name__ == '__main__':
    main()

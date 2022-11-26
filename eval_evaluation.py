import json
import argparse

# Reads the evaluation file and returns a list of the lines which were predicted incorrectly
# Example Line
# {"premise": "Two women are embracing while holding to go packages.", "hypothesis": "The sisters are hugging goodbye while holding to go packages after just eating lunch.", "label": 1, "predicted_scores": [-2.075383424758911, 4.610177040100098, -2.939692735671997], "predicted_label": 1}
# Want where label != predicted_label
def read_evaluation_file(evaluation_file):
    lines = []
    with open(evaluation_file, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

# Filter the lines to only those which were predicted incorrectly
def filter_incorrect_predictions(lines):
    incorrect_lines = []
    correct_lines = []
    for line in lines:
        if line['label'] != line['predicted_label']:
            incorrect_lines.append(line)
        else:
            correct_lines.append(line)
    return correct_lines, incorrect_lines

# Write the incorrect lines to a file
def write_incorrect_predictions(incorrect_lines, output_file):
    with open(output_file, 'w') as f:
        for line in incorrect_lines:
            f.write(json.dumps(line) + '\n')


def number_containing_not(lines):
    count = 0
    for line in lines:
        if 'not' in line['premise'] or 'not' in line['hypothesis']:
            count += 1
    return count

def number_with_double_negation(lines):
    count = 0
    for line in lines:
        if 'not' in line['premise'] and 'not' in line['hypothesis']:
            count += 1
    return count

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    lines = read_evaluation_file(args.evaluation_file)
    correct_lines, incorrect_lines = filter_incorrect_predictions(lines)

    print('Number of correct predictions: {}'.format(len(correct_lines)))
    print('Number of incorrect predictions: {}'.format(len(incorrect_lines)))
    print('Number of incorrect predictions containing not: {}'.format(number_containing_not(incorrect_lines)))
    print('Number of incorrect predictions with double negation: {}'.format(number_with_double_negation(incorrect_lines)))
    print('Number of correct predictions containing not: {}'.format(number_containing_not(correct_lines)))
    print('Number of correct predictions with double negation: {}'.format(number_with_double_negation(correct_lines)))

    # Same as above but expressed as a percentage
    print('Percentage of incorrect predictions containing not: {}'.format(number_containing_not(incorrect_lines) / len(incorrect_lines)))
    print('Percentage of incorrect predictions with double negation: {}'.format(number_with_double_negation(incorrect_lines) / len(incorrect_lines)))
    print('Percentage of correct predictions containing not: {}'.format(number_containing_not(correct_lines) / len(correct_lines)))
    print('Percentage of correct predictions with double negation: {}'.format(number_with_double_negation(correct_lines) / len(correct_lines)))

    # Total accuracy
    print('Total Accuracy: {}'.format(len(correct_lines) / len(lines)))

    # Accuracy of predictions containing not
    total_containing_not = number_containing_not(incorrect_lines) + number_containing_not(correct_lines)
    print('Accuracy of predictions containing not: {}'.format(number_containing_not(correct_lines) / total_containing_not))

    # Accuracy of predictions with double negation
    total_with_double_negation = number_with_double_negation(incorrect_lines) + number_with_double_negation(correct_lines)
    print('Accuracy of predictions with double negation: {}'.format(number_with_double_negation(correct_lines) / total_with_double_negation))


    write_incorrect_predictions(incorrect_lines, args.output_file)

if __name__ == '__main__':
    main()

import json
import argparse


# Read file from monli in jsonl format
def read_evaluation_file(evaluation_file):
    lines = []
    with open(evaluation_file, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


# Current format example:
# {"sentence1": "the man does not own a dog", "sentence2": "the man does not own a mammal", "gold_label": "neutral", "depth": 4, "sentence1_lex": "dog", "sentence2_lex": "mammal"}
# Switch label to int format with 0 = entailment, 1 = neutral, 2 = contradiction
# Desired format example:
# {"premise": "the man does not own a dog", "hypothesis": "the man does not own a mammal", "label": 1}


def convert_from_monli_format(lines):
    converted_lines = []
    for line in lines:
        converted_line = {}
        converted_line['premise'] = line['sentence1']
        converted_line['hypothesis'] = line['sentence2']
        converted_line['label'] = 0
        if line['gold_label'] == 'neutral':
            converted_line['label'] = 1
        elif line['gold_label'] == 'contradiction':
            converted_line['label'] = 2
        converted_lines.append(converted_line)
    return converted_lines


# Write the converted lines to a file
def write_converted_lines(converted_lines, output_file):
    with open(output_file, 'w') as f:
        for line in converted_lines:
            f.write(json.dumps(line) + '\n')


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    # Write output to file in output_dir with same name as evaluation_file, regardless of what directory it is in
    output_file = args.output_dir + '/' + args.evaluation_file.split('/')[-1]
    lines = read_evaluation_file(args.evaluation_file)
    converted_lines = convert_from_monli_format(lines)
    write_converted_lines(converted_lines, output_file)


if __name__ == '__main__':
    main()

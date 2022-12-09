import json
import argparse

# Read file from monli in jsonl format
# Sample lines:
# {"sentence1": "There is a man not wearing a hat staring at people on a subway.", "sentence2": "There is a man not wearing a sunhat staring at people on a subway.", "gold_label": "entailment", "depth": -1, "sentence1_lex": "hat", "sentence2_lex": "sunhat"}

def read_jsonl_file(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield json.loads(line)


# Desired format of new output:
# {"premise": "the man does not own a dog", "hypothesis": "the man does not own a husky", "label": 0}
# for the label 0 is neutral, 1 is entailment, 2 is contradiction
def generate_new_dataset(in_filename, out_filename):
    # Alternate removing the not and following space from the premise and hypothesis
    # If the gold_label is neutral then removing the not from the premise should result in the contradiction, but removing from the hypothesis should result in neutral
    # If the gold_label is entailment then removing the not from the premise should result in neutral, but removing from the hypothesis should result in contradiction

    with open(out_filename, 'w') as f:
        for line in read_jsonl_file(in_filename):
            if line['gold_label'] == 'neutral':
                premise = line['sentence1']
                modified_premise = premise.replace('not ', '')
                hypothesis = line['sentence2']
                modified_hypothesis = hypothesis.replace('not ', '')
                # first do the contradiction, then the neutral
                f.write(json.dumps({'premise': modified_premise, 'hypothesis': hypothesis, 'label': 2}) + '\n')
                f.write(json.dumps({'premise': premise, 'hypothesis': modified_hypothesis, 'label': 0}) + '\n')
            elif line['gold_label'] == 'entailment':
                premise = line['sentence1']
                modified_premise = premise.replace('not ', '')
                hypothesis = line['sentence2']
                modified_hypothesis = hypothesis.replace('not ', '')
                # first do the neutral, then the contradiction
                f.write(json.dumps({'premise': modified_premise, 'hypothesis': hypothesis, 'label': 0}) + '\n')
                f.write(json.dumps({'premise': premise, 'hypothesis': modified_hypothesis, 'label': 2}) + '\n')
            else:
                raise Exception('Unknown label: {}'.format(line['gold_label']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_f', type=str, required=True)
    parser.add_argument('--out_f', type=str, required=True)
    args = parser.parse_args()
    generate_new_dataset(args.in_f, args.out_f)


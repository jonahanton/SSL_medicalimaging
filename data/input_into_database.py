import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--database-path', '-p', default='./datasets_database.json')
parser.add_argument('--dataset-name', '-d', default='MNIST', help='dataset name')
parser.add_argument('--size', '-s', default=28, type=int, help='size of images')

def main():

    args = parser.parse_args()
    dataset_entry = {args.dataset_name:{'size':args.size}}
    with open(f'{args.database_path}', 'r') as jsonfile:
        database = json.load(jsonfile)
    database.update(dataset_entry)
    with open(f'{args.database_path}', 'w') as jsonfile:
        json.dump(database, jsonfile)


if __name__ == "__main__":
    main()



import csv
import random
from itertools import cycle
import nltk
from nltk.corpus import wordnet as wn
import csv
import random
from itertools import islice
import os

nltk.download('wordnet')
nouns = set(word.name().split('.')[0] for word in wn.all_synsets('n'))
verbs = set(word.name().split('.')[0] for word in wn.all_synsets('v'))
num_records = 5000
correct_words = list(nouns | verbs)

def introduce_errors(word, num_errors=1):
    word = list(word)
    for _ in range(num_errors):
        index = random.randint(0, len(word) - 1)
        word[index] = random.choice('abcdefghijklmnopqrstuvwxyz')
    return ''.join(word)

def make_data():
  dataset = []
  for correct_word in cycle(correct_words):
    if len(correct_word) < 3:  
        continue
    misspelled_word = introduce_errors(correct_word, num_errors=random.randint(1, 3))
    dataset.append((correct_word,misspelled_word))
    if len(dataset) >= num_records:
        break
  return dataset

def create_dataset():
    csv_filename = 'dataset.csv'
    dataset=make_data()
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(['clean', 'distorted'])  
      csvwriter.writerows(dataset)
    print(f'Dataset with {num_records} records saved to {csv_filename}')

def read_csv(filename):
    with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        data = list(csvreader)
    return data

def write_csv(filename, data):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)

def split_dataset():
   original_data = read_csv('dataset.csv')
   random.shuffle(original_data)
   train_size = int(0.7 * len(original_data))
   test_size = int(0.2 * len(original_data))
   dev_size = len(original_data) - train_size - test_size
   train_data = original_data[:train_size]
   test_data = original_data[train_size:train_size + test_size]
   dev_data = original_data[train_size + test_size:]
   write_csv('EnglishDataset\\train.csv', train_data)
   write_csv('EnglishDataset\\test.csv', test_data)
   write_csv('EnglishDataset\\dev.csv', dev_data)
   os.remove('dataset.csv')
   print(f'Dataset split into train.csv ({len(train_data)} records), test.csv ({len(test_data)} records), and dev.csv ({len(dev_data)} records).')
   

if __name__=="__main__":   
   create_dataset()
   split_dataset()

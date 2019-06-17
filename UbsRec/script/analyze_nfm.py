from os import path

import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('result_file', type=str)
  args = parser.parse_args()
  result_file = args.result_file

  with open(result_file, 'r') as fin:
    for line in fin.readlines():
      input(line)

if __name__ == '__main__':
  main()




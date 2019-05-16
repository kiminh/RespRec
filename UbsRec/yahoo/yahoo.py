from os import path

def main():
  data_dir = path.expanduser('~/Downloads/data/Webscope_R3')
  if not path.exists(data_dir):
    raise Exception('Please download the yahoo dataset')

if __name__ == '__main__':
  main()
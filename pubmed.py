#!/usr/bin/python
"""
pubmed.py Implementation of the helper functions to access pubmed

In order to test I ran the algorithm on the article http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0113812
that has been saved inside the file article.txt. In the future we plan to work on a generalized way of 
accessing pubmed article data.

I have already implemented a fetch_ids function that uses the Entrez module from biopython and fetches
at most 100 pubmed ids that correpsond to a certain term. The next step is to actually use beautiful
soup to parse the html of the articles.
"""

from Bio import Entrez
from bs4 import BeautifulSoup
from urllib import urlopen

def fetch_ids(term):
  '''
  Fetches a list of pubmed ids related to term from pubmed

  Args :
    term: to search for
  
  Returns:
    List of pubmed ids of articles related to term
  
  Raises :
      None
  '''
  handle = Entrez.esearch(db="pubmed", term=term, retmax=100)
  record = Entrez.read(handle)
  idlist = record["IdList"]
  return idlist

def get_text(pubmedid):
  url = "http://dx.plos.org/10.1371/journal.pone.0113812"
  html_doc = urlopen(url).read()
  soup = BeautifulSoup(html_doc)
  soup = BeautifulSoup(html_doc)
  print(soup.get_text())


def main():
  ids = fetch_ids('nucleus')
  get_text(ids[0])



if __name__ == "__main__":
    main()




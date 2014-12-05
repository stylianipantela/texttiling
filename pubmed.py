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

from bs4 import BeautifulSoup
from urllib import urlopen
import sys
import string

def get_text(url, outfile): 
    section_breaks = []    
    num_paragraphs = 0
    output_text = ""
    html_doc = urlopen(url).read()
    soup = BeautifulSoup(html_doc)

    for section in soup.find_all("div", {"class":"section"}):
        # non-standard section, skip it        
        if section.find("p").get('class') is not None:
            continue
        
        # record the start of a new section
        if(num_paragraphs > 0):
            section_breaks.append(num_paragraphs)

        # remove figures
        for figure in section.find_all("div", {"class":"figure"}):
            figure.extract()

        # process and write paragraphs
        for paragraph in section.find_all("p"):                
            for link in paragraph.find_all("a"):
                link.extract()
            ptext = filter(lambda x: x in string.printable, paragraph.get_text())
            output_text += ptext + "\n\n"
            num_paragraphs += 1

    # create the metadata about breaks    
    metadata = str(len(section_breaks)) + "\n"
    for brk in section_breaks:  
        metadata += str(brk) + "\n"

    # write to the output file    
    f = open(outfile, 'w')
    f.write(metadata)
    f.write(output_text)
    f.close()

def main(argv):
    # use plos one!
    link = "http://dx.plos.org/10.1371/journal.pone.0113812"
    get_text(link, argv[1])


if __name__ == "__main__":
    main(sys.argv)




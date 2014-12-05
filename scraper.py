#!/usr/bin/python
"""
Accesses articles through plosOne

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

    # the page was bad, didn't get the article
    if len(section_breaks) == 0:
        return False

    # create the metadata about breaks    
    metadata = str(len(section_breaks)) + "\n"
    for brk in section_breaks:  
        metadata += str(brk) + "\n"

    # write to the output file    
    f = open(outfile, 'w')
    f.write(metadata)
    f.write(output_text)
    f.close()
    return True
    
def main(argv):
    # use plos one!
    seed = "0113812"
    i = 0 
    j = 0
    while i < 10:
        article_id = str(int(seed) + j).zfill(len(seed))
        link = "http://dx.plos.org/10.1371/journal.pone." + article_id
        j += 1
        if get_text(link, "articles/article" + str(i).zfill(3) + ".txt"):
            i += 1

if __name__ == "__main__":
    main(sys.argv)




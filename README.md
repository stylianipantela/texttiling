*
* Kevin Mu, Jonathan Miller, Stella Pantela, and Dianna Hu
* CS187 - Computational Lingustics
* Final Project (TextTiling) - Group Implementation
* README
*

---- -- - - - - -  -   -
Setup Instructions
---- -- - - - - -  -   -
0) If nltk is already installed, skip to step 5.

1) Run "python ez_setup.py"
2) Run (sudo) "easy_install pip"
3) Run (sudo) "pip install -U nltk"
4) Run "python", then type "import nltk"
5) Type "nltk.download()". A new window should open,
   showing the nltk Downloader.
6) Click the "corpora" tab.
7) Select "Stopwords Corpus" (stopwords) 
   and "WordNet" (wordnet), and click Download.
8) Close the nltk downloader and exit python.


---- -- - - - - -  -   -
Running Instructions
---- -- - - - - -  -   -
1) cd into the project directory
2) Run: python texttiling.py <outfile>
    a) The scores_outfile is the file where you want the 
       results to be written

    e.g., python texttiling.py outfile.txt


---- -- - - - - -  -   -
Scraping Articles
---- -- - - - - -  -   -
1) If you would like to scrape other articles using scraper.py,
   you can do that by first installing BeautifulSoup.
2) Then change the 'seed' value in the main() function of scraper.py
3) You can also adjust the number of articles you scrape (N).
4) Run: python scraper.py
5) Verify that the articles were correctly scraped and placed in
   the articles folder.

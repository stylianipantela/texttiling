#!/usr/bin/python
"""
texttiling.py Implementation of the TextTiling Algorithm

The implementation is based on the article by Marti A. Hearst
with title TextTiling: Segmenting Text into Multi-Paragraph Subtopic Passages

The algorithm can be broken down into three parts:
1. Tokenization
2. Lexical Score Determination
  Blocks
  Vocabulary Introduction
3. Boundary Identification

We decided to split our work so that we all have an clear idea of how to specifically implement one
of the parts above while looking at the rest as well. I primarily focused on the implementation of the 
Vocabulary Introduction for measuring scores. Notice that the Vocabulary Introduction is one of the 
two alternatives for measuring scores. Also, in order to test I ran the algorithm on the article http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0113812
that has been saved inside the file article.txt. In the future we plan to work on a generalized way of 
accessing pubmed article data. More details can be found in the pubmed.py file.

NOTE: The implementation of tokenization and boundary identification was down by Kevin and Dianna 
respecitively.
"""

from __future__ import division
import re
import sys
from math import sqrt
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

def tokenize_string(input_str, w):
  """
  Tokenize a string using the following four steps:
      1) Turn all text into lowercase and split into tokens by
         removing all punctuation except for apostrophes and internal
         hyphens
      2) Remove common words that don't provide much information about
         the content of the text, called "stop words" 
      3) Reduce each token to its morphological root 
         (e.g. "dancing" -> "dance") using nltk's lemmatize function.
      4) Group the lemmatized tokens into groups of size w, which
         represents the pseudo-sentence size.

  Args :
      input_str : A string to tokenize
      w: pseudo-sentence size
  Returns:
      A tuple (token_sequences, unique_tokens), where:
          token_sequences: A list of token sequences, each w tokens long.
          unique_tokens: A set of all unique words used in the text.
  Raises :
      None
  """

  tokens = []
  for input_substr in input_str.lower().split():
      pat = r"((?:[a-z]+(?:[-'][a-z]+)*))"
      tokens.extend(re.findall(pat, input_substr))
  # remove stop words
  words = [word for word in tokens if word not in stop_words]
  # lemmatize the words
  words = [lemmatizer.lemmatize(word) for word in words]
  unique_tokens = set(words)
  token_sequences = []
  index = 0  
  cnt = Counter() 
  for i in xrange(len(words)):
      cnt[words[i]] += 1
      index += 1
      if index % w == 0:
          token_sequences.append(cnt)
          cnt = Counter()
          index = 0
  return (token_sequences, unique_tokens)


def vocabulary_introduction(token_sequences, w):
  """
  Computes lexical score for the gap between pairs of text blocks 
  Text blocks contain w adjacent sentences and act as moving windows
  Low lexical scores preceded and followed by high lexical scores
  would mean topic shifts

  w could be the average paragraph length for future implementations

  Args:
    w: size of a sequence

  Returns:
    list of scores where scores[i] corresponds to the score at gap position i

  Raises:
    None
  """
  new_words1 = set()
  new_words2 = set(token_sequences[0])
  w2 = w * 2

  # score[i] corresponds to gap position i, score[0] = 1 for the first position
  scores = [1]
  for i in xrange(1,len(token_sequences)-1):
    # new words to the left of the gap
    new_wordsb1 = set(token_sequences[i-1]).difference(new_words1)

    # new words to the right of the gap
    new_wordsb2 = set(token_sequences[i+1]).difference(new_words2)

    # calculate score and update score array
    score = len(new_wordsb1) + len(new_wordsb2) / w2
    scores.append(score)

    # update sets that keep track of new words
    new_words1 = new_words1.union(token_sequences[i-1])
    new_words2 = new_words2.union(token_sequences[i+1])

  # special case on last element
  b1 = len(set(token_sequences[len(token_sequences)-1]).difference(new_words1))
  scores.append(b1/w2)

  return scores

def getDepthCutoff(lexScores, liberal=True):
    """
    Get the cutoff for depth scores, above which gaps are considered boundaries.

    Args:
        lexScores: list of lexical scores for each token-sequence gap
        liberal: True IFF liberal criterion will be used for determining cutoff

    Returns:
        A float representing the depth cutoff score

    Raises:
        None
    """

    mean = np.mean(lexScores)
    stdev = np.std(lexScores)

    return mean - stdev if liberal else mean - stdev / 2

def getDepthSideScore(lexScores, currentGap, left):
    """
    Get the depth score for the specified side of the specified gap

    Args:
        lexScores: list of lexical scores for each token-sequence gap
        currentGap: index of gap for which to get depth side score
        left: True IFF the depth score for left side is desired

    Returns:
        A float representing the depth score for the specified side and gap,
        calculated by finding the "peak" on the side of the gap and returning
        the difference between the lexical scores of the peak and gap.

    Raises:
        None
    """

    depthScore = 0
    i = currentGap

    # continue traversing side while possible to find new peak
    while lexScores[i] - lexScores[currentGap] >= depthScore:
        # update depth score based on new peak
        depthScore = lexScores[i] - lexScores[currentGap]

        # go either left or right depending on specification
        i = i - 1 if left else i + 1

        # do not go beyond bounds of gap!
        if (i < 0 and left) or (i == len(lexScores) and not left):
            break

    return depthScore

def getGapBoundaries(lexScores):
    """
    Get the gaps to be considered as boundaries based on gap lexical scores.

    Args:
        lexScores: list of lexical scores for each token-sequence gap

    Returns:
        A list of gaps (identified by index) that are considered boundaries.

    Raises:
        None
    """

    boundaries = []
    cutoff = getDepthCutoff(lexScores)

    for i, score in enumerate(lexScores):
        # find maximum depth to left and right
        depthLeftScore = getDepthSideScore(lexScores, i, True)
        depthRightScore = getDepthSideScore(lexScores, i, False)

        # add gap to boundaries if depth score beyond threshold
        depthScore = depthLeftScore + depthRightScore
        if depthScore >= cutoff:
            boundaries.append(i)

    return boundaries

def getBoundaries(lexScores, pLocs, w):
    """
    Get locations of paragraphs where subtopic boundaries occur

    Args:
        lexScores: list of lexical scores for each token-sequence gap
        pLocs: list of token indices such that paragraph breaks occur after them
        w: number of tokens to be grouped into each token-sequence

    Returns:
        A sorted list of unique paragraph locations (measured in terms of token
        indices) after which a subtopic boundary occurs.

    Raises:
        None
    """

    # convert boundaries from gap indices to token indices
    gapBoundaries = getGapBoundaries(lexScores)
    tokBoundaries = [w * (gap + 1) for gap in gapBoundaries]

    # do not allow duplicates of boundaries
    parBoundaries = set()

    # convert raw token boundary index to closest index where paragraph occurs
    for i in xrange(len(tokBoundaries)):
        parBoundaries.add(min(pLocs, key=lambda b: abs(b - tokBoundaries[i])))

    return sorted(list(parBoundaries))


def main():

  # number of words in a sequence
  w = 35

  # test article from pubmed, can be found here
  # http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0113812
  with open("article.txt", "r") as f:
    lines = ""
    for line in f.readlines():
      lines = lines + line
    f.close()

    # tokenize string
    sequences, _ = tokenize_string(lines, w)

    # obtain scores
    scores = vocabulary_introduction(sequences, w)

    # obtain boundaries
    getBoundaries(scores,[],w)


if __name__ == "__main__":
  main()


#!/usr/bin/python

# *****************************************************************************
#
# Authors: Dianna Hu, Stella Pantela, Jonathan Miller, and Kevin Mu
# Class: Computer Science 187
# Date: December 1, 2014
# Final Project - Implementation of the TextTiling Algorithm
#
# Description: This implementation is based on the article by Marti A. Hearst,
# "TextTiling: Segmenting Text into Multi-Paragraph Subtopic Passages".
#
# The algorithm can be broken down into three parts:
# 1. Tokenization
# 2. Lexical Score Determination
#     a) Blocks
#     b) Vocabulary Introduction
# 3. Boundary Identification
#
# Before running, please make sure that nltk is installed and that you download
# the wordnet and stoplist corpuses. See README for instructions.
#
# *****************************************************************************

from __future__ import division
import re
import sys
import numpy as np
from math import sqrt
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

def tokenize_string(input_str, w):
    '''
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
        A tuple (token_sequences, unique_tokens, paragraph_breaks), where:
            token_sequences: A list of token sequences, each w tokens long.
            unique_tokens: A set of all unique words used in the text.
            paragraph_breaks: A list of indices such that paragraph breaks
                              occur immediately after each index.
    Raises :
        None
    '''
    paragraphs = [s.strip() for s in input_str.splitlines()]
    paragraphs = [s for s in paragraphs if s != ""]
    tokens = []
    paragraph_breaks = []
    token_count = 0
    pat = r"((?:[a-z]+(?:[-'][a-z]+)*))"
    for paragraph in paragraphs:
        pgrph_tokens = re.findall(pat, paragraph)
        tokens.extend(pgrph_tokens)
        token_count += len(pgrph_tokens)
        paragraph_breaks.append(token_count)
    paragraph_breaks = paragraph_breaks[:-1]

    token_sequences = []
    index = 0  
    cnt = Counter() 
    # split tokens into groups of size w
    for i in xrange(len(tokens)):
        cnt[tokens[i]] += 1
        index += 1
        if index % w == 0:
            token_sequences.append(cnt)
            cnt = Counter()
            index = 0

    # remove stop words from each sequence
    for i in xrange(len(token_sequences)):
        token_sequences[i] = [lemmatizer.lemmatize(word) for word in token_sequences[i] if word not in stop_words]
    # lemmatize the words in each sequence
    for i in xrange(len(token_sequences)):
        token_sequences[i] = [lemmatizer.lemmatize(word) for word in token_sequences[i]]
    # get unique tokens
    unique_tokens = [word for word in set(tokens) if word not in stop_words] 

    return (token_sequences, unique_tokens, paragraph_breaks)

def block_score(k, token_seq_ls, unique_tokens):
    """
    Provide similarity scores for adjacent blocks of token sequences.
    Args:
        k: the block size, as defined in the paper (Hearst 1997) 
        token_seq_ls: list of token sequences, each of the same length
        unique_tokens: A set of all unique words used in the text.
    Returns:
        list of block scores from gap k through gap (len(token_seq_ls)-k-2),
        inclusive.
    Raises:
        None.
    """
    score_ls = []
    # calculate score for each gap with at least k token sequences on each side
    for gap_index in range(1, len(token_seq_ls)):
        current_k = min(gap_index, k, len(token_seq_ls) - gap_index - 1)
        before_block = token_seq_ls[gap_index - current_k : gap_index]
        after_block = token_seq_ls[gap_index : gap_index + current_k]
        
        before_cnt = Counter()
        after_cnt = Counter()
        for j in xrange(current_k+1):
            before_cnt += Counter(token_seq_ls[gap_index + j - current_k])
            after_cnt += Counter(token_seq_ls[gap_index + j])
        
        # calculate and store score
        numerator = 0.0
        before_sq_sum = 0.0
        after_sq_sum = 0.0
        for token in unique_tokens:
            numerator += (before_cnt[token] * after_cnt[token])
            before_sq_sum += (before_cnt[token] ** 2)
            after_sq_sum += (after_cnt[token] ** 2)
        denominator = sqrt(before_sq_sum * after_sq_sum)
        score_ls.append(numerator / denominator)
    return score_ls

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

def writeTextTiles(boundaries, pLocs, inputText, outfile):
    """
    Get TextTiles in the input text based on paragraph locations and boundaries.

    Args:
        boundaries: list of paragraph locations where subtopic boundaries occur
        pLocs: list of token indices such that paragraph breaks occur after them
        inputText: a string of the initial (unsanitized) text

    Returns:
        A list of lists T such that T[i] is a list of paragraphs that occur
        within one subtopic.

    Raises:
        None
    """

    textTiles = []
    paragraphs = [s.strip() for s in inputText.splitlines()]
    paragraphs = [s for s in paragraphs if s != ""]

    print len(pLocs)
    assert len(paragraphs) == len(pLocs) + 1
    splitIndices = [pLocs.index(b) for b in boundaries]

    # get precision and recall
    precision_recall([], splitIndices)

    startIndex = 0

    # append section between subtopic boundaries as new TextTile
    for i in splitIndices:
        textTiles.append(paragraphs[startIndex:i + 1])
        startIndex = i + 1
    # tack on remaining paragraphs in last subtopic
    textTiles.append(paragraphs[startIndex:])
    
    f = open(outfile, 'w')
    for i, textTile in enumerate(textTiles):
        f.write('SUB-TOPIC' + str(i) + '\n')
        f.write('----------\n\n')
        for paragraph in textTile:
            f.write(paragraph + '\n\n')

def precision_recall(original_breaks, new_breaks):
    # assumes input has the topic changes
    original_breaks = [0,3,4,5,6,7,8,9,12,15,16]

    new_breaks_set = set(new_breaks)
    original_breaks_set = set(original_breaks)

    precision = len(new_breaks_set.intersection(original_breaks_set)) / len(new_breaks_set)

    recall = len(new_breaks_set.intersection(original_breaks_set)) / len(original_breaks)

    print "Precision is " + str(precision)
    print "Recall is " + str(recall)




def main(argv):
    '''
    Tokenize a file and compute gap scores using the algorithm described
    in Hearst's TextTiling paper.

    Args :
        argv[1] : The name of the file to analyze
    Returns:
        None
    Raises :
        None
    '''
    if (len(argv) != 3):
        print("\nUsage: python texttiling.py <infile> <outfile> \n")
        sys.exit(0)

    with open(argv[1], 'r') as f:
        # somewhat arbitrarily chosen constants for pseudo-sentence size
        # and block size, respectively.
        w = 16
        k = 10
        text = f.read()
        token_sequences, unique_tokens, paragraph_breaks = tokenize_string(text, w)
        scores1 = block_score(k, token_sequences, unique_tokens)
        scores2 = vocabulary_introduction(token_sequences, w)
        boundaries1 = getBoundaries(scores1, paragraph_breaks, w)
        boundaries2 = getBoundaries(scores2, paragraph_breaks, w)
        writeTextTiles(boundaries1, paragraph_breaks, text, argv[2])
        writeTextTiles(boundaries2, paragraph_breaks, text, argv[2])

if __name__ == "__main__":
  main(sys.argv)


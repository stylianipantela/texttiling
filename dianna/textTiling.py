#!/usr/bin/python -tt

"""
Implements (a very rough first draft of) the TextTiling algorithm

Dianna Hu
CS 187, Fall 2014
"""

from collections import Counter
import numpy as np
import nltk

W = 4
K = 2
TEST_STOPWORDS = ['x']

# roughly replicates Figure 3 in paper (paragraph is \n\n, X is stop word)
TEST_TEXT = 'A B C D\n\nA C E X\n\nB C E X A D E X\n\nE F G H\n\nB F H X B F G H G H I X'

def tokenize(inputText, w):
    """
    Tokenize the input text.

    Args:
        inputText: string of text to be tokenized, with all markup stripped
        w: number of tokens to be grouped into each token-sequence

    Returns:
        A tuple (S, P) where S is a list of lists such that S[i][j] denotes the
        frequency of token j appearing in token-sequence i (a token-sequence has
        tokens), and P is a list such that a paragraph break occurs in the
        original text after P[i] tokens (relative to the beginning of the text).
        Note that stop words are considered in determining the length but not
        the vocabulary of the token-sequence.

    Raises:
        None
    """

    # needs morphological root normalization

    pLocs = []
    tokenSeqs = []
    tokens = nltk.word_tokenize(inputText).lower()

    # record paragraph break locations
    paragraphs = inputText.split('\n\n')
    for i, paragraph in enumerate(paragraphs):
        location = len(nltk.word_tokenize(paragraph))
        if i > 0:
            location += pLocs[i - 1]
        if i < len(paragraphs) - 1:
            pLocs.append(location)

    # create count information
    counts = Counter(tokens)
    for stopword in TEST_STOPWORDS:
        del counts[stopword]
    uniqueTokens = list(counts.keys())

    # create token sequences from count information
    seqStart = 0
    while seqStart < len(tokens):
        # get counts for specified sequence
        seqCounts = Counter(tokens[seqStart:seqStart + w])
        for stopword in TEST_STOPWORDS:
            del counts[stopword]

        # convert to list with same token order and allowing 0 token count
        tokenSeq = []
        for token in uniqueTokens:
            tokenSeq.append(seqCounts[token])
        tokenSeqs.append(tokenSeq)

        seqStart += w

    return tokenSeqs, pLocs

def getLexScores(tokenSeqs, k):
    """
    Get scores for gaps between k-sized blocks of token-sequences.

    Args:
        tokenSeqs: list of token-sequences, where each token-sequence is a
            list of corresponding token frequencies
        k: block size, i.e., number of token-sequences in each block

    Returns:
        A list of gap scores between each block of k token-sequences. Note that
        the length of this list is 1 less than the length of tokenSeqs. Uses the
        block comparison approach.

    Raises:
        None
    """

    # needs other scoring methods (vocabulary introduction, lexical chains)

    scores = []
    numGaps = len(tokenSeqs) - 1

    for gapId in xrange(numGaps):
        # determine effective block size: usually k, but smaller at edges
        blockSize = k
        if gapId < k:
            blockSize = gapId + 1
        elif gapId > numGaps - k:
            blockSize = numGaps - gapId

        # consolidate token sequence counts within compared block
        left = gapId - blockSize + 1
        middle = left + blockSize
        right = middle + blockSize

        leftBlock = np.array(tokenSeqs[left:middle]).sum(axis=0)
        rightBlock = np.array(tokenSeqs[middle:right]).sum(axis=0)

        # calculate score for gap based on equation
        score = float (np.inner(leftBlock, rightBlock))
        score /= np.linalg.norm(leftBlock) * np.linalg.norm(rightBlock)

        scores.append(score)

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
        if depthLeftScore + depthRightScore >= cutoff:
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

    # needs smoothing, incorporation of local lexical distributions, prevention
    # of very close boundary assignment

    # convert boundaries from gap indices to token indices
    gapBoundaries = getGapBoundaries(lexScores)
    tokBoundaries = [w * (gap + 1) for gap in gapBoundaries]

    # do not allow duplicates of boundaries
    parBoundaries = set()

    # convert raw token boundary index to closest index where paragraph occurs
    for i in xrange(len(tokBoundaries)):
        parBoundaries.add(min(pLocs, key=lambda b: abs(b - tokBoundaries[i])))

    return sorted(list(parBoundaries))

def getText():
    """
    Get the text on which to run the TextTiling algorithm.

    Args:
        None

    Returns:
        A string to be used for TextTiling

    Raises:
        None

    """

    # needs actual text from an actual article (STUB)
    return TEST_TEXT

def sanitizeText(inputText):
    """
    Sanitize the input text by stripping markup and correcting paragraph format.

    Args:
        inputText: a string of the initial (unsanitized) text

    Returns:
        A string of text sans markup whose paragraph breaks are denoted by \n\n

    Raises:
        None
    """

    # needs actual sanitization (STUB)
    return inputText

def getTextTiles(boundaries, pLocs, inputText):
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
    paragraphs = inputText.split('\n\n')
    assert len(paragraphs) == len(pLocs) + 1

    splitIndices = [pLocs.index(b) for b in boundaries]
    startIndex = 0

    # append section between subtopic boundaries as new TextTile
    for i in splitIndices:
        textTiles.append(paragraphs[startIndex:i + 1])
        startIndex = i + 1

    # tack on remaining paragraphs in last subtopic
    textTiles.append(paragraphs[startIndex:])

    return textTiles

def displayTextTiles(inputText, textTiles):
    """
    Display the TextTiles given by the TextTiling algorithm.

    Args:
        inputText: string of sanitized text on which to run TextTiling
        textTiles: list of subtopics, where a subtopic is a list of paragraphs

    Returns:
        None

    Raises:
        None
    """

    print 'INPUT TEXT:\n'
    print inputText

    print '\nTEXTTILED TEXT:\n'
    for i, textTile in enumerate(textTiles):
        print 'SUBTOPIC', i
        print '----------'
        for paragraph in textTile:
            print paragraph + '\n'

def textTiling(inputText):
    """
    Perform the TextTiling algorithm on the input text.

    Args:
        inputText: string of sanitized text on which to run TextTiling

    Returns:
        A list of lists T such that T[i] is a list of paragraphs that occur
        within one subtopic.

    Raises:
        None
    """

    tokens, pLocs = tokenize(inputText, W)
    scores = getLexScores(tokens, K)
    boundaries = getBoundaries(scores, pLocs, W)
    return getTextTiles(boundaries, pLocs, inputText)

def main():
    """
    Perform the TextTiling algorithm and display the results

    Args:
        None

    Returns:
        None

    Raises:
        None
    """

    inputText = getText()
    sanitizedText = sanitizeText(inputText)
    textTiles = textTiling(sanitizedText)
    displayTextTiles(inputText, textTiles)

if __name__ == '__main__':
    main()
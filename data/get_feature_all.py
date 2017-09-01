import sys
import math
reload(sys)
sys.setdefaultencoding('utf-8')


def fuzzyMatch(w1,w2):
    num_matching_chars = len(set(w1)&set(w2))
    score = 2.0*num_matching_chars/(len(w1)+len(w2))
    return score
def getfeatures():

    for line in sys.stdin:
        words = []
        posTags = []
        sentenceOftags = []
        if len(line.strip()) < 1:
            continue
        line = line.strip().split(' ')
        for i in range(0,len(line)):
            el = line[i].strip().split('/')
            words.append(el[0])
            posTags.append(el[1])
            sentenceOftags.append(el[2])

        MAX_SEARCH_DISTANCE = 15
        FUZZY_MATCH_THRESHOLD = 0.8
        feats = ""
        for i in range(0,len(sentenceOftags)):
            feats = feats + words[i] + ' ' + posTags[i] + ' ' + posTags[i] + ' ' + sentenceOftags[i]
            currentInd = i
            currentPosTag = posTags[currentInd]
            word = words[currentInd]
            for j in range(MAX_SEARCH_DISTANCE,0,-1):
                if currentInd + j < len(words):
                    if words[currentInd + j] == word:
                        feats = feats + ' 1'
                    else:
                        feats = feats + ' 0'
                    if posTags[currentInd + j] == currentPosTag:
                        feats = feats + ' 1'
                    else:
                        feats = feats + ' 0'
                else:
                    feats = feats + ' 0'
                    feats = feats + ' 0'
                        
                

            for j in range(MAX_SEARCH_DISTANCE,0,-1):
                if currentInd - j > 0:
                    if words[currentInd - j] == word:
                        feats = feats + ' 1'
                    else:
                        feats = feats + ' 0'
                    if posTags[currentInd - j] == currentPosTag:
                        feats = feats + ' 1'
                    else:
                        feats = feats + ' 0'
                else:
                    feats = feats + ' 0'
                    feats = feats + ' 0'
            



            prev4Word = ''
            prev3Word = ''
            prev2Word = ''
            prevWord = ''
            nowWord = ''
            nextWord = ''
            next2Word = ''
            next3Word = ''
            next4Word = ''
            next5Word = ''
            nowWord = words[i]
            if (i-4) < 0:
                prev4Word = '#start#'
            else:
                prev4Word = words[i-4]
            if (i-3) < 0:
                prev3Word = '#start#'
            else:
                prev3Word = words[i-3]
            if (i-2) < 0:
                prev2Word = '#start#'
            else:
                prev2Word = words[i-2]
            if (i-1) < 0:
                prevWord = '#start#'
            else:
                prevWord = words[i-1]
            if (i+1) >= len(words):
                nextWord = '#end#'
            else:
                nextWord = words[i+1]
            if (i+2) >= len(words):
                next2Word = '#end#'
            else:
                next2Word = words[i+2]
            if (i+3) >= len(words):
                next3Word = '#end#'
            else:
                next3Word = words[i+3]
            if (i+4) >= len(words):
                next4Word = '#end#'
            else:
                next4Word = words[i+4]
            if (i+5) >= len(words):
                next5Word = '#end#'
            else:
                next5Word = words[i+5]

            prev4Pos = ''
            prev3Pos = ''
            prev2Pos = ''
            prevPos = ''
            nowPos = ''
            nextPos = ''
            next2Pos = ''
            next3Pos = ''
            next4Pos = ''
            next5Pos = ''
            nowPos = posTags[i]
            if (i-4) < 0:
                prev4Pos = '#start#'
            else:
                prev4Pos = posTags[i-4]
            if (i-3) < 0:
                prev3Pos = '#start#'
            else:
                prev3Pos = posTags[i-3]
            if (i-2) < 0:
                prev2Pos = '#start#'
            else:
                prev2Pos = posTags[i-2]
            if (i-1) < 0:
                prevPos = '#start#'
            else:
                prevPos = posTags[i-1]
            if (i+1) >= len(posTags):
                nextPos = '#end#'
            else:
                nextPos = posTags[i+1]
            if (i+2) >= len(posTags):
                next2Pos = '#end#'
            else:
                next2Pos = posTags[i+2]
            if (i+3) >= len(posTags):
                next3Pos = '#end#'
            else:
                next3Pos = posTags[i+3]
            if (i+4) >= len(posTags):
                next4Pos = '#end#'
            else:
                next4Pos = posTags[i+4]
            if (i+5) >= len(posTags):
                next5Pos = '#end#'
            else:
                next5Pos = posTags[i+5]



            prev_val = fuzzyMatch(nowWord,prevWord)
            if prev_val >  FUZZY_MATCH_THRESHOLD:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'

            next_val = fuzzyMatch(nowWord,nextWord)
            if next_val >  FUZZY_MATCH_THRESHOLD:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'





            word_Dumple = nowWord + "#" + nextWord
            prev4WordDumple = prev4Word + "#" + prev3Word
            prev3WordDumple = prev3Word + "#" + prev2Word
            prev2WordDumple = prev2Word + "#" + prevWord
            prevWordDumple = prevWord + "#" + nowWord
            nextWordDumple = nextWord + "#" + next2Word
            next2WordDumple = next2Word + "#" + next3Word
            next3WordDumple = next3Word + "#" + next4Word
            next4WordDumple = next4Word + "#" + next5Word
            if prev4WordDumple == word_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'
            if prev3WordDumple == word_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'
            if prev2WordDumple == word_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'
            if prevWordDumple == word_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'

            if nextWordDumple == word_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'
            if next2WordDumple == word_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'
            if next3WordDumple == word_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'
            if next4WordDumple == word_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'

            pos_Dumple = nowPos + "#" + nextPos
            prev4PosDumple = prev4Pos + "#" + prev3Pos
            prev3PosDumple = prev3Pos + "#" + prev2Pos
            prev2PosDumple = prev2Pos + "#" + prevPos
            prevPosDumple = prevPos + "#" + nowPos
            nextPosDumple = nextPos + "#" + next2Pos
            next2PosDumple = next2Pos + "#" + next3Pos
            next3PosDumple = next3Pos + "#" + next4Pos
            next4PosDumple = next4Pos + "#" + next5Pos
            if prev4PosDumple == pos_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'
            if prev3PosDumple == pos_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'
            if prev2PosDumple == pos_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'
            if prevPosDumple == pos_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'

            if nextPosDumple == pos_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'
            if next2PosDumple == pos_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'
            if next3PosDumple == pos_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'
            if next4PosDumple == pos_Dumple:
                feats = feats + ' 1'
            else:
                feats = feats + ' 0'

            print feats.strip()
            feats = ''
        print ''












if __name__ == "__main__":
    getfeatures()

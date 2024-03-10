import unicodedata

###############################################################
################### SHARED HELPER FUNCTIONS ###################
###############################################################

"""Description: Returns most common tuples of encoded integers and their frequency of occurence"""
def countCommonEncodedTuples(encodedInts: list[int], freqDict: dict[tuple[int, int], int] = None) -> None:
    freqDict = {} if freqDict is None else freqDict
    for i in range(len(encodedInts) - 1):
        pair = (encodedInts[i], encodedInts[i + 1])
        freqDict[pair] = freqDict.get(pair, 0) + 1 
    # {} [K, V] -> [(encodedInt1, encodedInt2), frequency]
    return freqDict

"""Description: Removes a tuple of integers from a list of integers"""
def merge(pair, ids: list[int], idx: int) -> list[int]:
    mergedIDs, i = [], 0 

    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            mergedIDs.append(idx)
            i += 2 
        else:
            mergedIDs.append(ids[i])
            i += 1 
    return mergedIDs

def replaceControlCharacters(s: str) -> str:
    # Dont print control characters which distort the output (e.g. \n or much worse)
    # Refs: https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117 & 
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = [ch if unicodedata.category(ch)[0] != "C" else f"\\u{ord(ch):04x}" for ch in s]        
    return "".join(chars)
    
def renderToken(t: bytes) -> str:
    # Pretty print a token, escaping control characters
    return replaceControlCharacters(t.decode('utf-8', errors='replace'))

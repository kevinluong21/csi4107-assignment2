import pandas as pd
import nltk
import string
import re
from spellchecker import SpellChecker
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

additional_stop_words = {
    "a", "about", "above", "ac", "according", "accordingly", "across", "actually", "ad", "adj", 
    "af", "after", "afterwards", "again", "against", "al", "albeit", "all", "almost", "alone", "along", 
    "already", "als", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", 
    "anybody", "anyhow", "anyone", "anything", "anyway", "anywhere", "apart", "apparently", "are", "aren", 
    "arise", "around", "as", "aside", "at", "au", "auf", "aus", "aux", "av", "avec", "away", "b", "be", "became", 
    "because", "become", "becomes", "becoming", "been", "before", "beforehand", "began", "begin", "beginning", 
    "begins", "behind", "bei", "being", "below", "beside", "besides", "best", "better", "between", "beyond", 
    "billion", "both", "briefly", "but", "by", "c", "came", "can", "cannot", "canst", "caption", "captions", 
    "certain", "certainly", "cf", "choose", "chooses", "choosing", "chose", "chosen", "clear", "clearly", "co", 
    "come", "comes", "con", "contrariwise", "cos", "could", "couldn", "cu", "d", "da", "dans", "das", "day", "de", 
    "degli", "dei", "del", "della", "delle", "dem", "den", "der", "deren", "des", "di", "did", "didn", "die", 
    "different", "din", "do", "does", "doesn", "doing", "don", "done", "dos", "dost", "double", "down", "du", 
    "dual", "due", "durch", "during", "e", "each", "ed", "eg", "eight", "eighty", "either", "el", "else", 
    "elsewhere", "em", "en", "end", "ended", "ending", "ends", "enough", "es", "especially", "et", "etc", "even", 
    "ever", "every", "everybody", "everyone", "everything", "everywhere", "except", "excepts", "excepted", 
    "excepting", "exception", "exclude", "excluded", "excludes", "excluding", "exclusive", "f", "fact", "facts", 
    "far", "farther", "farthest", "few", "ff", "fifty", "finally", "first", "five", "foer", "follow", "followed", 
    "follows", "following", "for", "former", "formerly", "forth", "forty", "forward", "found", "four", "fra", 
    "frequently", "from", "front", "fuer", "further", "furthermore", "furthest", "g", "gave", "general", "generally", 
    "get", "gets", "getting", "give", "given", "gives", "giving", "go", "going", "gone", "good", "got", "great", 
    "greater", "h", "had", "haedly", "half", "halves", "hardly", "has", "hasn", "hast", "hath", "have", "haven", 
    "having", "he", "hence", "henceforth", "her", "here", "hereabouts", "hereafter", "hereby", "herein", "hereto", 
    "hereupon", "hers", "herself", "het", "high", "higher", "highest", "him", "himself", "hindmost", "his", "hither", 
    "how", "however", "howsoever", "hundred", "hundreds", "i", "ie", "if", "ihre", "ii", "im", "immediately", 
    "important", "in", "inasmuch", "inc", "include", "included", "includes", "including", "indeed", "indoors", 
    "inside", "insomuch", "instead", "into", "inward", "is", "isn", "it", "its", "itself", "j", "ja", "journal", 
    "journals", "just", "k", "kai", "keep", "keeping", "kept", "kg", "kind", "kinds", "km", "l", "la", "large", 
    "largely", "larger", "largest", "las", "last", "later", "latter", "latterly", "le", "least", "les", "less", 
    "lest", "let", "like", "likely", "little", "ll", "long", "longer", "los", "low", "lower", "lowest", "ltd", "m", 
    "made", "mainly", "make", "makes", "making", "many", "may", "maybe", "me", "meantime", "meanwhile", "med", 
    "might", "million", "mine", "miss", "mit", "more", "moreover", "most", "mostly", "mr", "mrs", "ms", "much", 
    "mug", "must", "my", "myself", "n", "na", "nach", "namely", "nas", "near", "nearly", "necessarily", "necessary", 
    "need", "needs", "needed", "needing", "neither", "nel", "nella", "never", "nevertheless", "new", "next", "nine", 
    "ninety", "no", "nobody", "none", "nonetheless", "noone", "nope", "nor", "nos", "not", "note", "noted", 
    "notes", "noting", "nothing", "notwithstanding", "now", "nowadays", "nowhere", "o", "obtain", "obtained", 
    "obtaining", "obtains", "och", "of", "off", "often", "og", "ohne", "ok", "old", "om", "on", "once", "onceone", 
    "one", "only", "onto", "or", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", 
    "out", "outside", "over", "overall", "owing", "own", "p", "par", "para", "particular", "particularly", "past", 
    "per", "perhaps", "please", "plenty", "plus", "por", "possible", "possibly", "pour", "poured", "pouring", 
    "pours", "predominantly", "previously", "pro", "probably", "prompt", "promptly", "provide", "provides", 
    "provided", "providing", "q", "quite", "r", "rather", "re", "ready", "really", "recent", "recently", "regardless", 
    "relatively", "respectively", "round", "s", "said", "same", "sang", "save", "saw", "say", "second", "see", 
    "seeing", "seem", "seemed", "seeming", "seems", "seen", "sees", "seldom", "self", "selves", "send", "sending", 
    "sends", "sent", "ses", "seven", "seventy", "several", "shall", "shalt", "she", "short", "should", "shouldn", 
    "show", "showed", "showing", "shown", "shows", "si", "sideways", "significant", "similar", "similarly", 
    "simple", "simply", "since", "sing", "single", "six", "sixty", "sleep", "sleeping", "sleeps", "slept", "slew", 
    "slightly", "small", "smote", "so", "sobre", "some", "somebody", "somehow", "someone", "something", "sometime", 
    "sometimes", "somewhat", "somewhere", "soon", "spake", "spat", "speek", "speeks", "spit", "spits", "spitting", 
    "spoke", "spoken", "sprang", "sprung", "staves", "still", "stop", "strongly", "substantially", "successfully", 
    "such", "sui", "sulla", "sung", "supposing", "sur", "t", "take", "taken", "takes", "taking", "te", "ten", "tes", 
    "than", "that", "the", "thee", "their", "theirs", "them", "themselves", "then", "thence", "thenceforth", "there", 
    "thereabout", "thereabouts", "thereafter", "thereby", "therefor", "therefore", "therein", "thereof", "thereon", 
    "thereto", "thereupon", "these", "they", "thing", "things", "third", "thirty", "this", "those", "thou", "though", 
    "thousand", "thousands", "three", "thrice", "through", "throughout", "thru", "thus", "thy", "thyself", "til", 
    "till", "time", "times", "tis", "to", "together", "too", "tot", "tou", "toward", "towards", "trillion", "trillions", 
    "twenty", "two", "u", "ueber", "ugh", "uit", "un", "unable", "und", "under", "underneath", "unless", "unlike", 
    "unlikely", "until", "up", "upon", "upward", "us", "use", "used", "useful", "usefully", "user", "users", "uses", 
    "using", "usually", "v", "van", "various", "ve", "very", "via", "vom", "von", "voor", "vs", "w", "want", "was", 
    "wasn", "way", "ways", "we", "week", "weeks", "well", "went", "were", "weren", "what", "whatever", "whatsoever", 
    "when", "whence", "whenever", "whensoever", "where", "whereabouts", "whereafter", "whereas", "whereat", 
    "whereby", "wherefore", "wherefrom", "wherein", "whereinto", "whereof", "whereon", "wheresoever", "whereto", 
    "whereunto", "whereupon", "wherever", "wherewith", "whether", "whew", "which", "whichever", "whichsoever", 
    "while", "whilst", "whither", "who", "whoever", "whole", "whom", "whomever", "whomsoever", "whose", 
    "whosoever", "why", "wide", "widely", "will", "wilt", "with", "within", "without", "won", "worse", "worst", 
    "would", "wouldn", "wow", "x", "xauthor", "xcal", "xnote", "xother", "xsubj", "y", "ye", "year", "yes", "yet", 
    "yipee", "you", "your", "yours", "yourself", "yourselves", "yu", "z", "za", "ze", "zu", "zum"
}

# Merge the 2 lists together to create a larger stop word corpus
stop_words = stop_words.union(additional_stop_words)

# Used to split text into words without separating the punctuation (terms with hyphens remain intact because of words like "pre-diabetes" and "body-mass"). The punctuation is also maintained for each word in the case there is no space between sentences which results in cases like "I like to read.I like hats." where "read" and "I" should be 2 separate words.
word_splitter = RegexpTokenizer(pattern=r"\w+[-]\w+|\w+['.,!?]*|\w+|\S+")

# To check if a string is valid word (when splitting hyphenated words), use a spellchecker
spell = SpellChecker()

# Used to lemmatize (a form of stemming) words (better performance than the Porter and Lancaster stemmer)
lemmatizer = WordNetLemmatizer()

def is_hyphenated_compound_word(word:str) -> bool:
    '''
    Returns True if the strings in a hyphenated word, when split by its hyphens, are all actual words (a compound word). Otherwise, returns False.

    e.g. "pre-diabetes" is False, "hello" is False (does not contain a hyphen), and "body-mass" is True

    Parameters:
        word (str): Hyphenated word to test
    Returns:
        True or False
    '''
    # For hyphenated words, check if they are the combination of multiple "real" words together or just a prefix and/or suffix. If they are a combination of "real" words, split them into multiple words. Otherwise, remove the hyphen and make them into a single word.
    if "-" in word:
        terms = word.split("-")
        check_terms = [True if term in spell else False for term in terms]
        if all(check_terms):
            return True
    return False

def extract_index_terms(text:str) -> str:
    '''
    Given a string, extracts all of the relevant index terms along with their term frequencies, ignoring numbers, punctuation, and stopwords.

    Parameters:
        text (str): String to extract index terms
    Returns:
        index_terms (dict): A dictionary containing index terms as keys and its term frequency within the document as values.
    '''

    if not text:
        return {}
    
    text = text.lower().strip()
    # If there are any unicode characters in the text, decode them into their proper representations
    text = text.encode('unicode_escape').decode('unicode_escape')
    # Splits the given text into words
    words = word_splitter.tokenize(text)
    # Remove all non-letters from the string entirely (maintain hyphens)
    words = [re.sub(pattern=r'[^\x61-\x7A-]', string=word, repl="") for word in words]
    # If a hyphenated word is a compound word, split the word. If not, remove the hyphen.
    words = [word.split("-") if is_hyphenated_compound_word(word) else [word.replace("-", "")] for word in words]
    words = [word for list_of_words in words for word in list_of_words] # Flatten the list of lists into a single list
    # Remove any empty strings and stopwords
    words = [word for word in words if word and word not in stop_words]
    # Lemmatize each word
    words = [lemmatizer.lemmatize(word) for word in words]

    # Concatenate all words into a string
    return " ".join(words)
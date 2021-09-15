import re

def drop_kanji(sentence):
    words = sentence.split(' ')
    replaced_words = ['' if re.search('[一-鿐]', w) else w for w in words]
    return ' '.join(replaced_words)

def drop_kana(sentence):
    words = sentence.split(' ')
    replaced_words = ['' if re.search('[ｦ-ﾝ]', w) else w for w in words]
    return ' '.join(replaced_words)

def convert_num_to_zero(sentence):
    words = sentence.split(' ')
    replaced_words = [re.sub('[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', '0', w) for w in words]
    replaced_words = [re.sub('0+', '0', w) for w in replaced_words]
    return ' '.join(replaced_words)
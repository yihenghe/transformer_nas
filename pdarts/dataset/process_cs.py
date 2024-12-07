import re
import glob
import gzip
import os
from tqdm import tqdm

def add_newline(sentence):
    if not sentence.endswith('\n'):
        sentence += '\n'
    return sentence
    
def process_czeng():
    # download http://ufal.mff.cuni.cz/czeng
    filter_path = './convert_czeng16_to_17.pl'
    re_block = re.compile(r"^[^-]+-b(\d+)-\d\d[tde]")
    with open(filter_path, encoding="utf-8") as f:
        bad_blocks = {blk for blk in re.search(r"qw{([\s\d]*)}", f.read()).groups()[0].split()}
    
    path = './data.plaintext-format/*train.gz'
    cs_path = './czeng.cs-en.cs'
    en_path = './czeng.cs-en.en'
    with open(cs_path, mode='w', encoding='utf-8') as cs_f, open(en_path, mode='w', encoding='utf-8') as en_f:
        for gz_path in tqdm(sorted(glob.glob(path))):
            with open(gz_path, "rb") as g, gzip.GzipFile(fileobj=g) as f:
                filename = os.path.basename(gz_path)
                for line_id, line in enumerate(f):
                    line = line.decode("utf-8")  # required for py3
                    if not line.strip():
                        continue
                    id_, unused_score, cs, en = line.split("\t")
                    block_match = re.match(re_block, id_)
                    if block_match and block_match.groups()[0] in bad_blocks:
                        continue
                    cs = cs.strip()
                    en = en.strip()
                    
                    cs_f.write(add_newline(cs))
                    en_f.write(add_newline(en))
                
if __name__ == '__main__':
    process_czeng()

import pickle
import io

DNA = {
    "[PAD]": 0,
    "[CLS]": 1,
    "A": 2,
    "C": 3,
    "T": 4,
    "G": 5
}

RNA = {
    "[PAD]": 0,
    "[CLS]": 1,
    "A": 2,
    "C": 3,
    "U": 4,
    "G": 5
}

protein = {'[PAD]': 0, "[CLS]": 1, 'B': 2, 'Q': 3, 'I': 4, 'D': 5, 'M': 6, 'V': 7, 'G': 8,
           'K': 9, 'Y': 10, 'P': 11, 'H': 12, 'Z': 13, 'W': 14, 'U': 15, 'A': 16, 'N': 17, 'F': 18, 'R': 19, 'S': 20,
           'C': 21, 'E': 22, 'L': 23, 'T': 24, 'X': 25}

if __name__ == '__main__':
    f = open("./DNAtoken2index.pkl", "wb")
    pickle.dump(DNA, f)
    f.close()

    f = open("./RNAtoken2index.pkl", "wb")
    pickle.dump(RNA, f)
    f.close()
    #
    f = open("./proteintoken2index.pkl", "wb")
    pickle.dump(protein, f)
    f.close()

    # protein = pickle.load(open('./proteintoken2index.pkl', 'rb'))
    # print(protein)

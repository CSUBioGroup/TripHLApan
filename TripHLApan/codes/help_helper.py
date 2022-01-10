
import re


def cut_pep_14(pep_seq):
    peplen = len(pep_seq)
    insert_idx = int((peplen + 1) / 2) - 1
    pseq_pep_seq = pep_seq[0:insert_idx] + 'X' * (14 - peplen) + pep_seq[insert_idx:]
    return pseq_pep_seq


def cut_allele_200(allele_seq):
    if len(allele_seq) >= 200:
        pseq_allele_seq = allele_seq[0:200]
    else:
        pseq_allele_seq = allele_seq + 'X' * (200 - len(allele_seq))
    return pseq_allele_seq


def get_AAfea_phy():
    AAfea_phy_dict = {}
    in_f = open('../assistant_codes/phy/AAfea_phy.txt', 'r')
    for line in in_f:
        cols = re.split('[\t\n]', line)
        if len(cols) < 5:
            continue
        if cols[0] == 'AA':
            continue
        for idx in range(2, len(cols)):
            if cols[idx] == '':
                continue
            if cols[1] not in AAfea_phy_dict.keys():
                AAfea_phy_dict[cols[1]] = []
            AAfea_phy_dict[cols[1]].append(float(cols[idx]))
    in_f.close()
    return AAfea_phy_dict


def blosum50():
    blosum50_matrix = []
    blosum50_file = './blosum50.txt'
    in_f = open(blosum50_file, 'r')
    for line in in_f:
        encoded_residue = []
        cols = re.split('[\t\n]', line)
        if len(cols) < 20:
            continue
        for i in range(20):
            encoded_residue.append(int(cols[i]))
        blosum50_matrix.append(encoded_residue)
    in_f.close()
    return blosum50_matrix

def blosum62():
    blosum62_matrix = []
    blosum62_file = './blosum62.txt'
    in_f = open(blosum62_file, 'r')
    for line in in_f:
        encoded_residue = []
        cols = re.split('[\t\n]', line)
        if len(cols) < 20:
            continue
        for i in range(20):
            encoded_residue.append(int(cols[i]))
        blosum62_matrix.append(encoded_residue)
    in_f.close()
    return blosum62_matrix


aa_embedding = {"C": 0, "S": 1, "T": 2, "P": 3, "A": 4, "G": 5, "N": 6, "D": 7, "E": 8, "Q": 9, "H": 10, "R": 11, "K": 12,
      "M": 13, "I": 14, "L": 15, "V": 16, "F": 17, "Y": 18, "W": 19, "X": 20}
def get_embedding():
    embedding_protein_dict = {}
    in_f = open('./embedding_protein.txt', 'r')
    matrix = []
    for line in in_f:
        line = re.split('[\t\n\[\] ]', line)
        while '' in line:
            line.remove('')
        line = list(map(float, line))
        matrix.append(line)
    in_f.close()
    for residue in aa_embedding.keys():
        embedding_protein_dict[residue] = matrix[aa_embedding[residue]]

    #print(embedding_protein_dict)
    return embedding_protein_dict
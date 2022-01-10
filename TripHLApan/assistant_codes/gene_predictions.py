import warnings
warnings.filterwarnings("ignore")
import os
import re

def gene_prediction(cmd, dir_pep_files, alleles_file, printouts_dir):
	alleles = []
	in_f = open(alleles_file, 'r')
	for line in in_f:
		cols = re.split('[\n]', line)
		alleles.append(cols[0])
	in_f.close()

	for allele in alleles:
		# HLA-A*01:01_9
		allele_split = re.split('[*_]', allele)
		allele1 = allele_split[0]
		allele2 = allele_split[1]
		pep_len = allele_split[2]
        
		if os.path.exists(printouts_dir + allele + '.txt'):
			continue

		os.system(cmd + ' -p ' + dir_pep_files + 'rand_L_' + pep_len + 'random.txt' + ' -BA -a ' + allele1 + allele2 + ' > ' + printouts_dir + allele + '.txt')




execute_cmd = '../../analyse_data20201119/tools/IEDB_MHC_I-3.1/mhc_i/method/netmhcpan-4.1-executable/netmhcpan_4_1_executable/netMHCpan'
dir_pep_files = '../random_data/'
alleles_file = './need_random_neg.txt'
printouts_dir = '../prediction_result/'
gene_prediction(execute_cmd, dir_pep_files, alleles_file, printouts_dir)

print("end")
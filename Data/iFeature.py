#!/usr/bin/env python
#_*_coding:utf-8_*_

import argparse
import re
from codes import *
import pdb
import pickle
import csv
import pandas as pd


if __name__ == '__main__':
	parser = argparse.ArgumentParser(usage="it's usage tip.",
									 description="Generating various numerical representation schemes for protein sequences")
	parser.add_argument("--file", required=True, help="input fasta file")
	parser.add_argument("--type", required=True,
						choices=['AAC', 'EAAC', 'CKSAAP', 'DPC', 'DDE', 'TPC', 'BINARY',
								 'GAAC', 'EGAAC', 'CKSAAGP', 'GDPC', 'GTPC',
								 'AAINDEX', 'ZSCALE', 'BLOSUM62',
								 'NMBroto', 'Moran', 'Geary',
								 'CTDC', 'CTDT', 'CTDD',
								 'CTriad', 'KSCTriad',
								 'SOCNumber', 'QSOrder',
								 'PAAC', 'APAAC',
								 'KNNprotein', 'KNNpeptide',
								 'PSSM', 'SSEC', 'SSEB', 'Disorder', 'DisorderC', 'DisorderB', 'ASA', 'TA'
								 ],
						help="the encoding type")
	parser.add_argument("--path", dest='filePath',
						help="data file path used for 'PSSM', 'SSEB(C)', 'Disorder(BC)', 'ASA' and 'TA' encodings")
	parser.add_argument("--train", dest='trainFile',
						help="training file in fasta format only used for 'KNNprotein' or 'KNNpeptide' encodings")
	parser.add_argument("--label", dest='labelFile',
						help="sample label file only used for 'KNNprotein' or 'KNNpeptide' encodings")
	parser.add_argument("--order", dest='order',
						choices=['alphabetically', 'polarity', 'sideChainVolume', 'userDefined'],
						help="output order for of Amino Acid Composition (i.e. AAC, EAAC, CKSAAP, DPC, DDE, TPC) descriptors")
	parser.add_argument("--userDefinedOrder", dest='userDefinedOrder',
						help="user defined output order for of Amino Acid Composition (i.e. AAC, EAAC, CKSAAP, DPC, DDE, TPC) descriptors")
	args = parser.parse_args()

	fastas = []
	# 使用csv模块打开并读取CSV文件
	with open(args.file, 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			# 将每一行添加到data列表中
			if row[0] == "Q16881":
				continue
			fastas.append(row)

	# pdb.set_trace()
	userDefinedOrder = args.userDefinedOrder if args.userDefinedOrder != None else 'ACDEFGHIKLMNPQRSTVWY'
	userDefinedOrder = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '', userDefinedOrder)
	if len(userDefinedOrder) != 20:
		userDefinedOrder = 'ACDEFGHIKLMNPQRSTVWY'
	myAAorder = {
		'alphabetically': 'ACDEFGHIKLMNPQRSTVWY',
		'polarity': 'DENKRQHSGTAPYVMCWIFL',
		'sideChainVolume': 'GASDPCTNEVHQILMKRFYW',
		'userDefined': userDefinedOrder
	}
	myOrder = myAAorder[args.order] if args.order != None else 'ACDEFGHIKLMNPQRSTVWY'
	kw = {'path': args.filePath, 'train': args.trainFile, 'label': args.labelFile, 'order': myOrder}

	myFun = args.type + '.' + args.type + '(fastas, **kw)'
	print('Descriptor type: ' + args.type)
	encodings = eval(myFun)
	# pdb.set_trace()
	out_filename = "protein/protein_" + args.type + ".pkl"
	# 初始化一个空字典来存储编码
	encoding_dict = {}
	for encoding in encodings:
		key = encoding[0]  # 第一个元素作为键
		value = encoding[1:]  # 剩下的元素作为值
		encoding_dict[key] = value
	# 将字典保存为Pickle格式
	with open(out_filename, 'wb') as outfile:
		pickle.dump(encoding_dict, outfile)


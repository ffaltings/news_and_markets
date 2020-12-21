import gzip

def iterate_html(file):
	"""
	Iterate over a compressed set of html documents
	:param file: path to .html.gz file
	:return: yields html doc as string
	"""
	doc = ""
	with gzip.open(file, 'rb') as f:
		for line in f:
			if "</html>" not in line.decode('utf-8'):
				doc += line.decode('utf-8')
	#         print(line.decode('utf-8')[:-2])
			else:
				doc += line.decode('utf-8')
				yield doc
				doc = ""


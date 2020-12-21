"""
Code to process Common Crawl data on Spark. Taken and adapted from https://github.com/commoncrawl/cc-pyspark.
"""

import os
import logging
import gzip
import json
import boto3
import botocore
import time

from datetime import datetime
from tempfile import TemporaryFile
from warcio.archiveiterator import ArchiveIterator
from warcio.recordloader import ArchiveLoadFailed
from io import BytesIO
from pyspark import SparkContext
from selectolax.parser import HTMLParser
from goose3 import Goose

from src.data.utils.pipeline import preprocess_pipeline
from src.data.utils.filter import *
from src.data.utils.transformation import *

LOGGING_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'

class CCSparkJob(object):
	"""
	Basic Spark Job for Processing Common Crawl Data
	"""

	name = 'CCSparkJob'
	local_temp_dir = "data/temp"

	# counters
	records_processed = None
	warc_input_processed = None
	warc_input_failed = None
	unicode_decode_errors = None
	log_level = 'INFO'
	logging.basicConfig(level=log_level, format=LOGGING_FORMAT)

	def __init__(self, input_file, out_dir, n_inputs = 10):
		self.input_path = input_file
		self.out_dir = out_dir
		timestamp = datetime.timestamp(datetime.now())
		self.timestamp = timestamp
		self.out_path = os.path.join(self.out_dir, str(timestamp) + "_" + self.name)
		self.n_inputs = n_inputs

	def init_logging(self, level=None):
		if level is None:
			level = self.log_level
		else:
			self.log_level = level
		logging.basicConfig(level=level, format=LOGGING_FORMAT)

	def init_accumulators(self, sc):
		self.records_processed = sc.accumulator(0)
		self.warc_input_processed = sc.accumulator(0)
		self.warc_input_failed = sc.accumulator(0)
		self.unicode_decode_errors = sc.accumulator(0)

	def get_logger(self, spark_context=None):
		if spark_context is None:
			return logging.getLogger(self.name)
		return spark_context._jvm.org.apache.log4j.LogManager \
			.getLogger(self.name)

	def init_env(self, sc):
		"""Init any other job variables that depend on the spark context"""

	def run(self):

		sc = SparkContext(
			appName=self.name
		)

		sc.setLogLevel(self.log_level)

		self.init_accumulators(sc)

		self.init_env(sc)

		self.get_logger(sc).info('Starting job run')

		try:
			self.run_job(sc)
		finally:
			sc.stop()

	def log_aggregator(self, sc, agg, descr):
		self.get_logger(sc).info(descr.format(agg.value))

	def save_aggregator(self, f, agg, descr):
		f.write(descr.format(agg.value))

	def log_aggregators(self, sc):
		self.log_aggregator(sc, self.warc_input_processed,
							'WARC/WAT/WET input files processed = {}')
		self.log_aggregator(sc, self.warc_input_failed,
							'WARC/WAT/WET input files failed = {}')
		self.log_aggregator(sc, self.records_processed,
							'WARC/WAT/WET records processed = {}')

		self.log_aggregator(sc, self.unicode_decode_errors,
							'Unicode decode errors = {}')

	def save_aggregators(self, agg_out):
		with open(agg_out, 'w') as f:
			self.save_aggregator(f, self.warc_input_processed,
								 'WARC/WAT/WET input files processed = {}\n')
			self.save_aggregator(f, self.warc_input_failed,
								'WARC/WAT/WET input files failed = {}\n')
			self.save_aggregator(f, self.records_processed,
								'WARC/WAT/WET records processed = {}\n')

			self.save_aggregator(f, self.unicode_decode_errors,
								'Unicode decode errors = {}\n')

	def run_job(self, sc):
		"""
		Executes spark job
		:param sc: spark context
		:return: returns a desired output (should not return anything in later impl)
		"""

		input_uris = sc.textFile(self.input_path, self.n_inputs) # only works on uncompressed files
		# input_uris = input_uris.repartition(3*sc.defaultParallelism)
		output = input_uris.mapPartitions(self.process_warcs) #TODO make partitions very small!

		output.saveAsTextFile(self.out_path,
							  compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

		self.log_aggregators(sc)
		# agg_out = os.path.join(self.out_dir, str(timestamp)+ "_" + self.name + "_aggregators.txt")
		# self.save_aggregators(agg_out)

		# # file_path = os.path.join(self.out_dir, "processed_files.json.gz")
		# with gzip.GzipFile("hdfs:///processed_files.json.gz", 'wb') as f: #!!
		# 	f.write(json_str.encode('utf-8'))


	def process_warcs(self, iterator):
		bucketname = 'commoncrawl'
		no_sign_request = botocore.client.Config(
			signature_version=botocore.UNSIGNED)
		s3client = boto3.client('s3', config=no_sign_request)

		for uri in iterator:
			warctemp = TemporaryFile(mode='w+b', dir=self.local_temp_dir)
			try:
				s3client.download_fileobj(bucketname, uri, warctemp)
			except botocore.client.ClientError as exception:
				print('Failed to download {}: {}'.format(uri, exception))
				warctemp.close()
			warctemp.seek(0)
			stream = warctemp

			try:
				archive_iterator = ArchiveIterator(stream)
				for i, res in enumerate(self.iterate_records(uri, archive_iterator)):
					# if i > 499: break
					if not isinstance(res, dict):
						raise("Process_record must return a dict object!")
					yield json.dumps(res) # convert dict to json string
			except ArchiveLoadFailed as exception:
				print('Invalid WARC: {} - {}'.format(uri, exception))
			finally:
				stream.close()

	def iterate_records(self, _warc_uri, archive_iterator):
		for i, record in enumerate(archive_iterator):
			# if i > 9: break

			for res in self.process_record(record):
				yield res
			self.records_processed.add(1)

	def process_record(self, record):
		"""
		Process an invidual record
		:param record: WARC record object
		:return: dict object
		"""
		raise NotImplementedError('Processing record needs to be customized')


class CCSparkJobStreaming(CCSparkJob):
	name = 'CCSparkJobStreaming'

	warc_parse_http_header = True

	def process_warcs(self, iterator):
		bucketname = 'commoncrawl'
		no_sign_request = botocore.client.Config(
			signature_version=botocore.UNSIGNED)
		s3client = boto3.client('s3', config=no_sign_request)

		for uri in iterator:
			self.warc_input_processed.add(1)
			uri = uri.split(' ')[-1]
			try:
				obj = s3client.get_object(Bucket=bucketname, Key=uri)
			except botocore.client.ClientError as exception:
				print('Failed to download {}: {}'.format(uri, exception))
				self.warc_input_failed.add(1)
				continue

			try:
				archive_iterator = ArchiveIterator(obj["Body"])
				for i,res in enumerate(self.iterate_records(uri, archive_iterator)):
					# if i > 1: break
					if not isinstance(res, dict):
						raise ("Process_record must return a dict object!")
					yield str(res) # json.dumps(res) <= fix that date in transformation
			except ArchiveLoadFailed as exception:
				print('Invalid WARC: {} - {}'.format(uri, exception))


class CCSparkJobSaveS3(CCSparkJob):
	name = 'CCSparkJobFilter'

	# extra accumulators
	records_kept = None

	def __init__(self, input_file, out_dir, s3_out_bucket, n_inputs=10):
		super().__init__(input_file, out_dir, n_inputs)
		self.s3_out_bucket = s3_out_bucket
		self.s3_out = "CCNewsProcessed/" + str(self.timestamp) + "/"

	def init_accumulators(self, sc):
		super().init_accumulators(sc)
		self.records_kept = sc.accumulator(0)

	def log_aggregators(self, sc):
		super().log_aggregators(sc)
		self.log_aggregator(sc, self.records_kept,
							'WARC/WAT/WET records kept = {}')

	def process_warcs(self, iterator):
		bucketname = 'commoncrawl'
		my_bucketname = self.s3_out_bucket
		# no_sign_request = botocore.client.Config(
		# 	signature_version=botocore.UNSIGNED)
		s3client = boto3.client('s3')

		for uri in iterator:
			self.warc_input_processed.add(1)
			# if self.warc_input_processed % 10 == 0:
			# 	print("WARC Files processed: {}".format(self.warc_input_processed))
			uri = uri.split(' ')[-1]
			self.get_logger().info('Reading form S3 {}'.format(uri))
			try:
				obj = s3client.get_object(Bucket=bucketname, Key=uri)
			except botocore.client.ClientError as exception:
				self.get_logger().error('Failed to download {}: {}'.format(uri, exception))
				self.warc_input_failed.add(1)
				continue

			try:
				archive_iterator = ArchiveIterator(obj["Body"])
			except ArchiveLoadFailed as exception:
				self.warc_input_failed.add(1)
				self.get_logger().error('Invalid WARC: {} - {}'.format(uri, exception))
				continue

			buffer = BytesIO()
			zip_buffer = gzip.GzipFile(mode='wb', fileobj=buffer)
			for i,res in enumerate(self.iterate_records(uri, archive_iterator)):
				# if i > 1:
				# 	break
				if not isinstance(res, dict):
					raise ("Process_record must return a dict object!")
				json_res = str(res) + "\n" # change to: json.dumps(res) + "\n"
				zip_buffer.write(json_res.encode('utf-8')) #add new lines between HTML docs
				self.records_kept.add(1)
			zip_buffer.close()
			buffer.seek(0)
			my_key = self.s3_out + uri.split('/')[-1].split('.')[0] + ".jsonl.gz"  # take last unique part of key
			self.get_logger().info('Saving to S3: {}'.format(my_key))
			try:
				s3_response = s3client.put_object(Body=buffer.getvalue(), Bucket=my_bucketname, Key=my_key)
			except botocore.client.ClientError as exception:
				print('Failed to upload {}: {}'.format(my_bucketname + '/' + my_key, exception))
				continue
			finally:
				buffer.close()

			s3_data = {'request': {'key': my_key, 'bucket': my_bucketname}, 'response': s3_response}

			# out_file = os.path.join(self.out_dir, my_key + '.txt.gz')
			# with open(out_file, 'wb') as f:
			# 	f.write(buffer.getvalue())

			yield s3_data


class ExampleSparkJob(CCSparkJobStreaming):

	def process_record(self, record):
		if record.rec_type != 'response':
			# WARC request or metadata records
			return
		content_type = record.http_headers.get_header('content-type', None)
		if content_type is None or 'html' not in content_type:
			# skip non-HTML or unknown content types
			return

		try:
			data = record.content_stream().read().decode('utf-8')
		except UnicodeDecodeError:
			self.unicode_decode_errors.add(1)
			return
		yield {"data": data}

class ExampleFilterSparkJob(CCSparkJobSaveS3):

	def process_record(self, record):
		if record.rec_type != 'response':
			# WARC request or metadata records
			return
		content_type = record.http_headers.get_header('content-type', None)
		if content_type is None or 'html' not in content_type:
			# skip non-HTML or unknown content types
			return

		try:
			data = record.content_stream().read().decode('utf-8')
		except UnicodeDecodeError:
			return
		yield data

class FilterJobFast(CCSparkJobSaveS3):
	name = "FilterJobFast"

	drop_kw = ['Facebook, Inc.', 'Twitter', 'Google', 'Adobe Systems Inc']
	# drop_kw = []
	pipeline_ = None

	def __init__(self, input, output_dir, s3_out_bucket, kw_path, n_inputs=10):
		super().__init__(input, output_dir, s3_out_bucket, n_inputs)
		self.kw_path = kw_path

	def init_env(self, sc):
		super().init_env(sc)

		input_list = [
			slx_prefilter(self.kw_path, self.drop_kw),
			html_parser(),
			language_filter(),
			date_filter(utc_check = 'simple'),
			keywordExtraction(self.kw_path, keyword_type='dict')
		]
		pipeline_ = preprocess_pipeline(input_list)
		self.pipeline_ = sc.broadcast(pipeline_)

		kw_filter_ = keywordExtraction(self.kw_path)
		self.kw_filter = sc.broadcast(kw_filter_)

		goose_ = Goose({'enable_image_fetching': False})
		self.goose = sc.broadcast(goose_)

	def process_record(self, record):
		if record.rec_type != 'response':
			# WARC request or metadata records
			return
		content_type = record.http_headers.get_header('content-type', None)
		if content_type is None or 'html' not in content_type:
			# skip non-HTML or unknown content types
			return

		try:
			data = record.content_stream().read().decode('utf-8')
		except UnicodeDecodeError:
			return

		try:
			output_data = self.pipeline_.value(data)
		except:
			return
		if output_data:
			yield output_data


if __name__ == "__main__":
	start_time = time.process_time()
	n_inputs_to_process = 1000
	job = FilterJobFast("s3://path/to/warc/paths/on/s3/",
						"hdfs:///path/to/output/folder/on/hdfs/",
						"s3_out_bucket_name",
						"path/to/sp500_list_25-02-2020.txt",
						n_inputs_to_process)
	run_time = time.process_time() - start_time
	print("Job took {} seconds to complete".format(run_time))
	job.run()
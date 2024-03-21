from abc import ABC, abstractmethod

class promptInterface(ABC) :

	@abstractmethod
	def __init__(self):
		pass
	   
	@abstractmethod
	def prepare_model(self):
		pass

	@abstractmethod
	def prepare_script(self):
		pass

	@abstractmethod
	def inference(self):
		pass 

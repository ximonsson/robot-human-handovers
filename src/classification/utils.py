"""
File: utils.py
Description: Utility function for printing information and such during runtime.
"""
import sys


def progressbar(done, total, length=30):
	"""
	Return a progress bar visualizing the how far we have progressed.

	:param done: integer - number between 0 and total of current step
	:param total: integer - total steps
	:param length: integer - length of the progress bar
	:returns: string
	"""
	progress = done / total
	bar = ""
	for _ in range(int(length * progress)):
		bar += "#"
	for _ in range(length-int(length * progress)):
		bar += "-"
	return "[{}] {}%".format(bar, int(progress * 100))


def print_step(msg, *args):
	print(("\r"+msg).format(*args), flush=True, end="")


def find_arg(key, default=None):
	"""
	Find argument with key in the supplied command line arguments.
	In case not found it returns a default value.
	"""
	key = "--{}=".format(key)
	if any(map(lambda s: s.startswith(key), sys.argv)):
		arg = next(arg for arg in sys.argv if arg.startswith(key))
		return arg.split("=")[1]
	return default

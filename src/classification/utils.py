"""
File: utils.py
Description: Utility function for printing information and such during runtime.
"""


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
	return "[{}] {}% Done".format(bar, int(progress * 100))


def print_step(msg, *args):
	print(("\r"+msg).format(*args), flush=True, end="")

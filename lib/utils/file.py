import os


def assert_path(path: str):
	if not os.path.exists(path):
		os.makedirs(path, exist_ok=True)

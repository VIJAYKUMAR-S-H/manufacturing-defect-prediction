import joblib


def save_object(obj, file_path):
    """Save any Python object using joblib."""
    joblib.dump(obj, file_path)


def load_object(file_path):
    """Load a saved Python object."""
    return joblib.load(file_path)
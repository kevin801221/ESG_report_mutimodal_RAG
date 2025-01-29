# setup/nltk_setup.py
import ssl
import nltk

def setup_nltk():
    """設置 NLTK 並下載必要的數據"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = [
        'punkt',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng',
        'punkt_tab',
        'popular'
    ]

    for resource in resources:
        try:
            nltk.download(resource)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")

if __name__ == "__main__":
    setup_nltk()
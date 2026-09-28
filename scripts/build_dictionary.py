from app.config import Settings
from app.dictionary import SignDictionary


if __name__ == "__main__":
    settings = Settings.load()
    dictionary = SignDictionary.scan(settings.dictionary_root, settings.video_root)
    dictionary.save_index(settings.artifact_root / "dictionary")
    print(f"Indexed {len(dictionary.entries)} sign glosses")


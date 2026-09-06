import inspect
from pathlib import Path

from embedding.model import BGEEmbeddingModel


ROOT_DIR = Path(__file__).resolve().parent.parent


def print_method(cls, name):
    print("\n" + "=" * 80)
    print(f"METHOD: {name}")
    print("=" * 80)

    try:
        method = getattr(cls, name)
    except AttributeError:
        print("NOT FOUND")
        return

    try:
        print(inspect.getsource(method))
    except Exception as e:
        print(f"Could not inspect source: {e}")

    try:
        print("\nSIGNATURE:")
        print(inspect.signature(method))
    except Exception as e:
        print(f"Could not inspect signature: {e}")


def main():
    print("=" * 80)
    print("E17 - EMBEDDING PIPELINE AUDIT")
    print("=" * 80)

    print(f"ROOT: {ROOT_DIR}")

    cls = BGEEmbeddingModel

    print("\nCLASS")
    print("-" * 80)
    print(cls)

    print("\nMODULE")
    print("-" * 80)
    print(inspect.getmodule(cls))

    print("\nCLASS SOURCE")
    print("-" * 80)

    try:
        print(inspect.getsource(cls))
    except Exception as e:
        print(f"Could not inspect class source: {e}")

    method_names = [
        "__init__",
        "embed",
        "embed_query",
        "embed_documents",
        "encode",
    ]

    for name in method_names:
        if hasattr(cls, name):
            print_method(cls, name)

    print("\n" + "=" * 80)
    print("E17 - INSTANCE AUDIT")
    print("=" * 80)

    try:
        model = cls()

        print("\nINSTANCE")
        print("-" * 80)
        print(model)

        print("\nINSTANCE DICT")
        print("-" * 80)

        try:
            for key, value in vars(model).items():
                print(f"{key}: {value}")
        except Exception as e:
            print(f"Could not inspect vars(model): {e}")

    except Exception as e:
        print(f"Failed to initialize model: {e}")

    print("\n" + "=" * 80)
    print("E17 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
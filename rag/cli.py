from rag.pipeline import answer


def main():

    query = input("Question: ")

    print()
    print("=" * 80)
    print("Answer")
    print("=" * 80)

    response = answer(query)

    print(response)


if __name__ == "__main__":
    main()